import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import importlib, sys

import numpy as np
from sklearn.model_selection import KFold

# For ROC/AUC:
try:
    from sklearn.metrics import roc_curve, roc_auc_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

from src.dataset import BrainTumorDataset
from src.transforms_config import create_transform
import importlib

def _models():
    """Return the current src.models module (re-imported if it changed)."""
    if "src.models" in sys.modules:                     # ← reload if cached
        importlib.reload(sys.modules["src.models"])
    return importlib.import_module("src.models")
# thin wrappers so the rest of the file stays unchanged
def get_model(*args, **kwargs):
    return _models().get_model(*args, **kwargs)

def SimpleCNN(*args, **kwargs):
    return _models().SimpleCNN(*args, **kwargs)
class Experiment:
    """
    An experiment pipeline that can do:
      - single-run train+eval
      - or k-fold cross-validation (folds>1).
    Also accumulates confusion matrices & class metrics for final reporting.

    The key fix is a single `_train_one_epoch()` method so HPC code can do:
      train_loss, train_acc, _ = exp._train_one_epoch()
    """

    def __init__(
        self,
        data_root="cleaned_data",
        model_variant="multi",
        input_size=128,
        batch_size=8,
        lr=1e-3,
        epochs=1,
        disable_augment=False,
        adv_aug=False,
        strong_aug=False,
        dropout=0.0,
        weight_decay=0.0,
        earlystop_patience=0,
        use_class_weight=False,
        device="cpu",
        normalization="none",
        folds=1
    ):
        self.data_root = data_root
        self.model_variant = model_variant
        self.input_size = input_size
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.disable_augment = disable_augment
        self.adv_aug = adv_aug
        self.strong_aug = strong_aug
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.earlystop_patience = earlystop_patience
        self.use_class_weight = use_class_weight
        self.folds = folds
        self.device = torch.device(device)

        self.log_func = print  # default logger

        # Build transformations (no intensity normalization logic here by default):
        self.transform = create_transform(
            image_size=(input_size, input_size),
            augment=(not disable_augment),
            adv_aug=self.adv_aug,
            strong_aug=self.strong_aug
        )

        # Build model (placeholder for final #classes logic below)
        self.model, self.num_classes = get_model(
            model_variant=self.model_variant,
            input_size=self.input_size,
            dropout=self.dropout
        )
        self.model.to(self.device)

        self.optimizer = None
        # single-head criterion
        self.criterion = None
        # double variant => two separate losses
        self.criterion_bin = None
        self.criterion_multi = None

        # final results
        self.final_roc = {"fpr": None, "tpr": None, "auc": None}
        self.final_confmat = None
        self.final_class_metrics = {}

    def set_logger(self, log_callable):
        self.log_func = log_callable

    def load_data(self):
        """
        Loads Training + Testing sets for single-run (folds=1),
        or just the Training set for k-fold (folds>1).
        Then detect how many classes. If 'multi' or 'simple', we rebuild the model to match that #classes.
        """
        if self.folds > 1:
            self.log_func("Loading entire Training folder for k-fold CV.")
            self.train_dataset = BrainTumorDataset(
                root_dir=self.data_root,
                subset="Training",
                transform=self.transform,
                model_variant=self.model_variant
            )
            self.test_dataset = None
        else:
            # Normal single-run
            train_set = BrainTumorDataset(
                root_dir=self.data_root,
                subset="Training",
                transform=self.transform,
                model_variant=self.model_variant
            )
            val_set = BrainTumorDataset(
                root_dir=self.data_root,
                subset="Testing",
                transform=self.transform,
                model_variant=self.model_variant
            )
            self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
            self.val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
            self.train_dataset = train_set
            self.test_dataset = val_set

        # Detect number of classes from the variant
        if self.model_variant == "binary":
            real_num_classes = 2
        elif self.model_variant == "double":
            real_num_classes = None  # double has separate heads
        elif self.model_variant == "threeclass":
            real_num_classes = 3
        else:
            # "multi" or "simple" => read from dataset
            real_num_classes = len(self.train_dataset.classes)

        self.log_func(f"[load_data] Using model_variant='{self.model_variant}'. Detected {real_num_classes} class(es).")

        # If "multi"/"simple", rebuild a single CNN with that # classes
        if self.model_variant in ["multi", "simple"]:
            self.log_func(f"Rebuilding single-head model with {real_num_classes} classes.")
            self.model = SimpleCNN(
                num_classes=real_num_classes,
                input_size=self.input_size,
                dropout=self.dropout
            ).to(self.device)
            self.num_classes = real_num_classes

    def _setup_optimizer_criterion(self):
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        if self.model_variant == "double":
            # define 2 losses
            self.criterion_bin = nn.CrossEntropyLoss()
            self.criterion_multi = nn.CrossEntropyLoss()
            return

        # single-head
        if self.use_class_weight:
            # build class weights from train_dataset freq
            freq = [0]*self.num_classes
            for _, lbl in self.train_dataset.samples:
                freq[lbl] += 1
            total_count = float(sum(freq))
            weights = []
            for c in range(self.num_classes):
                if freq[c] > 0:
                    weights.append(total_count / freq[c])
                else:
                    weights.append(0.0)
            weights_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        else:
            self.criterion = nn.CrossEntropyLoss()

    # ----------------------------------------------------------------
    # HPC partial training calls `_train_one_epoch()`
    # ----------------------------------------------------------------
    def _train_one_epoch(self):
        """
        Single epoch training for HPC partial.
        If double => train both heads in the same pass (binary + multi).
        Otherwise => normal single-head cross-entropy.

        Returns (epoch_loss, epoch_acc, best_batch_acc).
        We'll measure epoch_acc from the *binary classification* if "double".
        """
        self.model.train()
        total_loss = 0.0
        total_correct_bin = 0
        total_samples = 0
        best_batch_acc = 0.0

        if not hasattr(self, 'train_loader'):
            # if HPC tries partial training but we haven't loaded sets, do a fallback
            return 0.0, 0.0, 0.0

        if self.model_variant == "double":
            # Train both heads in a single pass
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Binary => label=0 => no tumor, else => tumor
                bin_labels = (labels != 0).long()

                self.optimizer.zero_grad()

                # forward both heads
                out_bin = self.model(images, mode='binary')
                out_multi = self.model(images, mode='threeclass')

                # binary loss
                loss_bin = self.criterion_bin(out_bin, bin_labels)

                # multi => only for tumor images
                tumor_mask = (labels != 0)
                if torch.any(tumor_mask):
                    multi_labels = labels[tumor_mask] - 1  # shift [1->0,2->1,3->2]
                    multi_out = out_multi[tumor_mask]
                    loss_multi = self.criterion_multi(multi_out, multi_labels)
                else:
                    loss_multi = 0.0

                loss = loss_bin + loss_multi
                loss.backward()
                self.optimizer.step()

                bs = images.size(0)
                total_loss += float(loss) * bs
                total_samples += bs

                # measure binary accuracy
                _, bin_preds = torch.max(out_bin, dim=1)
                batch_correct = (bin_preds == bin_labels).sum().item()
                total_correct_bin += batch_correct

                batch_acc = 100.0 * batch_correct / bs
                if batch_acc > best_batch_acc:
                    best_batch_acc = batch_acc

            epoch_loss = total_loss / total_samples if total_samples else 0.0
            epoch_acc = 100.0 * total_correct_bin / total_samples if total_samples else 0.0
            return epoch_loss, epoch_acc, best_batch_acc

        else:
            # single-head
            total_correct = 0
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                bs = images.size(0)
                total_loss += float(loss) * bs
                total_samples += bs

                preds = logits.argmax(dim=1)
                batch_correct = (preds == labels).sum().item()
                total_correct += batch_correct

                batch_acc = 100.0 * batch_correct / bs
                if batch_acc > best_batch_acc:
                    best_batch_acc = batch_acc

            epoch_loss = total_loss / total_samples if total_samples else 0.0
            epoch_acc = 100.0 * total_correct / total_samples if total_samples else 0.0
            return epoch_loss, epoch_acc, best_batch_acc

    # ----------------------------------------------------------------
    # HPC partial code also calls `_evaluate()` after each epoch
    # ----------------------------------------------------------------
    def _evaluate(self):
        """
        Evaluate on `self.val_loader`. We'll do a 2x2 confusion matrix for "double"
        (the binary head). For single-head variants, we do a normal confusion matrix 
        sized to #classes.

        Returns a dict with "accuracy", "confusion_matrix", and optional "roc_*" keys.
        """
        if not hasattr(self, 'val_loader'):
            return {"accuracy":0.0, "confusion_matrix":None}

        self.model.eval()
        total_correct = 0
        total_samples = 0

        # confmat size
        if self.model_variant == "binary":
            confmat_dim = 2
        elif self.model_variant == "double":
            confmat_dim = 2  # we measure binary head's confmat
        elif self.model_variant == "threeclass":
            confmat_dim = 3
        else:
            # "multi"/"simple"
            confmat_dim = self.num_classes

        confmat = torch.zeros(confmat_dim, confmat_dim, dtype=torch.int32, device=self.device)
        all_labels = []
        all_scores = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                if self.model_variant == "double":
                    # evaluate binary head
                    bin_labels = (labels != 0).long()
                    out_bin = self.model(images, mode='binary')
                    probs_bin = nn.functional.softmax(out_bin, dim=1)
                    _, bin_preds = torch.max(out_bin, dim=1)

                    batch_correct = (bin_preds == bin_labels).sum().item()
                    total_correct += batch_correct
                    total_samples += labels.size(0)

                    for t,p in zip(bin_labels, bin_preds):
                        confmat[t,p] += 1

                    all_labels.extend(bin_labels.cpu().numpy().tolist())
                    all_scores.extend(probs_bin[:,1].cpu().numpy().tolist())

                else:
                    # single-head
                    logits = self.model(images)
                    probs = nn.functional.softmax(logits, dim=1)
                    preds = logits.argmax(dim=1)

                    batch_correct = (preds==labels).sum().item()
                    total_correct += batch_correct
                    total_samples += labels.size(0)

                    for t,p in zip(labels, preds):
                        confmat[t,p]+=1

                    if self.model_variant=="binary":
                        all_labels.extend(labels.cpu().numpy().tolist())
                        all_scores.extend(probs[:,1].cpu().numpy().tolist())

        accuracy = 100.0 * total_correct / total_samples if total_samples>0 else 0.0
        metrics = {
            "accuracy": accuracy,
            "confusion_matrix": confmat
        }
        if (self.model_variant in ["binary","double"]) and _HAS_SKLEARN:
            try:
                fpr, tpr, _ = roc_curve(all_labels, all_scores)
                auc_val = roc_auc_score(all_labels, all_scores)
                metrics["roc_fpr"] = fpr
                metrics["roc_tpr"] = tpr
                metrics["roc_auc"] = auc_val
            except ValueError:
                metrics["roc_auc"] = None
        return metrics

    def _extract_roc_data(self, val_metrics):
        if "roc_fpr" in val_metrics:
            return {
                "fpr": val_metrics["roc_fpr"],
                "tpr": val_metrics["roc_tpr"],
                "auc": val_metrics.get("roc_auc", None)
            }
        else:
            return {"fpr":None,"tpr":None,"auc":None}

    def _compute_class_metrics(self, val_metrics):
        if val_metrics["confusion_matrix"] is None:
            return {}
        cmat = val_metrics["confusion_matrix"].detach().cpu().numpy().astype(float)
        numc = cmat.shape[0]
        precision, recall, f1_vals, support = [], [], [], []

        for i in range(numc):
            row_sum = cmat[i,:].sum()
            col_sum = cmat[:,i].sum()
            tp = cmat[i,i]
            prec = tp/col_sum if col_sum>0 else 0.0
            rec = tp/row_sum if row_sum>0 else 0.0
            f1 = 2*prec*rec/(prec+rec) if (prec+rec)>1e-9 else 0.0
            precision.append(prec)
            recall.append(rec)
            f1_vals.append(f1)
            support.append(int(row_sum))

        total_tp = np.trace(cmat)
        total_sum = cmat.sum()
        micro_f1 = total_tp/total_sum if total_sum>0 else 0.0
        macro_f1 = float(np.mean(f1_vals))

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1_vals,
            "support": support,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1
        }

    # ----------------------------------------------------------------
    # K-Fold code (if needed)
    # ----------------------------------------------------------------
    def run_kfold(self, folds=5, epochs=5, progress_cb=None):
        """
        If folds>1, we only loaded 'Training' as self.train_dataset.
        We'll do a simpler approach, using `_train_one_epoch_fold()` for the fold logic,
        focusing on single-head or the binary part of 'double'. Adjust if needed.
        """
        from sklearn.model_selection import KFold
        n_samples = len(self.train_dataset)
        idx_arr = np.arange(n_samples)
        kf = KFold(n_splits=folds, shuffle=True, random_state=42)

        accum_confmat = None
        sum_metrics = None
        total_acc = 0.0

        for fold_i, (tr_idx, val_idx) in enumerate(kf.split(idx_arr)):
            self.log_func(f"\n--- Fold {fold_i+1}/{folds} ---")
            # re-init model each fold
            self.model, _ = get_model(
                model_variant=self.model_variant,
                input_size=self.input_size,
                dropout=self.dropout
            )
            self.model.to(self.device)
            self._setup_optimizer_criterion()

            from torch.utils.data import Subset
            tr_ds = Subset(self.train_dataset, tr_idx)
            val_ds = Subset(self.train_dataset, val_idx)
            tr_loader = DataLoader(tr_ds, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

            best_val = 0.0
            no_improve = 0
            for ep in range(epochs):
                tr_loss, tr_acc, _ = self._train_one_epoch_fold(tr_loader)
                val_metrics = self._evaluate_fold(val_loader)
                val_acc = val_metrics["accuracy"]

                msg = f"[Fold {fold_i+1}, Ep {ep+1}/{epochs}] TrainAcc={tr_acc:.1f}%, ValAcc={val_acc:.1f}%"
                self.log_func(msg)

                if val_acc>best_val:
                    best_val = val_acc
                    no_improve = 0
                    best_confmat = val_metrics["confusion_matrix"]
                    best_mets = self._compute_class_metrics(val_metrics)
                else:
                    no_improve += 1
                    if self.earlystop_patience>0 and no_improve>=self.earlystop_patience:
                        self.log_func(f"[EarlyStop fold {fold_i+1} at ep {ep+1}]")
                        break

            total_acc += best_val
            if accum_confmat is None:
                accum_confmat = best_confmat.clone()
            else:
                accum_confmat += best_confmat

            if sum_metrics is None:
                sum_metrics = {}
                for k, v in best_mets.items():
                    sum_metrics[k] = np.array(v, dtype=float)
            else:
                for k, v in best_mets.items():
                    sum_metrics[k] += np.array(v, dtype=float)

        avg_acc = total_acc/folds
        final_metrics = {}
        if sum_metrics:
            for k, arr_val in sum_metrics.items():
                final_metrics[k] = (arr_val/folds).tolist()

        self.final_confmat = accum_confmat
        # compute micro/macro from accum
        final_val_metrics = {"accuracy": avg_acc, "confusion_matrix": accum_confmat}
        extra = self._compute_class_metrics(final_val_metrics)
        for kk, vv in extra.items():
            final_metrics[kk] = vv

        return avg_acc, accum_confmat, final_metrics, None

    def _train_one_epoch_fold(self, loader):
        """
        Minimal k-fold training epoch on a custom loader. 
        We'll do the same logic as `_train_one_epoch`, but ignoring multi-head for "double"
        (or just training the binary head). Adjust if needed.
        Returns (epoch_loss, epoch_acc, best_batch_acc).
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        best_batch_acc = 0.0

        if self.model_variant == "double":
            # e.g. train just the binary head in folds
            for images, labels in loader:
                images = images.to(self.device)
                bin_labels = (labels!=0).long()

                self.optimizer.zero_grad()
                out_bin = self.model(images, mode='binary')
                loss_bin = self.criterion_bin(out_bin, bin_labels)
                loss_bin.backward()
                self.optimizer.step()

                bs = images.size(0)
                total_loss += float(loss_bin)*bs
                total_samples += bs

                preds = out_bin.argmax(dim=1)
                batch_correct = (preds==bin_labels).sum().item()
                total_correct += batch_correct

                batch_acc = 100.0*batch_correct/bs
                if batch_acc>best_batch_acc:
                    best_batch_acc = batch_acc

        else:
            # single-head
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                bs = images.size(0)
                total_loss += float(loss)*bs
                preds = outputs.argmax(dim=1)
                batch_correct = (preds==labels).sum().item()
                total_correct += batch_correct
                total_samples += bs

                batch_acc = 100.0*batch_correct/bs
                if batch_acc>best_batch_acc:
                    best_batch_acc = batch_acc

        epoch_loss = total_loss / total_samples if total_samples else 0.0
        epoch_acc = 100.0 * total_correct / total_samples if total_samples else 0.0
        return epoch_loss, epoch_acc, best_batch_acc

    def _evaluate_fold(self, loader):
        """
        Evaluate on the fold's loader. 
        We'll do a binary confmat for 'double' or 'binary',
        else a single-head confmat sized to #classes.
        Returns a dict with "accuracy", "confusion_matrix", plus optional ROC.
        """
        self.model.eval()
        total_correct = 0
        total_samples = 0

        if self.model_variant=="double" or self.model_variant=="binary":
            confmat_dim = 2
        else:
            confmat_dim = self.num_classes

        confmat = torch.zeros(confmat_dim, confmat_dim, dtype=torch.int32, device=self.device)
        all_labels, all_scores = [], []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                if self.model_variant=="double":
                    bin_labels = (labels!=0).long()
                    out_bin = self.model(images, mode='binary')
                    preds_bin = out_bin.argmax(dim=1)
                    probs_bin = nn.functional.softmax(out_bin, dim=1)

                    bs = images.size(0)
                    total_samples += bs
                    total_correct += (preds_bin==bin_labels).sum().item()

                    for t,p in zip(bin_labels, preds_bin):
                        confmat[t,p]+=1

                    all_labels.extend(bin_labels.cpu().numpy().tolist())
                    all_scores.extend(probs_bin[:,1].cpu().numpy().tolist())

                else:
                    logits = self.model(images)
                    preds = logits.argmax(dim=1)
                    probs = nn.functional.softmax(logits, dim=1)

                    bs = images.size(0)
                    total_samples += bs
                    total_correct += (preds==labels).sum().item()

                    for t,p in zip(labels, preds):
                        confmat[t,p]+=1

                    if self.model_variant=="binary":
                        all_labels.extend(labels.cpu().numpy().tolist())
                        all_scores.extend(probs[:,1].cpu().numpy().tolist())

        acc = 100.0 * total_correct / total_samples if total_samples else 0.0
        metrics = {
            "accuracy": acc,
            "confusion_matrix": confmat
        }
        if (self.model_variant in ["binary","double"]) and _HAS_SKLEARN:
            try:
                fpr, tpr, _ = roc_curve(all_labels, all_scores)
                aucv = roc_auc_score(all_labels, all_scores)
                metrics["roc_fpr"] = fpr
                metrics["roc_tpr"] = tpr
                metrics["roc_auc"] = aucv
            except ValueError:
                metrics["roc_auc"] = None
        return metrics
