# File: src/experiment.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

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
from src.models import get_model

class Experiment:
    """
    An experiment pipeline that can do:
      - single-run train+eval
      - or k-fold cross-validation (folds>1)
    Also accumulates confusion matrices & class metrics for final reporting.
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

        # Build transformations
        self.transform = create_transform(
            image_size=(input_size, input_size),
            augment=(not disable_augment),
            adv_aug=self.adv_aug,
            strong_aug=self.strong_aug
        )

        # We'll create a placeholder model now. We'll finalize the correct num_classes after data is loaded
        # For "binary" or "multi" or "double", we do normal approach
        # If "transfer" or "custom", we do #TODO placeholders:
        if model_variant in ["transfer", "custom"]:
            # Just raise for now
            raise NotImplementedError(f"'{model_variant}' => TODO: implement your advanced architecture.")
        else:
            self.model, self.num_classes = get_model(
                model_variant=self.model_variant,
                input_size=self.input_size,
                dropout=self.dropout
            )
            self.model.to(self.device)

        self.optimizer = None
        self.criterion = None

        # final results
        self.final_roc = {"fpr": None, "tpr": None, "auc": None}
        self.final_confmat = None
        self.final_class_metrics = {}

    def set_logger(self, log_callable):
        self.log_func = log_callable

    def load_data(self):
        """
        Loads Training + Testing sets (for single-run approach).
        If folds>1, we only use the 'Training' set for k-fold, ignoring 'Testing'.
        Then detect how many classes truly exist in the train set.
        If 'multi', rebuild the model to match that class count (unless double).
        """
        if self.folds > 1:
            self.log_func("Loading entire Training folder for k-fold CV.")
            self.train_dataset = BrainTumorDataset(
                root_dir=self.data_root,
                subset="Training",
                transform=self.transform,
                model_variant=self.model_variant  # pass so it can unify classes if needed
            )
            self.test_dataset = None
        else:
            # Normal approach
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
            self.train_loader = DataLoader(
                train_set, batch_size=self.batch_size, shuffle=True
            )
            self.val_loader = DataLoader(
                val_set, batch_size=self.batch_size, shuffle=False
            )
            self.train_dataset = train_set
            self.test_dataset = val_set

        # how many classes are we effectively using
        if self.model_variant == "binary":
            real_num_classes = 2
        elif self.model_variant == "double":
            real_num_classes = None  # double => separate heads
        else:
            # multi
            real_num_classes = len(self.train_dataset.classes)

        self.log_func(f"[load_data] Using model_variant='{self.model_variant}'. Detected {real_num_classes} class(es).")

        # If "multi", we might rebuild the model to match real_num_classes
        if self.model_variant == "multi":
            self.log_func(f"Rebuilding 'multi' model with {real_num_classes} output classes.")
            from src.models import SimpleCNN
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

        if self.model_variant == "transfer" or self.model_variant == "custom":
            raise NotImplementedError(f"Model variant '{self.model_variant}' => TODO advanced architecture.")

        if self.model_variant == "double":
            # separate heads
            self.criterion_bin = nn.CrossEntropyLoss()
            self.criterion_multi = nn.CrossEntropyLoss()
            return

        # single-head
        if self.model_variant == "binary":
            # we only have 2 classes
            pass

        if self.use_class_weight and self.model_variant != "double":
            freq = [0] * self.num_classes
            for _, label in self.train_dataset.samples:
                freq[label] += 1
            total_count = float(sum(freq))
            weights = []
            for i in range(self.num_classes):
                if freq[i] > 0:
                    weights.append(total_count / freq[i])
                else:
                    weights.append(0.0)
            class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Single-run training
    # ------------------------------------------------------------------
    def run_single_training(self, epochs=5, progress_cb=None):
        self._setup_optimizer_criterion()
        all_train_acc = []
        all_val_acc = []

        best_val_acc = 0.0
        no_improve_count = 0

        for ep in range(epochs):
            train_loss, train_acc, _ = self._train_one_epoch()
            val_metrics = self._evaluate()
            val_acc = val_metrics["accuracy"]

            all_train_acc.append(train_acc)
            all_val_acc.append(val_acc)

            if progress_cb:
                pct = int(100 * (ep+1) / epochs)
                progress_cb.emit(pct)

            msg = f"Epoch {ep+1}/{epochs} | TrainAcc={train_acc:.2f}% | ValAcc={val_acc:.2f}%"
            self.log_func(msg)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve_count = 0
                self.final_roc = self._extract_roc_data(val_metrics)
                self.final_confmat = val_metrics["confusion_matrix"]
                self.final_class_metrics = self._compute_class_metrics(val_metrics)
            else:
                no_improve_count += 1

            if self.earlystop_patience>0 and no_improve_count>=self.earlystop_patience:
                self.log_func(f"[EarlyStopping triggered at epoch {ep+1}]")
                break

        return best_val_acc, all_train_acc, all_val_acc

    def _train_one_epoch(self):
        self.model.train()
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        best_batch_acc = 0.0

        if self.model_variant == "double":
            # Basic example: we train only the binary head. Or you can do more advanced.
            # TODO: properly train both heads if you prefer.
            for (images, labels) in self.train_loader:
                images = images.to(self.device)
                # Convert 4-class label into binary => 0 if label=notumor, else 1
                bin_labels = torch.where(labels==0, torch.tensor(0, device=self.device), torch.tensor(1, device=self.device))

                self.optimizer.zero_grad()
                out_bin = self.model(images, mode='binary')
                loss = self.criterion_bin(out_bin, bin_labels)
                loss.backward()
                self.optimizer.step()

                total_loss += float(loss) * images.size(0)
                _, bin_preds = torch.max(out_bin, 1)
                batch_correct = (bin_preds == bin_labels).sum().item()
                total_correct += batch_correct
                total_samples += labels.size(0)

                batch_acc = 100.0 * batch_correct / labels.size(0)
                if batch_acc > best_batch_acc:
                    best_batch_acc = batch_acc

        else:
            for (images, labels) in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += float(loss) * images.size(0)
                _, preds = torch.max(outputs, 1)
                batch_correct = (preds == labels).sum().item()
                total_correct += batch_correct
                total_samples += labels.size(0)

                batch_acc = 100.0 * batch_correct / labels.size(0)
                if batch_acc > best_batch_acc:
                    best_batch_acc = batch_acc

        epoch_loss = (total_loss / total_samples) if total_samples else 0.0
        epoch_acc = 100.0 * total_correct / total_samples if total_samples else 0.0
        return epoch_loss, epoch_acc, best_batch_acc

    def _evaluate(self):
        self.model.eval()
        total_correct = 0
        total_samples = 0

        if self.model_variant == "binary":
            confmat_dim = 2
        elif self.model_variant == "multi":
            confmat_dim = self.num_classes
        else:  # double => we do binary head for val
            confmat_dim = 2

        confmat = torch.zeros(confmat_dim, confmat_dim, dtype=torch.int32, device=self.device)
        all_labels = []
        all_scores = []

        with torch.no_grad():
            for (images, labels) in self.val_loader:
                images = images.to(self.device)

                if self.model_variant == "double":
                    # Evaluate only binary head
                    bin_labels = torch.where(labels==0, torch.tensor(0, device=self.device), torch.tensor(1, device=self.device))
                    out_bin = self.model(images, mode='binary')
                    probs_bin = nn.functional.softmax(out_bin, dim=1)
                    _, bin_preds = torch.max(out_bin, 1)

                    batch_correct = (bin_preds == bin_labels).sum().item()
                    total_correct += batch_correct
                    total_samples += labels.size(0)

                    for t, p in zip(bin_labels, bin_preds):
                        confmat[t, p] += 1

                    all_labels.extend(bin_labels.cpu().numpy().tolist())
                    all_scores.extend(probs_bin[:,1].cpu().numpy().tolist())

                else:
                    labels = labels.to(self.device)
                    outputs = self.model(images)
                    probs = nn.functional.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)

                    batch_correct = (preds == labels).sum().item()
                    total_correct += batch_correct
                    total_samples += labels.size(0)

                    for t, p in zip(labels, preds):
                        confmat[t, p] += 1

                    if self.model_variant == "binary":
                        # Save for ROC
                        all_labels.extend(labels.cpu().numpy().tolist())
                        all_scores.extend(probs[:,1].cpu().numpy().tolist())

        accuracy = 100.0 * total_correct / total_samples if total_samples else 0.0

        # class-wise accuracy
        class_acc = {}
        for c in range(confmat_dim):
            row_sum = confmat[c,:].sum().item()
            correct = confmat[c,c].item()
            class_acc[c] = 100.0 * correct / row_sum if row_sum>0 else 0.0

        metrics = {
            "accuracy": accuracy,
            "confusion_matrix": confmat,
            "class_accuracy": class_acc
        }
        if (self.model_variant in ["binary","double"]) and _HAS_SKLEARN:
            from sklearn.metrics import roc_curve, roc_auc_score
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
            return {"fpr": None, "tpr": None, "auc": None}

    def _compute_class_metrics(self, val_metrics):
        confmat = val_metrics["confusion_matrix"].clone().detach().cpu().numpy().astype(float)
        num_classes = confmat.shape[0]

        precision = []
        recall = []
        f1_vals = []
        support = []

        for c in range(num_classes):
            row_sum = confmat[c,:].sum()
            col_sum = confmat[:,c].sum()
            tp = confmat[c,c]
            prec = tp/col_sum if col_sum>0 else 0.0
            rec = tp/row_sum if row_sum>0 else 0.0
            f1 = 2*prec*rec/(prec+rec) if (prec+rec)>1e-9 else 0.0
            precision.append(prec)
            recall.append(rec)
            f1_vals.append(f1)
            support.append(int(row_sum))

        total_tp = np.trace(confmat)
        total_sum = confmat.sum()
        micro_f1 = (total_tp/total_sum) if total_sum>0 else 0.0
        macro_f1 = float(np.mean(f1_vals))

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1_vals,
            "support": support,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1
        }

    def run_kfold(self, folds=5, epochs=5, progress_cb=None):
        if not hasattr(self, 'train_dataset'):
            raise ValueError("Call load_data() first with folds>1")

        from sklearn.model_selection import KFold
        n_samples = len(self.train_dataset)
        idx_arr = np.arange(n_samples)
        kf = KFold(n_splits=folds, shuffle=True, random_state=42)

        total_acc = 0.0
        accum_confmat = None
        sum_metrics = None

        train_accs_over_folds = []
        val_accs_over_folds = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(idx_arr)):
            self.log_func(f"\n--- Fold {fold_idx+1}/{folds} ---")

            # Re-init model
            if self.model_variant in ["transfer","custom"]:
                raise NotImplementedError(f"{self.model_variant} => TODO advanced model")

            self.model, _ = get_model(
                model_variant=self.model_variant,
                input_size=self.input_size,
                dropout=self.dropout
            )
            self.model.to(self.device)
            self._setup_optimizer_criterion()

            fold_train_ds = Subset(self.train_dataset, train_idx)
            fold_val_ds = Subset(self.train_dataset, val_idx)

            fold_train_loader = DataLoader(fold_train_ds, batch_size=self.batch_size, shuffle=True)
            fold_val_loader = DataLoader(fold_val_ds, batch_size=self.batch_size, shuffle=False)

            best_val_acc = 0.0
            no_improve_count = 0
            local_train_accs = []
            local_val_accs = []

            for ep in range(epochs):
                tr_loss, tr_acc, _ = self._train_one_epoch_fold(fold_train_loader)
                val_metrics = self._evaluate_fold(fold_val_loader)
                val_acc = val_metrics["accuracy"]
                local_train_accs.append(tr_acc)
                local_val_accs.append(val_acc)

                msg = f"[Fold {fold_idx+1} Ep {ep+1}/{epochs}] TrainAcc={tr_acc:.2f}% | ValAcc={val_acc:.2f}%"
                self.log_func(msg)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    no_improve_count = 0
                    self.final_roc = self._extract_roc_data(val_metrics)
                    best_confmat = val_metrics["confusion_matrix"]
                    best_metrics = self._compute_class_metrics(val_metrics)
                else:
                    no_improve_count += 1

                if self.earlystop_patience>0 and no_improve_count>=self.earlystop_patience:
                    self.log_func(f"[EarlyStop fold {fold_idx+1} at ep {ep+1}]")
                    break

            total_acc += best_val_acc

            if accum_confmat is None:
                accum_confmat = best_confmat.clone()
            else:
                accum_confmat += best_confmat

            if sum_metrics is None:
                sum_metrics = {}
                for k, v in best_metrics.items():
                    sum_metrics[k] = np.array(v, dtype=float)
            else:
                for k, v in best_metrics.items():
                    sum_metrics[k] += np.array(v, dtype=float)

            train_accs_over_folds.append(local_train_accs)
            val_accs_over_folds.append(local_val_accs)

        avg_acc = total_acc / folds
        final_metrics = {}
        for k, arr_val in sum_metrics.items():
            if isinstance(arr_val, np.ndarray):
                final_metrics[k] = (arr_val / folds).tolist()
            else:
                final_metrics[k] = arr_val / folds

        self.final_confmat = accum_confmat
        final_val_metrics = {
            "confusion_matrix": accum_confmat,
            "accuracy": avg_acc
        }
        extra_metrics = self._compute_class_metrics(final_val_metrics)
        for kk, vv in extra_metrics.items():
            final_metrics[kk] = vv if not isinstance(vv, list) else vv

        return avg_acc, accum_confmat, final_metrics, (train_accs_over_folds, val_accs_over_folds)

    def _train_one_epoch_fold(self, loader):
        self.model.train()
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        best_batch_acc = 0.0

        if self.model_variant == "double":
            # placeholder training for bin head
            for (images, labels) in loader:
                images = images.to(self.device)
                bin_labels = torch.where(labels==0, torch.tensor(0, device=self.device), torch.tensor(1, device=self.device))

                self.optimizer.zero_grad()
                out_bin = self.model(images, mode='binary')
                loss = self.criterion_bin(out_bin, bin_labels)
                loss.backward()
                self.optimizer.step()

                total_loss += float(loss) * images.size(0)
                _, bin_preds = torch.max(out_bin, 1)
                batch_correct = (bin_preds == bin_labels).sum().item()
                total_correct += batch_correct
                total_samples += labels.size(0)
        else:
            for (images, labels) in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += float(loss) * images.size(0)
                _, preds = torch.max(outputs, 1)
                total_correct += (preds==labels).sum().item()
                total_samples += labels.size(0)

        epoch_loss = (total_loss / total_samples) if total_samples else 0.0
        epoch_acc = 100.0 * total_correct / total_samples if total_samples else 0.0
        return epoch_loss, epoch_acc, best_batch_acc

    def _evaluate_fold(self, loader):
        self.model.eval()
        total_correct = 0
        total_samples = 0

        if self.model_variant == "binary":
            confmat_dim = 2
        elif self.model_variant == "multi":
            confmat_dim = self.num_classes
        else:
            confmat_dim = 2

        confmat = torch.zeros(confmat_dim, confmat_dim, dtype=torch.int32, device=self.device)
        all_labels = []
        all_scores = []

        with torch.no_grad():
            for (images, labels) in loader:
                images = images.to(self.device)

                if self.model_variant == "double":
                    bin_labels = torch.where(labels==0, torch.tensor(0, device=self.device), torch.tensor(1, device=self.device))
                    out_bin = self.model(images, mode='binary')
                    probs_bin = nn.functional.softmax(out_bin, dim=1)
                    _, bin_preds = torch.max(out_bin, 1)

                    total_correct += (bin_preds==bin_labels).sum().item()
                    total_samples += labels.size(0)

                    for t, p in zip(bin_labels, bin_preds):
                        confmat[t,p] += 1

                    all_labels.extend(bin_labels.cpu().numpy().tolist())
                    all_scores.extend(probs_bin[:,1].cpu().numpy().tolist())

                else:
                    labels = labels.to(self.device)
                    outputs = self.model(images)
                    probs = nn.functional.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)
                    total_correct += (preds==labels).sum().item()
                    total_samples += labels.size(0)

                    for t, p in zip(labels, preds):
                        confmat[t,p] += 1

                    if self.model_variant == "binary":
                        all_labels.extend(labels.cpu().numpy().tolist())
                        all_scores.extend(probs[:,1].cpu().numpy().tolist())

        accuracy = 100.0 * total_correct / total_samples if total_samples else 0.0
        metrics = {
            "accuracy": accuracy,
            "confusion_matrix": confmat
        }
        if (self.model_variant in ["binary","double"]) and _HAS_SKLEARN:
            from sklearn.metrics import roc_curve, roc_auc_score
            try:
                fpr, tpr, _ = roc_curve(all_labels, all_scores)
                auc_val = roc_auc_score(all_labels, all_scores)
                metrics["roc_fpr"] = fpr
                metrics["roc_tpr"] = tpr
                metrics["roc_auc"] = auc_val
            except ValueError:
                metrics["roc_auc"] = None

        return metrics
