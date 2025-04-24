# File: src/experiment.py

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# For ROC/AUC:
try:
    from sklearn.metrics import roc_curve, roc_auc_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    print("[Warning] sklearn is not installed. ROC/AUC will be unavailable.")

from src.dataset import BrainTumorDataset
from src.transforms_config import create_transform
from src.models import get_model  # <-- (NEW) import from models.py


class Experiment:
    """
    A single experiment pipeline that:
      1) Builds a model (binary / multi / double), or any future variant.
      2) Loads the data (train/validation).
      3) Trains for a set number of epochs (with early stopping).
      4) Evaluates on the validation set, computing accuracy, confusion matrix,
         and optionally ROC/AUC if it's a binary or double model.
      5) Provides a predict_single_image() method to classify external images.
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
        normalization="zscore",   # <--- ADDED THIS PARAM
    ):
        """
        Additional param: normalization => e.g. "zscore", "minmax", or None to skip.
        """
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
        self.device = torch.device(device)
        self.normalization = normalization  # store

        # Build transformations (data augment, normalization, etc.)
        # If user passed normalization=None, we skip or handle "none" logic below:
        norm_str = (None if self.normalization == "none" else self.normalization)
        self.transform = create_transform(
            image_size=(input_size, input_size),
            normalization=norm_str,
            augment=(not disable_augment),
            adv_aug=self.adv_aug,
            strong_aug=self.strong_aug
        )

        # Retrieve the chosen model
        self.model, self.num_classes = get_model(
            model_variant=self.model_variant,
            input_size=self.input_size,
            dropout=self.dropout
        )
        self.model.to(self.device)

        # Construct the optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # We'll create one or more criterion after loading data
        self.criterion = None       # for single-head
        self.criterion_bin = None   # for double-head binary
        self.criterion_multi = None # for double-head multi

        # Early stopping
        self.best_val_acc = 0.0
        self.no_improve_count = 0
        self.earlystop_triggered = False

        # Data loaders (created in load_data())
        self.train_loader = None
        self.val_loader = None

    def load_data(self):
        """
        Load training + validation sets and prepare class-weighting if needed.
        For a double-headed model, the dataset is the same, but we handle 'binary'
        vs. 'multi' logic in train_one_epoch().
        """
        train_set = BrainTumorDataset(
            root_dir=self.data_root,
            subset="Training",
            transform=self.transform
        )
        val_set = BrainTumorDataset(
            root_dir=self.data_root,
            subset="Testing",
            transform=self.transform
        )

        self.train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True
        )

        # If single-head => create self.criterion with optional class weights.
        # If double-head => create separate criterion_bin and criterion_multi.
        if self.model_variant != "double":
            if self.use_class_weight:
                freq = [0] * self.num_classes
                for _, label in train_set.samples:
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
        else:
            # double => separate heads
            self.criterion_bin = nn.CrossEntropyLoss()
            self.criterion_multi = nn.CrossEntropyLoss()

    def reset_earlystop(self):
        self.no_improve_count = 0

    def update_earlystop(self):
        self.no_improve_count += 1
        if self.earlystop_patience > 0 and self.no_improve_count >= self.earlystop_patience:
            self.earlystop_triggered = True

    def train_one_epoch(self, epoch_idx=0):
        """
        Train for one epoch. If 'double', do 2 passes: (binary => no_tumor vs tumor),
        (multi => only for tumor images).
        Otherwise a standard single-head cross-entropy.
        Returns:
          epoch_loss, epoch_acc, best_batch_acc
        """
        self.model.train()
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        best_batch_acc = 0.0

        if self.model_variant == "double":
            for (images, labels) in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Binary pass => label=0 => no tumor, label>0 => tumor=1
                bin_labels = (labels > 0).long()
                out_bin = self.model(images, mode='binary')
                loss_bin = self.criterion_bin(out_bin, bin_labels)

                # Multi pass => only tumor images => label>0 => 1..3 => shift by 1 => 0..2
                tumor_mask = (labels > 0)
                if tumor_mask.any():
                    multi_images = images[tumor_mask]
                    multi_labels = labels[tumor_mask] - 1
                    out_multi = self.model(multi_images, mode='multi')
                    loss_multi = self.criterion_multi(out_multi, multi_labels)
                else:
                    loss_multi = 0.0

                total_loss_batch = loss_bin
                if isinstance(loss_multi, torch.Tensor):
                    total_loss_batch += loss_multi

                self.optimizer.zero_grad()
                total_loss_batch.backward()
                self.optimizer.step()

                # Evaluate correctness in binary classification
                _, bin_preds = torch.max(out_bin, 1)
                batch_correct = (bin_preds == bin_labels).sum().item()
                batch_size = labels.size(0)

                total_correct += batch_correct
                total_samples += batch_size
                total_loss += float(total_loss_batch) * batch_size

                batch_acc = 100.0 * batch_correct / batch_size
                if batch_acc > best_batch_acc:
                    best_batch_acc = batch_acc

            epoch_loss = (total_loss / total_samples) if total_samples else 0.0
            epoch_acc = 100.0 * total_correct / total_samples if total_samples else 0.0

            return epoch_loss, epoch_acc, best_batch_acc

        else:
            # single-head
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

    def evaluate(self, epoch_idx=0):
        """
        Evaluate on the validation set. For double model, we only measure
        the binary classification portion here (toy example).
        Returns a dict with 'accuracy', 'best_batch_acc', 'confusion_matrix',
        'class_accuracy', and optional 'roc_fpr','roc_tpr','roc_auc'.
        """
        self.model.eval()
        total_correct = 0
        total_samples = 0
        best_batch_acc = 0.0

        if self.model_variant == "binary":
            confmat_dim = 2
        elif self.model_variant == "multi":
            confmat_dim = 4
        else:  # 'double' => track 2-class for demonstration
            confmat_dim = 2

        import torch
        confmat = torch.zeros(confmat_dim, confmat_dim, dtype=torch.int32)
        all_labels = []
        all_scores = []

        with torch.no_grad():
            if self.model_variant == "double":
                # Evaluate only binary head
                for (images, labels) in self.val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    bin_labels = (labels > 0).long()
                    out_bin = self.model(images, mode='binary')
                    _, bin_preds = torch.max(out_bin, 1)

                    bin_probs = torch.softmax(out_bin, dim=1)
                    tumor_prob = bin_probs[:, 1].cpu().numpy()

                    batch_correct = (bin_preds == bin_labels).sum().item()
                    bs = labels.size(0)
                    total_correct += batch_correct
                    total_samples += bs

                    batch_acc = 100.0 * batch_correct / bs
                    if batch_acc > best_batch_acc:
                        best_batch_acc = batch_acc

                    for t, p in zip(bin_labels, bin_preds):
                        confmat[t, p] += 1

                    all_labels.extend(bin_labels.cpu().numpy().tolist())
                    all_scores.extend(tumor_prob.tolist())

            else:
                # single-head
                for (images, labels) in self.val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(images)
                    probs = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)

                    batch_correct = (preds == labels).sum().item()
                    bs = labels.size(0)
                    total_correct += batch_correct
                    total_samples += bs

                    batch_acc = 100.0 * batch_correct / bs
                    if batch_acc > best_batch_acc:
                        best_batch_acc = batch_acc

                    for t, p in zip(labels, preds):
                        if t < confmat_dim and p < confmat_dim:
                            confmat[t, p] += 1

                    # If binary => store for ROC
                    if self.model_variant == "binary":
                        all_labels.extend(labels.cpu().numpy().tolist())
                        all_scores.extend(probs[:, 1].cpu().numpy().tolist())

        accuracy = 100.0 * total_correct / total_samples if total_samples else 0.0

        # Class-wise accuracy
        class_acc = {}
        for c in range(confmat_dim):
            row_sum = confmat[c, :].sum().item()
            correct = confmat[c, c].item()
            class_acc[c] = 100.0 * correct / row_sum if row_sum > 0 else 0.0

        metrics = {
            "accuracy": accuracy,
            "best_batch_acc": best_batch_acc,
            "confusion_matrix": confmat,
            "class_accuracy": class_acc
        }

        # If binary/double => attempt ROC
        if (self.model_variant in ["binary", "double"]) and _HAS_SKLEARN:
            import numpy as np
            from sklearn.metrics import roc_curve, roc_auc_score
            try:
                fpr, tpr, _ = roc_curve(all_labels, all_scores)
                roc_auc = roc_auc_score(all_labels, all_scores)
                metrics["roc_fpr"] = fpr
                metrics["roc_tpr"] = tpr
                metrics["roc_auc"] = roc_auc
            except ValueError:
                metrics["roc_auc"] = None

        return metrics

    def predict_single_image(self, pil_image):
        """
        Classify a single external image. For 'double', do 2-step classification:
            1) binary => tumor vs. no tumor
            2) if tumor => multi => which type
        """
        self.model.eval()
        t_img = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.model_variant == "binary":
                outputs = self.model(t_img)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                pred_class = probs.argmax()
                return {
                    "class_index": int(pred_class),
                    "probabilities": probs.tolist()
                }

            elif self.model_variant == "multi":
                outputs = self.model(t_img)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                pred_class = probs.argmax()
                return {
                    "class_index": int(pred_class),
                    "probabilities": probs.tolist()
                }

            elif self.model_variant == "double":
                out_bin = self.model(t_img, mode='binary')
                probs_bin = torch.softmax(out_bin, dim=1).cpu().numpy()[0]
                pred_bin = probs_bin.argmax()

                if pred_bin == 0:
                    return {
                        "prediction": "no_tumor",
                        "binary_probs": probs_bin.tolist()
                    }
                else:
                    out_multi = self.model(t_img, mode='multi')
                    probs_multi = torch.softmax(out_multi, dim=1).cpu().numpy()[0]
                    pred_multi = probs_multi.argmax()
                    return {
                        "prediction": f"tumor_type_{pred_multi}",
                        "binary_probs": probs_bin.tolist(),
                        "multi_probs": probs_multi.tolist()
                    }

            else:
                raise ValueError(f"Unknown model_variant: {self.model_variant}")
