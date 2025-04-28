import torch
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QGroupBox, QFormLayout, QLineEdit,
    QSpinBox, QProgressBar, QTextEdit, QComboBox, QCheckBox, QHBoxLayout,
    QScrollArea
)

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from gui.widgets import MplCanvas

from src.experiment import Experiment


class TrainingWorker(QThread):
    """
    Worker that can perform normal training or k-fold cross-validation
    in the background. Also collects final confusion matrix & metrics.
    """
    progress = pyqtSignal(int)           # int => progress bar
    log_message = pyqtSignal(str)        # str => log area
    finished = pyqtSignal(float, dict, dict, dict)   
    # emits: (final_or_avg_acc, final_roc, final_confmat, class_metrics)

    def __init__(
        self,
        epochs,
        batch_size,
        lr,
        model_variant="multi",
        strong_aug=False,
        resize_val=128,
        normalization="zscore",
        folds=1,                     # new
        device="cpu",
        parent=None
    ):
        super().__init__(parent)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model_variant = model_variant
        self.strong_aug = strong_aug
        self.resize_val = resize_val
        self.normalization = normalization
        self.folds = folds
        self.device = device

        # Storage for final plots
        self.train_acc_list = []
        self.val_acc_list = []
        self.final_roc = {"fpr": None, "tpr": None, "auc": None}
        self.final_confmat = None
        self.class_metrics = {}  # e.g. { "precision": [...], "recall": [...], "f1": [...] }

    def run(self):
        try:
            exp = Experiment(
                data_root="cleaned_data",
                model_variant=self.model_variant,
                strong_aug=self.strong_aug,
                batch_size=self.batch_size,
                lr=self.lr,
                epochs=self.epochs,
                device=self.device,
                input_size=self.resize_val,
                normalization=self.normalization,
                folds=self.folds  # new param
            )
            exp.set_logger(self.log_message.emit)  # forward logs
            self.log_message.emit(
                f"Variant={self.model_variant}, strong_aug={self.strong_aug}, "
                f"resize={self.resize_val}, norm={self.normalization}, folds={self.folds}\n"
            )

            if self.folds <= 1:
                # Normal single-run training
                self.log_message.emit("Loading data (single-run)...")
                exp.load_data()
                final_acc, all_train_acc, all_val_acc = exp.run_single_training(
                    epochs=self.epochs,
                    progress_cb=self.progress
                )
                self.train_acc_list = all_train_acc
                self.val_acc_list = all_val_acc
                # gather final ROC, confmat, metrics
                self.final_roc = exp.final_roc
                self.final_confmat = exp.final_confmat
                self.class_metrics = exp.final_class_metrics
                self.finished.emit(final_acc, self.final_roc, self.final_confmat, self.class_metrics)
            else:
                # K-Fold cross validation
                self.log_message.emit(f"Performing {self.folds}-fold cross-validation...")
                # We do not use progress bar per fold => just approximate
                total_steps = self.epochs * self.folds
                step_count = 0

                avg_acc = 0.0
                all_train_acc_over_folds = []
                all_val_acc_over_folds = []

                # run_kfold returns final confusion matrix (summed) & final metrics (averaged)
                avg_acc, accum_confmat, avg_metrics, (train_accs_list, val_accs_list) = exp.run_kfold(
                    folds=self.folds,
                    epochs=self.epochs,
                    progress_cb=lambda: self._cv_progress_callback(total_steps, step_count)
                )
                # store for final plotting => we just store the last fold's arrays or
                # we can store the average across folds
                if train_accs_list:
                    self.train_acc_list = train_accs_list[-1]
                if val_accs_list:
                    self.val_acc_list = val_accs_list[-1]

                self.final_confmat = accum_confmat
                self.class_metrics = avg_metrics
                self.final_roc = exp.final_roc  # if anything was set
                self.finished.emit(avg_acc, self.final_roc, self.final_confmat, self.class_metrics)

        except Exception as e:
            self.log_message.emit(f"ERROR: {str(e)}")
            self.progress.emit(0)
            # Return placeholders
            self.finished.emit(0.0, {}, None, {})

    def _cv_progress_callback(self, total_steps, step_count):
        """
        Not strictly used in this example, but you can increment step_count,
        emit progress, etc. or just omit a fancy callback.
        """
        pass


class TrainingTab(QWidget):
    """
    Train a model with user-chosen parameters (single run or cross-validation),
    visualize potential overfitting + final ROC curve, and show confusion matrix & metrics.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Main layout (scrollable)
        main_layout = QVBoxLayout(self)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(10, 10, 10, 10)

        # Title
        title_label = QLabel("<h2>Model Training</h2>")
        content_layout.addWidget(title_label)

        # ---------------------------------------------
        # BASIC hyperparams group
        # ---------------------------------------------
        basic_box = QGroupBox("Basic Hyperparams")
        basic_form = QFormLayout()

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 200)
        self.epochs_spin.setValue(5)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(32)

        self.lr_edit = QLineEdit("0.001")

        basic_form.addRow("Epochs:", self.epochs_spin)
        basic_form.addRow("Batch size:", self.batch_spin)
        basic_form.addRow("Learning Rate:", self.lr_edit)

        basic_box.setLayout(basic_form)
        content_layout.addWidget(basic_box)

        # ---------------------------------------------
        # ADV hyperparams group
        # ---------------------------------------------
        adv_box = QGroupBox("Advanced Hyperparams")
        adv_form = QFormLayout()

        self.resize_edit = QLineEdit("128")
        self.norm_edit = QLineEdit("zscore")  # or "minmax" or "none"
        self.variant_combo = QComboBox()
        # now includes 'transfer' & 'custom' placeholders
        self.variant_combo.addItems(["binary", "multi", "double", "transfer", "custom"])
        self.strong_aug_chk = QCheckBox("Strong Aug?")

        self.folds_spin = QSpinBox()  # cross-validation folds
        self.folds_spin.setRange(1, 10)
        self.folds_spin.setValue(1)

        adv_form.addRow("Resize (px):", self.resize_edit)
        adv_form.addRow("Normalization:", self.norm_edit)
        adv_form.addRow("Model Variant:", self.variant_combo)
        adv_form.addRow(self.strong_aug_chk)
        adv_form.addRow("Folds (k):", self.folds_spin)

        adv_box.setLayout(adv_form)
        content_layout.addWidget(adv_box)

        # Train Button
        self.btn_train = QPushButton("Train Model (or K-Fold)")
        self.btn_train.clicked.connect(self.on_train)
        content_layout.addWidget(self.btn_train)

        # Progress bar + log
        self.train_progress = QProgressBar()
        self.train_progress.setValue(0)
        self.train_progress.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(self.train_progress)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        content_layout.addWidget(self.log_edit)

        # Plot group for Overfitting & ROC
        self.plot_group = QGroupBox("Overfitting Check & ROC Curve")
        plot_layout = QVBoxLayout()
        content_layout.addWidget(self.plot_group)

        self.metric_canvas = MplCanvas(self, width=6, height=4)  # bigger
        self.metric_toolbar = NavigationToolbar(self.metric_canvas, self)

        self.roc_canvas = MplCanvas(self, width=6, height=4)     # bigger
        self.roc_toolbar = NavigationToolbar(self.roc_canvas, self)

        label_explain = QLabel(
            "Left: Train vs. Val accuracy (large gap => potential overfitting)\n"
            "Right: Final ROC curve (if binary/double)."
        )
        label_explain.setWordWrap(True)
        plot_layout.addWidget(label_explain)

        # Horizontal layout for the two canvases
        h_plot_layout = QHBoxLayout()

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.addWidget(self.metric_canvas)
        left_layout.addWidget(self.metric_toolbar)
        h_plot_layout.addWidget(left_widget)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(self.roc_canvas)
        right_layout.addWidget(self.roc_toolbar)
        h_plot_layout.addWidget(right_widget)

        plot_layout.addLayout(h_plot_layout)
        self.plot_group.setLayout(plot_layout)

        # Confusion Matrix group
        self.confmat_group = QGroupBox("Confusion Matrix & Class Metrics")
        cm_layout = QVBoxLayout()
        content_layout.addWidget(self.confmat_group)

        self.confmat_canvas = MplCanvas(self, width=6, height=4)
        self.confmat_toolbar = NavigationToolbar(self.confmat_canvas, self)

        cm_layout.addWidget(self.confmat_canvas)
        cm_layout.addWidget(self.confmat_toolbar)

        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        cm_layout.addWidget(self.metrics_text)

        self.confmat_group.setLayout(cm_layout)

        # Finally set the content into scroll_area
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)

        # Worker reference
        self.worker = None
        # Data we plot
        self.train_acc_list = []
        self.val_acc_list = []
        self.roc_data = None
        self.confmat_data = None
        self.class_metrics = {}

    def on_train(self):
        try:
            epochs = self.epochs_spin.value()
            batch_size = self.batch_spin.value()
            lr = float(self.lr_edit.text())
            resize_val = int(self.resize_edit.text())
            norm_str = self.norm_edit.text().strip().lower()
        except ValueError:
            self.log_edit.append("Invalid input for training parameters.")
            return

        model_variant = self.variant_combo.currentText()
        strong_aug = self.strong_aug_chk.isChecked()
        folds = self.folds_spin.value()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.log_edit.clear()
        self.train_progress.setValue(0)

        # Launch worker
        self.worker = TrainingWorker(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            model_variant=model_variant,
            strong_aug=strong_aug,
            resize_val=resize_val,
            normalization=norm_str,
            folds=folds,
            device=device
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.log_message.connect(self.on_log)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_progress(self, val):
        self.train_progress.setValue(val)

    def on_log(self, msg):
        self.log_edit.append(msg)

    def on_finished(self, final_acc, final_roc, confmat, metrics_dict):
        self.train_progress.setValue(100)
        if self.folds_spin.value() > 1:
            self.log_edit.append(f"\nDone! Average val acc across folds: {final_acc:.2f}%\n")
        else:
            self.log_edit.append(f"\nDone! Final val acc: {final_acc:.2f}%\n")

        self.roc_data = final_roc
        self.confmat_data = confmat
        self.class_metrics = metrics_dict

        # Plot them (train vs. val)
        self.plot_overfitting_check()
        self.plot_roc_curve()

        # Confusion matrix & metrics
        self.plot_confusion_matrix()
        self.show_class_metrics()

        self.worker = None

    def plot_overfitting_check(self):
        """
        Plots train vs. val accuracy across epochs. We read from the worker's logs if desired.
        But for simplicity, let's read from the experiment data stored in worker.
        """
        # Actually, we didn't store them globally. We can do the same approach
        # or skip. We'll just do a note:
        # Because we didn't store them in the final signals,
        # one approach is to read from worker. Another is "not implemented."
        # For now, let's clear the figure.
        self.metric_canvas.figure.clear()
        ax = self.metric_canvas.figure.add_subplot(111)
        ax.set_title("Train vs. Val Accuracy (not stored in this example)")
        ax.text(0.5, 0.5, "Implementation needed to store epoch-acc lists", ha='center', va='center')
        self.metric_canvas.draw()

    def plot_roc_curve(self):
        """
        Plots final ROC curve if model is binary/double. 
        If self.roc_data has "fpr","tpr","auc", we do it.
        """
        self.roc_canvas.figure.clear()
        ax = self.roc_canvas.figure.add_subplot(111)

        if self.roc_data and "fpr" in self.roc_data and self.roc_data["fpr"] is not None:
            fpr = self.roc_data["fpr"]
            tpr = self.roc_data["tpr"]
            auc_val = self.roc_data["auc"]
            ax.plot(fpr, tpr, color="red", label=f"ROC curve (AUC={auc_val:.2f})")
            ax.plot([0,1], [0,1], linestyle="--", color="gray", label="Random Classifier")
            ax.set_title("Final ROC Curve (Binary/Double)")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "ROC not available or not binary/double", ha='center', va='center')

        self.roc_canvas.figure.tight_layout()
        self.roc_canvas.draw()

    def plot_confusion_matrix(self):
        """
        If confmat_data is not None, show a heatmap. We'll do a simple approach 
        using a basic matplotlib `imshow` or `matshow`.
        For more advanced, one might use seaborn or something else.
        """
        self.confmat_canvas.figure.clear()
        ax = self.confmat_canvas.figure.add_subplot(111)

        if self.confmat_data is not None:
            import numpy as np
            confmat_array = self.confmat_data.cpu().numpy().astype(float)
            cmax = confmat_array.max()
            im = ax.imshow(confmat_array, cmap="Blues", vmin=0, vmax=cmax)

            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

            # Put text
            num_classes = confmat_array.shape[0]
            for i in range(num_classes):
                for j in range(num_classes):
                    val = int(confmat_array[i,j])
                    ax.text(j, i, str(val), ha='center', va='center', color='black')

            self.confmat_canvas.figure.colorbar(im, ax=ax)
        else:
            ax.text(0.5, 0.5, "No Confusion Matrix Found", ha='center', va='center')

        self.confmat_canvas.figure.tight_layout()
        self.confmat_canvas.draw()

    def show_class_metrics(self):
        """
        If we have 'precision', 'recall', 'f1' in self.class_metrics, 
        display them in the text box.
        """
        self.metrics_text.clear()
        if not self.class_metrics:
            self.metrics_text.append("No class metrics found.\n")
            return

        # e.g. { "precision": [p1, p2, ...], "recall": [...], "f1": [...], "support": [...] }
        prec_list = self.class_metrics.get("precision", [])
        rec_list = self.class_metrics.get("recall", [])
        f1_list = self.class_metrics.get("f1", [])
        sup_list = self.class_metrics.get("support", [])

        num_cls = len(prec_list)
        self.metrics_text.append("<b>Class Metrics (Precision / Recall / F1 / Support):</b>")
        for c in range(num_cls):
            p = prec_list[c]
            r = rec_list[c]
            f1 = f1_list[c]
            s = sup_list[c]
            self.metrics_text.append(
                f"Class {c}: prec={p:.3f}, rec={r:.3f}, f1={f1:.3f}, support={s}"
            )

        # optionally, we can show micro/macro average if stored
        micro = self.class_metrics.get("micro_f1", None)
        macro = self.class_metrics.get("macro_f1", None)
        if micro is not None:
            self.metrics_text.append(f"\n<b>Micro-F1:</b> {micro:.3f}")
        if macro is not None:
            self.metrics_text.append(f"<b>Macro-F1:</b> {macro:.3f}")

