# File: gui/training_tab.py

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
    progress = pyqtSignal(int)           # int => progress bar
    log_message = pyqtSignal(str)        # str => log area
    finished = pyqtSignal(float, dict)   # final_acc, final_roc_dict

    def __init__(
        self,
        epochs,
        batch_size,
        lr,
        model_variant="multi",
        strong_aug=False,
        resize_val=128,
        normalization="zscore",
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
        self.device = device

        # We'll store per-epoch data for plotting
        self.train_acc_list = []
        self.val_acc_list = []

        # ROC data (only used if binary/double):
        self.final_roc = {"fpr": None, "tpr": None, "auc": None}

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
                normalization=self.normalization  # Pass the normalization param!
            )
            self.log_message.emit("Loading data...")
            exp.load_data()

            self.log_message.emit(
                f"Variant={self.model_variant}, strong_aug={self.strong_aug}, "
                f"resize={self.resize_val}, norm={self.normalization}\n"
            )

            final_acc = 0.0
            for ep in range(self.epochs):
                # 1) Train
                train_loss, train_acc, _ = exp.train_one_epoch(ep)

                # 2) Validate
                val_metrics = exp.evaluate(ep)
                final_acc = val_metrics["accuracy"]

                # Track for plotting
                self.train_acc_list.append(train_acc)
                self.val_acc_list.append(final_acc)

                # If binary/double, gather final ROC data
                if self.model_variant in ["binary", "double"]:
                    if "roc_fpr" in val_metrics and "roc_tpr" in val_metrics:
                        self.final_roc["fpr"] = val_metrics["roc_fpr"]
                        self.final_roc["tpr"] = val_metrics["roc_tpr"]
                        self.final_roc["auc"] = val_metrics.get("roc_auc", None)

                msg = (
                    f"Epoch {ep+1}/{self.epochs} | "
                    f"TrainAcc={train_acc:.2f}% | ValAcc={final_acc:.2f}%"
                )
                self.log_message.emit(msg)
                self.progress.emit(int(100 * (ep+1) / self.epochs))

            # Return the final accuracy + final ROC data
            self.finished.emit(final_acc, self.final_roc)

        except Exception as e:
            self.log_message.emit(f"ERROR: {str(e)}")
            self.progress.emit(0)
            self.finished.emit(0.0, {"fpr": None, "tpr": None, "auc": None})


class TrainingTab(QWidget):
    """
    Train a model with user-chosen parameters and visualize 
    potential overfitting + final ROC curve.
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
        self.variant_combo.addItems(["binary", "multi", "double"])
        self.strong_aug_chk = QCheckBox("Strong Aug?")

        adv_form.addRow("Resize (px):", self.resize_edit)
        adv_form.addRow("Normalization:", self.norm_edit)
        adv_form.addRow("Model Variant:", self.variant_combo)
        adv_form.addRow(self.strong_aug_chk)

        adv_box.setLayout(adv_form)
        content_layout.addWidget(adv_box)

        # Train Button
        self.btn_train = QPushButton("Train Model")
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

        # Inside the group, we’ll do the same approach
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

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

        # Finally set the content into scroll_area
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)

        self.setLayout(main_layout)

        self.worker = None
        self.train_acc_list = []
        self.val_acc_list = []
        self.roc_data = None

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
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.log_edit.clear()
        self.train_progress.setValue(0)

        self.worker = TrainingWorker(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            model_variant=model_variant,
            strong_aug=strong_aug,
            resize_val=resize_val,
            normalization=norm_str,
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

    def on_finished(self, final_acc, final_roc):
        self.train_progress.setValue(100)
        self.log_edit.append(f"\nDone! Final val acc: {final_acc:.2f}%\n")

        # Retrieve the arrays from the worker
        self.train_acc_list = self.worker.train_acc_list
        self.val_acc_list = self.worker.val_acc_list
        self.roc_data = final_roc  # { "fpr":..., "tpr":..., "auc":... }

        # Plot them
        self.plot_overfitting_check()
        self.plot_roc_curve()

        self.worker = None

    def plot_overfitting_check(self):
        """
        Plots train vs. val accuracy across epochs.
        """
        self.metric_canvas.figure.clear()
        ax = self.metric_canvas.figure.add_subplot(111)

        epochs_range = range(1, len(self.train_acc_list) + 1)
        ax.plot(epochs_range, self.train_acc_list, label="Train Accuracy", marker='o')
        ax.plot(epochs_range, self.val_acc_list, label="Val Accuracy", marker='s')

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Training vs. Validation Accuracy")
        ax.legend()
        ax.grid(True)

        self.metric_canvas.figure.tight_layout()
        self.metric_canvas.draw()

    def plot_roc_curve(self):
        """
        Plots final ROC curve if model is binary/double.
        Otherwise, clears the canvas or shows a note that it's not available.
        """
        self.roc_canvas.figure.clear()
        ax = self.roc_canvas.figure.add_subplot(111)

        fpr = self.roc_data.get("fpr")
        tpr = self.roc_data.get("tpr")
        auc_val = self.roc_data.get("auc")

        if fpr is not None and tpr is not None and auc_val is not None:
            ax.plot(fpr, tpr, color="red", label=f"ROC curve (AUC={auc_val:.2f})")
            ax.plot([0,1], [0,1], linestyle="--", color="gray", label="Random Classifier")
            ax.set_title("Final ROC Curve (Binary/Double)")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "ROC not available (model not binary/double?)",
                    ha='center', va='center', fontsize=12)

        self.roc_canvas.figure.tight_layout()
        self.roc_canvas.draw()
