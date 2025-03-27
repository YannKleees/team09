from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QGroupBox, QFormLayout, QLineEdit, QSpinBox
)
from PyQt5.QtCore import Qt

class TrainingTab(QWidget):
    """
    Placeholder for CNN training/metrics. Replace 'print' statements with real training logic.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()

        title_label = QLabel("<h2>Model Training & Evaluation</h2>")
        layout.addWidget(title_label)

        # Group box: Hyperparameters
        hyperparam_box = QGroupBox("Hyperparameters")
        form = QFormLayout()

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 200)
        self.epochs_spin.setValue(5)
        self.epochs_spin.setToolTip("Number of training epochs.")

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(32)
        self.batch_spin.setToolTip("Batch size during training.")

        self.lr_edit = QLineEdit("0.001")
        self.lr_edit.setToolTip("Learning rate for optimizer.")

        form.addRow("Epochs:", self.epochs_spin)
        form.addRow("Batch size:", self.batch_spin)
        form.addRow("Learning rate:", self.lr_edit)
        hyperparam_box.setLayout(form)
        layout.addWidget(hyperparam_box)

        # Buttons
        self.btn_train_custom = QPushButton("Train Custom CNN")
        self.btn_train_custom.setToolTip("Train a custom architecture on the 'cleaned_data' folder.")
        self.btn_train_custom.clicked.connect(self.on_train_custom_cnn)

        self.btn_train_resnet = QPushButton("Train Transfer Learning (ResNet)")
        self.btn_train_resnet.setToolTip("Use a pretrained ResNet model for transfer learning.")
        self.btn_train_resnet.clicked.connect(self.on_train_resnet)

        self.btn_plot_metrics = QPushButton("Plot Metrics / Confusion Matrix")
        self.btn_plot_metrics.setToolTip("Plot training curves or confusion matrix (placeholder).")
        self.btn_plot_metrics.clicked.connect(self.on_plot_metrics)

        layout.addWidget(self.btn_train_custom)
        layout.addWidget(self.btn_train_resnet)
        layout.addWidget(self.btn_plot_metrics)

        self.setLayout(layout)

    def on_train_custom_cnn(self):
        print("Custom CNN training placeholder. (Not implemented)")

    def on_train_resnet(self):
        print("ResNet transfer learning placeholder. (Not implemented)")

    def on_plot_metrics(self):
        print("Plot metrics placeholder. (Not implemented)")
