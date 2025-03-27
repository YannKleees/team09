import os

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QGroupBox, QFormLayout, QLineEdit, 
    QSpinBox, QProgressBar, QTextEdit, QMessageBox
)

# If your code is under "src.experiment", adjust the import accordingly:
from src.experiment import Experiment


class TrainingWorker(QThread):
    """
    Runs the training routine in a background thread to avoid blocking the UI.
    """
    progress = pyqtSignal(int)      # integer progress percentage (0..100)
    log_message = pyqtSignal(str)   # string log messages to display in the UI
    finished = pyqtSignal(float)    # final accuracy or any metric you wish to emit
    
    def __init__(self, epochs, batch_size, lr, parent=None):
        super().__init__(parent)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def run(self):
        try:
            # Initialize your training experiment
            experiment = Experiment(
                data_root="cleaned_data",  # or wherever your preprocessed data is
                batch_size=self.batch_size,
                lr=self.lr,
                epochs=self.epochs
            )

            self.log_message.emit("Loading data...")
            experiment.load_data()
            self.log_message.emit("Data loaded. Beginning training.\n")

            for ep in range(self.epochs):
                # Run one epoch
                experiment.train_one_epoch()
                # Evaluate
                accuracy = experiment.evaluate()
                
                # Notify log area
                msg = f"Epoch {ep+1}/{self.epochs}, Val Accuracy: {accuracy:.2f}%"
                self.log_message.emit(msg)

                # Update progress: simple approach => integer percentage
                prog_percent = int(100 * (ep + 1) / self.epochs)
                self.progress.emit(prog_percent)

            # Training complete
            self.finished.emit(accuracy)
        except Exception as e:
            # If something goes wrong, you might handle it here
            self.log_message.emit(f"ERROR during training: {str(e)}")
            self.progress.emit(0)
            self.finished.emit(0.0)


class TrainingTab(QWidget):
    """
    A tab for model training & evaluation.
    Now includes:
      - QThread-based worker for training
      - A text log and progress bar
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()

        title_label = QLabel("<h2>Model Training & Evaluation</h2>")
        layout.addWidget(title_label)

        # -- Group box: Hyperparameters
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

        # -- Buttons for training
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

        # -- Progress bar to show training progress
        self.train_progress = QProgressBar()
        self.train_progress.setValue(0)
        self.train_progress.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.train_progress)

        # -- Log text area to show training messages
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        layout.addWidget(self.log_edit)

        self.setLayout(layout)

        # We will store our worker here
        self.worker = None

    def on_train_custom_cnn(self):
        """
        Launch the training worker in a separate thread to train a 'custom CNN'.
        """
        # Grab hyperparams from GUI
        epochs = self.epochs_spin.value()
        batch_size = self.batch_spin.value()
        try:
            lr = float(self.lr_edit.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Invalid LR", "Learning rate must be a floating-point number.")
            return

        # Clear any previous logs
        self.log_edit.clear()
        self.train_progress.setValue(0)
        
        # Initialize worker
        self.worker = TrainingWorker(epochs, batch_size, lr)
        # Connect signals
        self.worker.progress.connect(self._on_training_progress)
        self.worker.log_message.connect(self._on_training_log)
        self.worker.finished.connect(self._on_training_finished)
        # Start training
        self.worker.start()

    def on_train_resnet(self):
        """
        Placeholder for transfer learning approach. 
        You could create a second QThread worker specialized in ResNet training,
        or pass a parameter to the same worker to switch the pipeline.
        """
        self.log_edit.append("ResNet transfer learning placeholder. (Not implemented yet)\n")

    def on_plot_metrics(self):
        """
        Placeholder for metrics plotting. 
        E.g., pop up a separate dialog or display in a Matplotlib canvas.
        """
        self.log_edit.append("Plot metrics placeholder. (Not implemented yet)\n")

    # ---------------------
    #  Worker signal slots
    # ---------------------
    def _on_training_progress(self, percent):
        self.train_progress.setValue(percent)

    def _on_training_log(self, msg):
        self.log_edit.append(msg)

    def _on_training_finished(self, final_accuracy):
        self.train_progress.setValue(100)
        if final_accuracy <= 0.0:
            self.log_edit.append("\nTraining finished with errors or zero accuracy.")
        else:
            self.log_edit.append(f"\nTraining completed! Final validation accuracy: {final_accuracy:.2f}%")

        # Clean up worker
        self.worker = None
