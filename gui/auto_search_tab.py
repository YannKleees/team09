# File: gui/auto_search_tab.py

import os
import ast

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QGroupBox,
    QSpinBox, QTextEdit, QMessageBox, QComboBox, QProgressBar,
    QLineEdit, QHBoxLayout, QCheckBox, QGridLayout, QScrollArea
)
from PyQt5.QtCore import Qt

from logic.auto_experiment import AutoSearchWorker


class AutoSearchTab(QWidget):
    """
    A single-tab approach for a full hyperparameter search using AutoSearchWorker.
    All step-by-step logic is removed: just one button to run everything at once.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None

        # ---------------------
        # Main layout (scrollable)
        # ---------------------
        main_layout = QVBoxLayout(self)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(10, 10, 10, 10)

        # Title and short description
        title_label = QLabel("<h2>Auto Hyperparameter Search</h2>")
        title_label.setWordWrap(True)
        content_layout.addWidget(title_label)

        desc_label = QLabel(
            "Fill out the hyperparameter ranges below, then click "
            "'Run Auto Search' to perform a combined search. "
            "The log at the bottom will show progress and final results."
        )
        desc_label.setWordWrap(True)
        content_layout.addWidget(desc_label)

        # ---------------------
        # Hyperparameter grid
        # ---------------------
        cfg_box = QGroupBox("Hyperparameter Settings")
        grid = QGridLayout()

        # Row 0
        lbl_data = QLabel("Data Root:")
        self.data_root = QComboBox()
        self.data_root.addItem("data/archive")
        self.data_root.setEditable(True)
        self.data_root.setToolTip("Folder containing 'Training' and 'Testing' subfolders.")

        lbl_resize = QLabel("Resize List:")
        self.resize_edit = QLineEdit("128,160")
        self.resize_edit.setToolTip("Comma-separated image sizes, e.g. 128,160,224.")

        grid.addWidget(lbl_data, 0, 0)
        grid.addWidget(self.data_root, 0, 1)
        grid.addWidget(lbl_resize, 0, 2)
        grid.addWidget(self.resize_edit, 0, 3)

        # Row 1
        lbl_norm = QLabel("Norm List:")
        self.norm_edit = QLineEdit("minmax,zscore,none")
        self.norm_edit.setToolTip("Comma-separated normalization methods.")

        lbl_disable_aug = QLabel("Disable Aug:")
        self.disable_aug_edit = QLineEdit("False,True")
        self.disable_aug_edit.setToolTip("Whether to skip basic augmentation (True/False).")

        grid.addWidget(lbl_norm, 1, 0)
        grid.addWidget(self.norm_edit, 1, 1)
        grid.addWidget(lbl_disable_aug, 1, 2)
        grid.addWidget(self.disable_aug_edit, 1, 3)

        # Row 2
        lbl_adv_aug = QLabel("Adv Aug:")
        self.adv_aug_edit = QLineEdit("False,True")
        self.adv_aug_edit.setToolTip("Whether to apply advanced aug (random crops, etc.).")

        lbl_drop = QLabel("Dropout:")
        self.dropout_edit = QLineEdit("0.0,0.3")
        self.dropout_edit.setToolTip("Comma-separated dropout rates, e.g. 0.0,0.3,0.5.")

        grid.addWidget(lbl_adv_aug, 2, 0)
        grid.addWidget(self.adv_aug_edit, 2, 1)
        grid.addWidget(lbl_drop, 2, 2)
        grid.addWidget(self.dropout_edit, 2, 3)

        # Row 3
        lbl_wd = QLabel("Weight Decay:")
        self.wd_edit = QLineEdit("0.0,0.0001")
        self.wd_edit.setToolTip("Comma-separated weight decay values, e.g. 0.0,0.0001,0.001.")

        lbl_early = QLabel("Early Stop:")
        self.earlystop_edit = QLineEdit("0,3")
        self.earlystop_edit.setToolTip("Comma-separated early-stop patience values (0=off).")

        grid.addWidget(lbl_wd, 3, 0)
        grid.addWidget(self.wd_edit, 3, 1)
        grid.addWidget(lbl_early, 3, 2)
        grid.addWidget(self.earlystop_edit, 3, 3)

        # Row 4
        lbl_class_weight = QLabel("Class Weights:")
        self.class_weight_edit = QLineEdit("False,True")
        self.class_weight_edit.setToolTip("Whether to apply class weighting (True/False).")

        lbl_epoch = QLabel("Epochs:")
        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(1, 50)
        self.epoch_spin.setValue(5)
        self.epoch_spin.setToolTip("Number of epochs for each combo.")

        grid.addWidget(lbl_class_weight, 4, 0)
        grid.addWidget(self.class_weight_edit, 4, 1)
        grid.addWidget(lbl_epoch, 4, 2)
        grid.addWidget(self.epoch_spin, 4, 3)

        # Row 5
        lbl_batch = QLabel("Batch Size:")
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 256)
        self.batch_spin.setValue(16)
        self.batch_spin.setToolTip("Batch size used during training.")

        lbl_lr = QLabel("Learning Rate:")
        self.lr_edit = QLineEdit("0.001")
        self.lr_edit.setToolTip("Initial learning rate for optimizer.")

        grid.addWidget(lbl_batch, 5, 0)
        grid.addWidget(self.batch_spin, 5, 1)
        grid.addWidget(lbl_lr, 5, 2)
        grid.addWidget(self.lr_edit, 5, 3)

        # Row 6
        lbl_model_variant = QLabel("Model Variant:")
        self.model_variant_combo = QComboBox()
        self.model_variant_combo.addItems(["binary", "multi", "double"])
        self.model_variant_combo.setToolTip("binary=2-class, multi=4-class, double=2-headed network.")

        self.strong_aug_chk = QCheckBox("Strong Aug?")
        self.strong_aug_chk.setToolTip("Apply stronger perspective/warp transformations?")

        grid.addWidget(lbl_model_variant, 6, 0)
        grid.addWidget(self.model_variant_combo, 6, 1)
        grid.addWidget(self.strong_aug_chk, 6, 2, 1, 2)

        cfg_box.setLayout(grid)
        content_layout.addWidget(cfg_box)

        # ---------------------
        # Run Button
        # ---------------------
        self.run_btn = QPushButton("Run Auto Search")
        self.run_btn.setToolTip("Start a full hyperparameter search with the given settings.")
        self.run_btn.clicked.connect(self.on_run_search)
        content_layout.addWidget(self.run_btn)

        # ---------------------
        # Progress & Log
        # ---------------------
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(self.progress_bar)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        content_layout.addWidget(self.log_area)

        # Wrap in scroll area
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)

    def on_run_search(self):
        """
        Parse user inputs, create an AutoSearchWorker, run the search, and 
        display logs. No placeholders: it does the entire search.
        """
        # Clear log & reset progress
        self.log_area.clear()
        self.progress_bar.setValue(0)

        # Validate data root
        root_dir = self.data_root.currentText().strip()
        if not os.path.isdir(root_dir):
            QMessageBox.warning(self, "Invalid Folder", f"'{root_dir}' not a valid directory.")
            return

        # Try to parse user inputs
        try:
            resize_list = [int(x.strip()) for x in self.resize_edit.text().split(",")]
            disable_aug_list = [ast.literal_eval(x.strip()) for x in self.disable_aug_edit.text().split(",")]
            adv_aug_list = [ast.literal_eval(x.strip()) for x in self.adv_aug_edit.text().split(",")]
            dropout_list = [float(x.strip()) for x in self.dropout_edit.text().split(",")]
            wd_list = [float(x.strip()) for x in self.wd_edit.text().split(",")]
            es_list = [int(x.strip()) for x in self.earlystop_edit.text().split(",")]
            cw_list = [ast.literal_eval(x.strip()) for x in self.class_weight_edit.text().split(",")]
            epochs = self.epoch_spin.value()
            batch_size = self.batch_spin.value()
            lr = float(self.lr_edit.text())
            model_variant = self.model_variant_combo.currentText()
            strong_aug = self.strong_aug_chk.isChecked()
        except Exception as e:
            QMessageBox.critical(self, "Parse Error", f"Failed to parse inputs:\n{e}")
            return

        # Decide how many classes for the search
        # (If model_variant == "binary" => 2 classes, else => 4)
        if model_variant == "binary":
            num_classes = 2
        else:
            # "multi" or "double" => 4
            num_classes = 4

        # Log the input config
        self.log_area.append("<b>Running Auto Search with:</b>\n")
        self.log_area.append(f"Data Root: {root_dir}")
        self.log_area.append(f"resize_list = {resize_list}")
        self.log_area.append(f"disable_aug_list = {disable_aug_list}")
        self.log_area.append(f"adv_aug_list = {adv_aug_list}")
        self.log_area.append(f"dropout_list = {dropout_list}")
        self.log_area.append(f"weight_decay_list = {wd_list}")
        self.log_area.append(f"earlystop_list = {es_list}")
        self.log_area.append(f"class_weight_list = {cw_list}")
        self.log_area.append(f"epochs = {epochs}, batch_size = {batch_size}, lr = {lr}")
        self.log_area.append(f"model_variant = {model_variant}, strong_aug = {strong_aug}")
        self.log_area.append(f"num_classes = {num_classes}\n")

        # Create & setup the worker
        self.worker = AutoSearchWorker()
        self.worker.setup_search(
            data_root=root_dir,
            resize_list=resize_list,
            disable_augment_list=disable_aug_list,
            adv_aug_list=adv_aug_list,
            dropout_list=dropout_list,
            weight_decay_list=wd_list,
            earlystop_list=es_list,
            class_weight_list=cw_list,
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            num_classes=num_classes,
            device="cpu"
        )

        # Connect signals
        self.worker.log_msg.connect(self._on_log)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_search_finished)

        # Disable run button during search
        self.run_btn.setEnabled(False)

        # Start the search thread
        self.log_area.append("<b>Starting hyperparameter search...</b>\n")
        self.worker.start()

    def _on_log(self, msg):
        """
        Called whenever the search worker logs a message.
        """
        self.log_area.append(msg)

    def _on_progress(self, val):
        """
        Update progress bar
        """
        self.progress_bar.setValue(val)

    def _on_search_finished(self):
        """
        Called when the worker finishes completely.
        """
        self.progress_bar.setValue(100)
        self.log_area.append("\n<b>Auto Search Completed!</b>")
        # Re-enable the run button
        self.run_btn.setEnabled(True)
        self.worker = None
