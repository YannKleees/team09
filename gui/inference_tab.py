"""
inference_tab.py

A PyQt widget (tab) dedicated to model inference on single images,
with a more visually structured layout.
"""

import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QComboBox,
    QLineEdit, QPushButton, QHBoxLayout, QFileDialog,
    QTextEdit, QMessageBox, QGroupBox, QFormLayout, QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import Qt
from PIL import Image

from logic.inference import InferenceModel


class InferenceTab(QWidget):
    """
    A dedicated tab for running single-image inference:
      - "Double (1-file)" or
      - "Two-Step (2-files)" approach.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Optional: set a simple stylesheet for the group boxes / labels
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ccc;
                border-radius: 8px;
                margin-top: 10px;
                padding: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 4px 10px;
            }
            QLabel {
                font-size: 10pt;
            }
            QPushButton {
                background-color: #0078d7;
                color: #ffffff;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #005FA3;
            }
            QLineEdit, QTextEdit {
                font-size: 9pt;
            }
        """)

        # Main vertical layout for the entire tab
        main_layout = QVBoxLayout(self)

        # ------------------------------------------------------
        # GROUP 1: Model Approach & Files
        # ------------------------------------------------------
        model_group = QGroupBox("Model Approach & Files")
        model_group_layout = QVBoxLayout(model_group)

        # (A) Approach selection
        approach_layout = QHBoxLayout()
        approach_label = QLabel("Approach:")
        self.approach_combo = QComboBox()
        self.approach_combo.addItems(["Double (1-file)", "Two-Step (2-files)"])
        self.approach_combo.currentIndexChanged.connect(self.on_approach_changed)
        approach_layout.addWidget(approach_label)
        approach_layout.addWidget(self.approach_combo)
        approach_layout.addSpacerItem(QSpacerItem(20, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        model_group_layout.addLayout(approach_layout)

        # (B) Single-file model row
        single_form = QFormLayout()
        self.single_model_edit = QLineEdit()
        btn_browse_single = QPushButton("Browse Model...")
        btn_browse_single.clicked.connect(self.on_browse_single_model)
        single_hbox = QHBoxLayout()
        single_hbox.addWidget(self.single_model_edit)
        single_hbox.addWidget(btn_browse_single)
        single_form.addRow("Checkpoint:", single_hbox)
        model_group_layout.addLayout(single_form)
        self.single_model_row = single_form  # keep reference to hide/show

        # (C) Two-step model rows (binary + multi)
        two_step_box = QGroupBox("Two-Step Models")
        two_step_layout = QFormLayout(two_step_box)
        # Binary model
        self.binary_model_edit = QLineEdit()
        btn_browse_binary = QPushButton("Browse Binary...")
        btn_browse_binary.clicked.connect(self.on_browse_binary_model)
        h_bin = QHBoxLayout()
        h_bin.addWidget(self.binary_model_edit)
        h_bin.addWidget(btn_browse_binary)
        two_step_layout.addRow("Binary Model:", h_bin)
        # Multi model
        self.multi_model_edit = QLineEdit()
        btn_browse_multi = QPushButton("Browse Multi...")
        btn_browse_multi.clicked.connect(self.on_browse_multi_model)
        h_multi = QHBoxLayout()
        h_multi.addWidget(self.multi_model_edit)
        h_multi.addWidget(btn_browse_multi)
        two_step_layout.addRow("Multi Model:", h_multi)

        model_group_layout.addWidget(two_step_box)
        self.two_step_container = two_step_box  # keep reference to hide/show

        # (D) Load button
        self.load_model_btn = QPushButton("Load Model(s)")
        self.load_model_btn.clicked.connect(self.on_load_models)
        model_group_layout.addWidget(self.load_model_btn)

        main_layout.addWidget(model_group)

        # ------------------------------------------------------
        # GROUP 2: Image Selection & Inference
        # ------------------------------------------------------
        inference_group = QGroupBox("Image & Inference")
        inference_layout = QVBoxLayout(inference_group)

        # (A) Image selection
        img_form = QFormLayout()
        self.img_edit = QLineEdit()
        btn_browse_img = QPushButton("Browse Image...")
        btn_browse_img.clicked.connect(self.on_browse_image)
        img_hbox = QHBoxLayout()
        img_hbox.addWidget(self.img_edit)
        img_hbox.addWidget(btn_browse_img)
        img_form.addRow("Input Image:", img_hbox)
        inference_layout.addLayout(img_form)

        # (B) Predict button
        self.predict_btn = QPushButton("Predict")
        self.predict_btn.clicked.connect(self.on_predict)
        inference_layout.addWidget(self.predict_btn)

        # (C) Output area
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        inference_layout.addWidget(self.output_text)

        main_layout.addWidget(inference_group)

        # Internals (model references)
        self.inference_model_single = None  # for "Double" approach
        self.inference_model_binary = None  # for "Two-step" => binary
        self.inference_model_multi = None   # for "Two-step" => multi

        # Default hyperparams used in InferenceModel
        self.input_size = 128
        self.normalization = "zscore"
        self.device = "cpu"
        self.dropout = 0.0

        self.setLayout(main_layout)
        self.on_approach_changed()  # hide/show relevant fields on startup

    # --------------------------------------------------------------------------
    # Approach UI toggling
    # --------------------------------------------------------------------------
    def on_approach_changed(self):
        approach = self.approach_combo.currentText()
        if "Double" in approach:
            # Show single-file row, hide two-step
            self.show_layout(self.single_model_row, True)
            self.two_step_container.setVisible(False)
        else:
            self.show_layout(self.single_model_row, False)
            self.two_step_container.setVisible(True)
        self.output_text.clear()

    def show_layout(self, layout, do_show=True):
        """Show/hide all widgets in a QFormLayout (or QBoxLayout)."""
        if hasattr(layout, "count"):
            for i in range(layout.count()):
                item = layout.itemAt(i)
                # Each form row has labelWidget & fieldWidget, or a single layout
                if hasattr(item, "widget") and item.widget():
                    item.widget().setVisible(do_show)
                elif hasattr(item, "layout") and item.layout():
                    self.set_layout_visible(item.layout(), do_show)

    def set_layout_visible(self, layout_or_item, do_show=True):
        if hasattr(layout_or_item, 'count'):
            for i in range(layout_or_item.count()):
                w = layout_or_item.itemAt(i).widget()
                if w:
                    w.setVisible(do_show)

    # --------------------------------------------------------------------------
    # File pickers
    # --------------------------------------------------------------------------
    def on_browse_single_model(self):
        dlg = QFileDialog(self)
        dlg.setNameFilter("PyTorch Models (*.pth *.pt);;All Files (*)")
        if dlg.exec_():
            self.single_model_edit.setText(dlg.selectedFiles()[0])

    def on_browse_binary_model(self):
        dlg = QFileDialog(self)
        dlg.setNameFilter("PyTorch Models (*.pth *.pt);;All Files (*)")
        if dlg.exec_():
            self.binary_model_edit.setText(dlg.selectedFiles()[0])

    def on_browse_multi_model(self):
        dlg = QFileDialog(self)
        dlg.setNameFilter("PyTorch Models (*.pth *.pt);;All Files (*)")
        if dlg.exec_():
            self.multi_model_edit.setText(dlg.selectedFiles()[0])

    def on_browse_image(self):
        dlg = QFileDialog(self)
        dlg.setNameFilter("Images (*.png *.jpg *.jpeg);;All Files (*)")
        if dlg.exec_():
            self.img_edit.setText(dlg.selectedFiles()[0])

    # --------------------------------------------------------------------------
    # Load models
    # --------------------------------------------------------------------------
    def on_load_models(self):
        self.output_text.clear()
        self.inference_model_single = None
        self.inference_model_binary = None
        self.inference_model_multi = None

        approach = self.approach_combo.currentText()
        if "Double" in approach:
            # Single-file approach
            ckpt_path = self.single_model_edit.text().strip()
            if not ckpt_path or not os.path.isfile(ckpt_path):
                QMessageBox.warning(self, "Invalid Checkpoint", "Select a valid DoubleHeadCNN checkpoint.")
                return
            try:
                self.inference_model_single = InferenceModel(
                    checkpoint_path=ckpt_path,
                    model_variant="double",
                    input_size=self.input_size,
                    normalization=self.normalization,
                    device=self.device,
                    dropout=self.dropout
                )
                self.output_text.append(f"Double model loaded:\n{ckpt_path}\n")
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Error loading double model:\n{e}")
                return
        else:
            # Two-step approach => separate binary + multi
            bin_ckpt = self.binary_model_edit.text().strip()
            mul_ckpt = self.multi_model_edit.text().strip()
            if not bin_ckpt or not os.path.isfile(bin_ckpt):
                QMessageBox.warning(self, "Invalid Checkpoint", "Select a valid binary checkpoint.")
                return
            if not mul_ckpt or not os.path.isfile(mul_ckpt):
                QMessageBox.warning(self, "Invalid Checkpoint", "Select a valid multi checkpoint.")
                return

            # Load binary
            try:
                self.inference_model_binary = InferenceModel(
                    checkpoint_path=bin_ckpt,
                    model_variant="binary",
                    input_size=self.input_size,
                    normalization=self.normalization,
                    device=self.device,
                    dropout=self.dropout
                )
                self.output_text.append(f"Binary model loaded:\n{bin_ckpt}\n")
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Error loading binary model:\n{e}")
                return

            # Load multi
            try:
                self.inference_model_multi = InferenceModel(
                    checkpoint_path=mul_ckpt,
                    model_variant="multi",
                    input_size=self.input_size,
                    normalization=self.normalization,
                    device=self.device,
                    dropout=self.dropout
                )
                self.output_text.append(f"Multi model loaded:\n{mul_ckpt}\n")
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Error loading multi model:\n{e}")
                return

        self.output_text.append("Model(s) loaded successfully.\n")

    # --------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------
    def on_predict(self):
        approach = self.approach_combo.currentText()
        img_path = self.img_edit.text().strip()
        if not img_path or not os.path.isfile(img_path):
            QMessageBox.warning(self, "Invalid Image", "Select a valid image file.")
            return

        try:
            pil_img = Image.open(img_path).convert('L')
        except Exception as e:
            QMessageBox.critical(self, "Image Error", f"Failed to open image:\n{e}")
            return

        self.output_text.clear()
        self.output_text.append("<b>Inference Output</b>\n")

        if "Double" in approach:
            # Single double-headed model
            if not self.inference_model_single:
                QMessageBox.warning(self, "Not Loaded", "Load the DoubleHeadCNN model first.")
                return
            try:
                result = self.inference_model_single.predict(pil_img)
                self.display_double_result(result)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Double inference error:\n{e}")
                return
        else:
            # Two-step approach
            if not (self.inference_model_binary and self.inference_model_multi):
                QMessageBox.warning(self, "Not Loaded", "Load both binary and multi models first.")
                return
            try:
                # Step 1: binary
                bin_result = self.inference_model_binary.predict(pil_img)
                idx = bin_result["class_index"]
                probs = bin_result["probabilities"]
                tumor_conf = probs[1] * 100.0

                if idx == 0:
                    self.output_text.append(
                        f"Binary => NO TUMOR (confidence ~{100.0 - tumor_conf:.1f}%)"
                    )
                else:
                    self.output_text.append(
                        f"Binary => TUMOR (confidence ~{tumor_conf:.1f}%)\n"
                    )
                    # Step 2: multi
                    mul_result = self.inference_model_multi.predict(pil_img)
                    self.display_multi_result(mul_result)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Two-step inference error:\n{e}")
                return

        self.output_text.append("\n--- Done ---\n")

    # --------------------------------------------------------------------------
    # Display results
    # --------------------------------------------------------------------------
    def display_double_result(self, result: dict):
        """
        If result => {
          'prediction': 'no_tumor' OR 'tumor_subtype_X',
          'binary_probs': [...],
          'multi_probs': [...] optional
        }
        """
        if "binary_probs" not in result:
            self.output_text.append("Unexpected result structure from 'double' model.\n")
            return

        bin_probs = result["binary_probs"]
        tumor_conf = bin_probs[1] * 100.0
        self.output_text.append(
            f"Binary head => p(no_tumor)={bin_probs[0]:.3f}, p(tumor)={bin_probs[1]:.3f}"
        )
        self.output_text.append(f"\nTumor confidence => {tumor_conf:.1f}%\n")

        if result["prediction"] == "no_tumor":
            self.output_text.append("==> Prediction: NO TUMOR\n")
        else:
            self.output_text.append(f"==> Prediction: {result['prediction']}\n")
            if "multi_probs" in result:
                multi_probs = result["multi_probs"]
                self.output_text.append(f"Subtype probabilities => {multi_probs}\n")

    def display_multi_result(self, mul_result: dict):
        """
        If mul_result => {
          'class_index': int,
          'probabilities': [p0, p1, p2, ...]
        }
        e.g. 3 or 4 classes
        """
        idx = mul_result["class_index"]
        probs = mul_result["probabilities"]
        self.output_text.append("Multi => Subtype classification:\n")
        self.output_text.append(f"Class Index => {idx}\n")
        self.output_text.append(f"Probabilities => {probs}\n")
