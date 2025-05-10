"""
inference_tab.py

A PyQt widget (tab) dedicated to model inference on single images,
with a more visually structured layout and Grad-CAM analysis.
"""

import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QComboBox,
    QLineEdit, QPushButton, QHBoxLayout, QFileDialog,
    QTextEdit, QMessageBox, QGroupBox, QFormLayout, QSpacerItem, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage # <-- Added QImage, QPixmap
from PyQt5.QtCore import Qt
from PIL import Image

# --- Grad-CAM and ML Imports ---
import torch
from torchvision import transforms # Keep for type hinting or if needed directly
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget # Optional: for specific targets
import matplotlib.pyplot as plt # Though show_cam_on_image uses it, we don't call plt.show()
import numpy as np

from logic.inference import InferenceModel


class InferenceTab(QWidget):
    """
    A dedicated tab for running single-image inference:
      - "Double (1-file)" or
      - "Two-Step (2-files)" approach.
      - Includes Grad-CAM visualization.
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
            QLabel#GradCamDisplayLabel { /* Style for the Grad-CAM image label */
                border: 1px solid #ccc;
                background-color: #f0f0f0;
                min-height: 256px; /* Adjust as needed */
                min-width: 256px;  /* Adjust as needed */
                alignment: center;
            }
        """)

        # Main vertical layout for the entire tab
        main_layout = QVBoxLayout(self)

        # ... (GROUP 1: Model Approach & Files)
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
        self.single_model_edit.setPlaceholderText("Path to .pth or .pt model file")
        btn_browse_single = QPushButton("Browse Model...")
        btn_browse_single.clicked.connect(self.on_browse_single_model)
        single_hbox = QHBoxLayout()
        single_hbox.addWidget(self.single_model_edit)
        single_hbox.addWidget(btn_browse_single)
        single_form.addRow("Checkpoint:", single_hbox)
        model_group_layout.addLayout(single_form)
        # Store the container widget for the form layout to hide/show all elements
        self.single_model_container = QWidget() 
        self.single_model_container.setLayout(single_form)
        model_group_layout.addWidget(self.single_model_container)


        # (C) Two-step model rows (binary + multi)
        two_step_box = QGroupBox("Two-Step Models") # This is already a QGroupBox
        two_step_layout = QFormLayout(two_step_box)
        # Binary model
        self.binary_model_edit = QLineEdit()
        self.binary_model_edit.setPlaceholderText("Path to binary classification model")
        btn_browse_binary = QPushButton("Browse Binary...")
        btn_browse_binary.clicked.connect(self.on_browse_binary_model)
        h_bin = QHBoxLayout()
        h_bin.addWidget(self.binary_model_edit)
        h_bin.addWidget(btn_browse_binary)
        two_step_layout.addRow("Binary Model:", h_bin)
        # Multi model
        self.multi_model_edit = QLineEdit()
        self.multi_model_edit.setPlaceholderText("Path to multi-class classification model")
        btn_browse_multi = QPushButton("Browse Multi...")
        btn_browse_multi.clicked.connect(self.on_browse_multi_model)
        h_multi = QHBoxLayout()
        h_multi.addWidget(self.multi_model_edit)
        h_multi.addWidget(btn_browse_multi)
        two_step_layout.addRow("Multi Model:", h_multi)

        model_group_layout.addWidget(two_step_box)
        self.two_step_container = two_step_box


        # (D) Load button
        self.load_model_btn = QPushButton("Load Model(s)")
        self.load_model_btn.clicked.connect(self.on_load_models)
        model_group_layout.addWidget(self.load_model_btn)
        main_layout.addWidget(model_group)


        # ------------------------------------------------------
        # GROUP 2: Image Selection, Inference & Grad-CAM
        # ------------------------------------------------------
        inference_group = QGroupBox("Image, Inference & Grad-CAM") # Renamed Group
        inference_group_layout = QVBoxLayout(inference_group) # Changed to QVBoxLayout for easier stacking

        # (A) Image selection
        img_form_layout = QFormLayout() # Use a QFormLayout for this part
        self.img_edit = QLineEdit()
        self.img_edit.setPlaceholderText("Path to input image file")
        btn_browse_img = QPushButton("Browse Image...")
        btn_browse_img.clicked.connect(self.on_browse_image)
        img_hbox = QHBoxLayout()
        img_hbox.addWidget(self.img_edit)
        img_hbox.addWidget(btn_browse_img)
        img_form_layout.addRow("Input Image:", img_hbox)
        inference_group_layout.addLayout(img_form_layout)

        # (B) Action Buttons (Predict & Grad-CAM)
        action_buttons_layout = QHBoxLayout()
        self.predict_btn = QPushButton("Predict")
        self.predict_btn.clicked.connect(self.on_predict)
        action_buttons_layout.addWidget(self.predict_btn)

        self.grad_cam_btn = QPushButton("GRAD-CAM Analysis") # <-- New Button
        self.grad_cam_btn.clicked.connect(self.on_grad_cam_analysis) # <-- Connect new slot
        action_buttons_layout.addWidget(self.grad_cam_btn)
        inference_group_layout.addLayout(action_buttons_layout)


        # (C) Output area (Text)
        self.output_text_label = QLabel("Prediction Output:") # Added label for clarity
        inference_group_layout.addWidget(self.output_text_label)
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setMinimumHeight(100) # Set a minimum height
        inference_group_layout.addWidget(self.output_text)

        # (D) Grad-CAM Display Area (Image)
        self.grad_cam_display_label = QLabel("Grad-CAM output will appear here.") # <-- New QLabel for image
        self.grad_cam_display_label.setObjectName("GradCamDisplayLabel") # For styling
        self.grad_cam_display_label.setAlignment(Qt.AlignCenter)
        inference_group_layout.addWidget(self.grad_cam_display_label)

        main_layout.addWidget(inference_group)
        main_layout.addStretch(1) # Pushes content to top

        # Internals (model references)
        self.inference_model_single = None
        self.inference_model_binary = None
        self.inference_model_multi = None

        # Default hyperparams used in InferenceModel (Consider moving to config or InferenceModel defaults)
        self.input_size = 128  # Or get from loaded model config if possible
        self.normalization = "zscore" # Or get from loaded model config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dropout = 0.0 # Should ideally come from the loaded model's config

        self.setLayout(main_layout)
        self.on_approach_changed()

    # --- Helper to show/hide QFormLayout items properly ---
    def set_form_layout_row_visible(self, form_layout_item_widget, visible):
        """
        Sets the visibility of a widget that is part of a QFormLayout row.
        This typically involves finding the parent widget of the label and field.
        For simplicity, if 'form_layout_item_widget' is the container of a row (e.g. self.single_model_container)
        we just set that container's visibility.
        """
        if form_layout_item_widget:
            form_layout_item_widget.setVisible(visible)


    def on_approach_changed(self):
        approach = self.approach_combo.currentText()
        if "Double" in approach:
            self.set_form_layout_row_visible(self.single_model_container, True)
            self.two_step_container.setVisible(False)
        else: # Two-Step
            self.set_form_layout_row_visible(self.single_model_container, False)
            self.two_step_container.setVisible(True)
        self.output_text.clear()
        self.grad_cam_display_label.clear() # Clear Grad-CAM image too
        self.grad_cam_display_label.setText("Grad-CAM output will appear here.")


    # ... (on_browse_single_model, on_browse_binary_model, etc. REMAIN THE SAME) ...
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
        dlg.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)")
        if dlg.exec_():
            self.img_edit.setText(dlg.selectedFiles()[0])
            self.grad_cam_display_label.clear() # Clear previous Grad-CAM on new image selection
            self.grad_cam_display_label.setText("Grad-CAM output will appear here.")


    def on_load_models(self):
        self.output_text.clear()
        self.grad_cam_display_label.clear()
        self.grad_cam_display_label.setText("Grad-CAM output will appear here.")
        self.inference_model_single = None
        self.inference_model_binary = None
        self.inference_model_multi = None

        # Basic device determination
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_text.append(f"Using device: {self.device}\n")

        approach = self.approach_combo.currentText()
        try:
            if "Double" in approach:
                ckpt_path = self.single_model_edit.text().strip()
                if not ckpt_path or not os.path.isfile(ckpt_path):
                    QMessageBox.warning(self, "Invalid Checkpoint", "Select a valid DoubleHeadCNN checkpoint.")
                    return
                # Assuming InferenceModel can determine input_size, norm, dropout from checkpoint
                # or uses sensible defaults if not found.
                self.inference_model_single = InferenceModel(
                    checkpoint_path=ckpt_path, model_variant="double", device=self.device
                )
                # Update these if InferenceModel exposes them after loading
                # self.input_size = self.inference_model_single.input_size 
                # self.normalization = self.inference_model_single.normalization
                self.output_text.append(f"Double model loaded: {os.path.basename(ckpt_path)}\n")
            else: # Two-Step
                bin_ckpt = self.binary_model_edit.text().strip()
                mul_ckpt = self.multi_model_edit.text().strip()
                if not (bin_ckpt and os.path.isfile(bin_ckpt) and mul_ckpt and os.path.isfile(mul_ckpt)):
                    QMessageBox.warning(self, "Invalid Checkpoint(s)", "Select valid binary and multi checkpoints.")
                    return
                
                self.inference_model_binary = InferenceModel(
                    checkpoint_path=bin_ckpt, model_variant="binary", device=self.device
                )
                self.output_text.append(f"Binary model loaded: {os.path.basename(bin_ckpt)}\n")
                
                self.inference_model_multi = InferenceModel(
                    checkpoint_path=mul_ckpt, model_variant="multi", device=self.device
                )
                self.output_text.append(f"Multi model loaded: {os.path.basename(mul_ckpt)}\n")
            
            self.output_text.append("Model(s) loaded successfully.\n")

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Error loading model(s):\n{str(e)}")
            self.output_text.append(f"Error loading model(s): {str(e)}\n")


    def on_predict(self):
        # ... (on_predict method - mostly the same, just ensure output_text is appended to)
        approach = self.approach_combo.currentText()
        img_path = self.img_edit.text().strip()
        if not img_path or not os.path.isfile(img_path):
            QMessageBox.warning(self, "Invalid Image", "Select a valid image file.")
            return

        try:
            pil_img = Image.open(img_path).convert('L') # Ensure grayscale
        except Exception as e:
            QMessageBox.critical(self, "Image Error", f"Failed to open image:\n{e}")
            return

        current_output = self.output_text.toPlainText()
        self.output_text.clear() # Clear before new prediction, or append if desired
        self.output_text.append(current_output.split("--- Done ---")[0]) # Keep previous loaded model info
        self.output_text.append("<b>Inference Output</b>\n")
        self.output_text.append(f"Image: {os.path.basename(img_path)}\n")


        if "Double" in approach:
            if not self.inference_model_single:
                QMessageBox.warning(self, "Not Loaded", "Load the DoubleHeadCNN model first.")
                return
            try:
                result = self.inference_model_single.predict(pil_img)
                self.display_double_result(result)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Double inference error:\n{e}")
                self.output_text.append(f"Double inference error: {e}\n")
                return
        else: # Two-step
            if not (self.inference_model_binary and self.inference_model_multi):
                QMessageBox.warning(self, "Not Loaded", "Load both binary and multi models first.")
                return
            try:
                bin_result = self.inference_model_binary.predict(pil_img)
                idx = bin_result["class_index"]
                probs = bin_result["probabilities"]
                no_tumor_conf = probs[0] * 100.0 # Assuming class 0 is no_tumor
                tumor_conf = probs[1] * 100.0    # Assuming class 1 is tumor

                if idx == 0: # No tumor
                    self.output_text.append(
                        f"Binary => NO TUMOR (confidence ~{no_tumor_conf:.1f}%)\n"
                    )
                else: # Tumor
                    self.output_text.append(
                        f"Binary => TUMOR (confidence ~{tumor_conf:.1f}%)\n"
                    )
                    mul_result = self.inference_model_multi.predict(pil_img)
                    self.display_multi_result(mul_result)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Two-step inference error:\n{e}")
                self.output_text.append(f"Two-step inference error: {e}\n")
                return
        self.output_text.append("\n--- Done ---\n")

    # --------------------------------------------------------------------------
    # Grad-CAM Analysis
    # --------------------------------------------------------------------------
    def on_grad_cam_analysis(self):
        self.output_text.append("\n--- Starting GRAD-CAM Analysis ---\n")
        img_path = self.img_edit.text().strip()

        if not img_path or not os.path.isfile(img_path):
            QMessageBox.warning(self, "Invalid Image", "Select a valid image file for Grad-CAM.")
            self.output_text.append("Error: Invalid image path for Grad-CAM.\n")
            return

        approach = self.approach_combo.currentText()
        pytorch_model_instance = None
        model_transform = None
        model_to_log = ""

        if "Double" in approach:
            if not self.inference_model_single or not hasattr(self.inference_model_single, 'model') or not hasattr(self.inference_model_single, 'transform'):
                QMessageBox.warning(self, "Model Issue", "Double model not loaded or not compatible for Grad-CAM.")
                self.output_text.append("Error: Double model issue for Grad-CAM.\n")
                return
            pytorch_model_instance = self.inference_model_single.model
            model_transform = self.inference_model_single.transform
            model_to_log = "Double Model"
        else: # Two-Step, typically Grad-CAM on the multi-class model if tumor is detected
            if not self.inference_model_multi or not hasattr(self.inference_model_multi, 'model') or not hasattr(self.inference_model_multi, 'transform'):
                QMessageBox.warning(self, "Model Issue", "Multi-class model not loaded or not compatible for Grad-CAM.")
                self.output_text.append("Error: Multi-class model issue for Grad-CAM.\n")
                return
            pytorch_model_instance = self.inference_model_multi.model
            model_transform = self.inference_model_multi.transform
            model_to_log = "Multi-Class Model (Two-Step)"
            # Optional: could also run on binary if user wants
            # pytorch_model_instance = self.inference_model_binary.model
            # model_transform = self.inference_model_binary.transform


        if not pytorch_model_instance or not model_transform:
            QMessageBox.warning(self, "Internal Error", "Could not retrieve model or transform for Grad-CAM.")
            self.output_text.append("Error: Could not retrieve model/transform.\n")
            return

        # --- Determine Target Layer ---
        # This is crucial and model-dependent. There may be a need to make this more flexible.
        # trying common names or allowing user input.
        target_layer = None
        # Common layer names for CNNs, adjust based on final model architectures
        possible_layer_names = ['conv2', 'layer4', 'features'] # Need to be adjusted while all models done
        final_conv_layer = None

        for name, module in pytorch_model_instance.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                final_conv_layer = module # Keep track of the last conv layer
            for pl_name in possible_layer_names:
                if pl_name in name.split('.'): # Check if part of the name
                    target_layer_candidate = getattr(pytorch_model_instance, name.split('.')[0], None)
                    if hasattr(target_layer_candidate, name.split('.')[-1]):
                         target_layer = target_layer_candidate[-1] if isinstance(target_layer_candidate, torch.nn.Sequential) else getattr(target_layer_candidate, name.split('.')[-1], None)
                         if isinstance(target_layer, torch.nn.Conv2d): break
            if target_layer and isinstance(target_layer, torch.nn.Conv2d): break
        
        if not target_layer and final_conv_layer: # Fallback to last found conv layer
            target_layer = final_conv_layer
            self.output_text.append(f"Warning: Using last convolutional layer found as target for Grad-CAM.\n")

        if not target_layer:
            QMessageBox.critical(self, "Grad-CAM Error", "Could not find a suitable target layer (e.g., 'conv2', 'layer4') in the model.")
            self.output_text.append("Error: Could not find target layer for Grad-CAM.\n")
            return
        
        self.output_text.append(f"Using target layer for Grad-CAM: {target_layer}\n")


        try:
            pil_img = Image.open(img_path).convert('RGB') # Grad-CAM utils often expect RGB
            
            # Preprocess the image using the model's own transform
            # The transform should output a tensor
            input_tensor = model_transform(pil_img).unsqueeze(0).to(self.device)
            if input_tensor.shape[1] == 1: # if grayscale after transform, repeat for RGB
                 input_tensor = input_tensor.repeat(1, 3, 1, 1)


            # Ensure model is in eval mode
            pytorch_model_instance.eval()

            # Initialize GradCAM
            # target_layers should be a list of layer objects
            cam = GradCAM(model=pytorch_model_instance, target_layers=[target_layer], use_cuda=torch.cuda.is_available())
            
            # You can specify the target class for Grad-CAM, or let it pick the predicted class.
            # For simplicity, we use `targets=None` to use the predicted class.
            grayscale_cam = cam(input_tensor=input_tensor, targets=None) # Or specify targets
            grayscale_cam = grayscale_cam[0, :] # Take the first (and only) image's CAM

            # Prepare original image for overlay:
            # For show_cam_on_image, the input_image should be [0,1] range
            rgb_img_for_overlay = np.array(pil_img) / 255.0
            if len(rgb_img_for_overlay.shape) == 2: # Grayscale
                rgb_img_for_overlay = np.stack([rgb_img_for_overlay]*3, axis=-1)


            visualization = show_cam_on_image(rgb_img_for_overlay, grayscale_cam, use_rgb=True)

            # Convert NumPy array (RGB) to QPixmap
            h, w, ch = visualization.shape
            if ch == 3: # RGB
                # Ensure data is uint8 for QImage
                vis_uint8 = (visualization * 255).astype(np.uint8)
                q_img = QImage(vis_uint8.data, w, h, ch * w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                self.grad_cam_display_label.setPixmap(pixmap.scaled(
                    self.grad_cam_display_label.width(),
                    self.grad_cam_display_label.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))
                self.output_text.append(f"Grad-CAM generated for {model_to_log}.\n")

                # Optional: Save the image
                save_dir = "gradcam_outputs_gui"
                os.makedirs(save_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                save_path = os.path.join(save_dir, f"gradcam_{base_name}_{model_to_log.replace(' ','_')}.png")
                # To save the visualization, matplotlib can be used or QPixmap directly
                pixmap.save(save_path)
                self.output_text.append(f"Grad-CAM image saved to: {save_path}\n")

            else:
                self.output_text.append("Error: Grad-CAM visualization is not in RGB format.\n")
                self.grad_cam_display_label.setText("Error in Grad-CAM format.")

        except Exception as e:
            QMessageBox.critical(self, "Grad-CAM Error", f"Failed to generate Grad-CAM:\n{e}")
            self.output_text.append(f"Error during Grad-CAM: {e}\n")
            import traceback
            self.output_text.append(traceback.format_exc()) 
            self.grad_cam_display_label.setText("Grad-CAM Generation Failed.")
        
        self.output_text.append("--- Grad-CAM Analysis Done ---\n")


    
    def display_double_result(self, result: dict):
        if "binary_probs" not in result:
            self.output_text.append("Unexpected result structure from 'double' model.\n")
            return

        bin_probs = result["binary_probs"]
        no_tumor_conf = bin_probs[0] * 100.0
        tumor_conf = bin_probs[1] * 100.0
        self.output_text.append(
            f"Binary head => p(no_tumor)={bin_probs[0]:.3f}, p(tumor)={bin_probs[1]:.3f}\n"
        )
        self.output_text.append(f"Tumor confidence => {tumor_conf:.1f}%\n")

        if result.get("prediction_binary", "no_tumor" if bin_probs[0] > bin_probs[1] else "tumor") == "no_tumor": 
            self.output_text.append("==> Prediction: NO TUMOR\n")
        else:
            predicted_class = result.get("prediction_multi", "Unknown Tumor Type") 
            self.output_text.append(f"==> Prediction: {predicted_class}\n")
            if "multi_probs" in result:
                multi_probs = result["multi_probs"]
                # Assuming multi_probs is a list/array of probabilities
                probs_str = ", ".join([f"{p:.3f}" for p in multi_probs])
                self.output_text.append(f"Subtype probabilities => [{probs_str}]\n")


    def display_multi_result(self, mul_result: dict):
        predicted_class_name = mul_result.get("class_name", f"Class Index {mul_result.get('class_index', 'N/A')}")
        probs = mul_result.get("probabilities", [])
        
        self.output_text.append("Multi-Class Subtype Classification:\n")
        self.output_text.append(f"Predicted Type => {predicted_class_name}\n")
        if probs:
            probs_str = ", ".join([f"{p:.3f}" for p in probs])
            self.output_text.append(f"Probabilities => [{probs_str}]\n")
