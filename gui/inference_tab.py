"""
inference_tab.py

Upgraded InferenceTab with:
 - Collapsible HPC info (ConfMat, ROC, metrics)
 - Single vs. Batch inference with HPC combos
 - Single Grad-CAM (unchanged logic, minor refactoring)
 - NEW: Batch Grad-CAM box
 - Class label fix for "simple" (4-class) => [0=NoTumor,1=Pituitary,2=Meningioma,3=Glioma]
 - HPC references kept for combos (no mention of disable_aug or adv_aug).
"""

import os
import json
import csv
import warnings
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QComboBox, QLineEdit, QPushButton, QHBoxLayout,
    QFileDialog, QTextEdit, QMessageBox, QGroupBox, QFormLayout, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea, QRadioButton,
    QCheckBox, QDialog, QGridLayout, QSlider
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5 import QtGui

# Matplotlib canvas for HPC info plots
from gui.widgets import MplCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

### Attempt to import multiple Grad-CAM variants
try:
    from pytorch_grad_cam import GradCAM, ScoreCAM, XGradCAM, EigenCAM
    from pytorch_grad_cam import GradCAMPlusPlus as GradCAMPP
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    # If not installed, set them to None. We'll fallback to normal GradCAM if possible
    GradCAM = None
    ScoreCAM = None
    XGradCAM = None
    EigenCAM = None
    GradCAMPP = None
    show_cam_on_image = None

import numpy as np

from logic.inference import InferenceModel

_HPC_RESULTS_JSON = "hpc_results.json"

###
### IMPORTANT: For "simple" or "multi" 4-class, we define:
###   0=NoTumor, 1=Pituitary, 2=Meningioma, 3=Glioma
###
CLASS_LABELS_4 = ["notumor", "pituitary", "meningioma", "glioma"]


def load_hpc_results_json(path=_HPC_RESULTS_JSON):
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        combos = []
        for item in data:
            combos.append({
                "combo_str":     item.get("combo_str", ""),
                "val_acc":       item.get("val_acc", 0.0),
                "model_path":    item.get("model_path", ""),
                "model_variant": item.get("model_variant", ""),
                "confmat":       item.get("confmat", None),
                "roc":           item.get("roc", None),
                "metrics":       item.get("metrics", None)
            })
        return combos
    except Exception as e:
        print(f"[WARN] Could not load HPC combos from {path}: {e}")
        return []


class BatchInferenceWorker(QThread):
    """
    Threaded batch inference worker.
    """
    progress_msg = pyqtSignal(str)
    finished_batch = pyqtSignal(list)

    def __init__(self, inference_model, image_paths=None, folder_path=None, export_csv=None, parent=None):
        super().__init__(parent)
        self.inference_model = inference_model
        self.image_paths = image_paths or []
        self.folder_path = folder_path
        self.export_csv = export_csv

    def run(self):
        try:
            results = self.inference_model.predict_batch(
                image_paths=self.image_paths,
                folder_path=self.folder_path,
                export_csv=self.export_csv
            )
            msg = f"Batch inference complete: {len(results)} images processed."
            self.progress_msg.emit(msg)
            self.finished_batch.emit(results)
        except Exception as e:
            self.progress_msg.emit(f"Batch inference error: {e}")
            self.finished_batch.emit([])


### NEW BATCH-GRAD-CAM DIALOG ###
class GradCamDialog(QDialog):
    """
    A simple dialog that shows multiple Grad-CAM results in a grid layout.
    Each selected row from the results table => 1 image.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Grad-CAM Results")
        self.resize(900, 600)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.scroll_area)
        self.setLayout(layout)

        self.container = QWidget()
        self.container_layout = QGridLayout(self.container)
        self.container_layout.setSpacing(6)

        self.scroll_area.setWidget(self.container)

    def add_gradcam_image(self, pixmap, title="", row=0, col=0):
        """
        Adds a labeled image in the grid.
        """
        lbl_title = QLabel(title)
        lbl_title.setAlignment(Qt.AlignCenter)

        lbl_img = QLabel()
        lbl_img.setPixmap(pixmap)
        lbl_img.setAlignment(Qt.AlignCenter)

        self.container_layout.addWidget(lbl_title, row*2, col)
        self.container_layout.addWidget(lbl_img, row*2+1, col)


### MAIN INFERENCE TAB ###
class InferenceTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Basic style
        self.setStyleSheet("""
            QGroupBox {
                border: 1px solid #cccccc;
                border-radius: 6px;
                margin-top: 10px;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                color: #0078D7;
                padding: 4px 10px;
                font-weight: bold;
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
                background-color: #005fa3;
            }
            QTableWidget {
                font-size: 9pt;
                gridline-color: #ccc;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QProgressBar {
                height: 16px;
            }
            QLabel#GradCamDisplayLabel {
                border: 1px solid #ccc;
                background-color: #f0f0f0;
                min-height: 256px;
                min-width: 256px;
                alignment: center;
            }
        """)

        main_layout = QVBoxLayout(self)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(10, 10, 10, 10)
        container_layout.setSpacing(8)

        # HPC combos
        hpc_box = QGroupBox("Select HPC Model")
        vb_hpc = QVBoxLayout(hpc_box)
        self.hpc_combos = load_hpc_results_json()
        self.model_combo = QComboBox()
        vb_hpc.addWidget(QLabel("Available HPC combos:"))
        vb_hpc.addWidget(self.model_combo)
        container_layout.addWidget(hpc_box)

        # HPC info collapsible
        self.chk_show_info = QCheckBox("Expand HPC Info (ConfMat, ROC, Metrics)")
        self.chk_show_info.setChecked(False)
        self.chk_show_info.stateChanged.connect(self._toggle_hpc_info)
        container_layout.addWidget(self.chk_show_info)

        self.info_box = QGroupBox("HPC Model Info")
        vb_info = QVBoxLayout(self.info_box)

        # row of Confmat + ROC
        row_plots = QHBoxLayout()
        self.confmat_canvas = MplCanvas(self, width=4, height=3)
        self.confmat_toolbar = NavigationToolbar(self.confmat_canvas, self)
        left_g = QVBoxLayout()
        left_g.addWidget(QLabel("Confusion Matrix"))
        left_g.addWidget(self.confmat_canvas)
        left_g.addWidget(self.confmat_toolbar)
        row_plots.addLayout(left_g)

        self.roc_canvas = MplCanvas(self, width=4, height=3)
        self.roc_toolbar = NavigationToolbar(self.roc_canvas, self)
        right_g = QVBoxLayout()
        right_g.addWidget(QLabel("ROC (Binary/Double)"))
        right_g.addWidget(self.roc_canvas)
        right_g.addWidget(self.roc_toolbar)
        row_plots.addLayout(right_g)

        vb_info.addLayout(row_plots)

        # HPC metrics
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        vb_info.addWidget(self.metrics_text)

        # Refresh HPC info
        self.btn_show_info = QPushButton("Refresh HPC Info")
        self.btn_show_info.clicked.connect(self.show_hpc_info)
        vb_info.addWidget(self.btn_show_info)

        container_layout.addWidget(self.info_box)
        self.info_box.hide()

        # single vs. batch
        mode_box = QGroupBox("Inference Mode")
        hb_mode = QHBoxLayout(mode_box)
        self.radio_single = QRadioButton("Single Image")
        self.radio_batch = QRadioButton("Batch")
        self.radio_single.setChecked(True)
        hb_mode.addWidget(self.radio_single)
        hb_mode.addWidget(self.radio_batch)
        container_layout.addWidget(mode_box)
        self.radio_single.toggled.connect(self._on_mode_changed)

        # Single group
        self.single_box = QGroupBox("Single Image Inference & Grad-CAM")
        form_s = QFormLayout(self.single_box)

        self.single_img_line = QLineEdit()
        btn_browse_single = QPushButton("Browse Image...")
        btn_browse_single.clicked.connect(self.on_browse_single)
        h_s = QHBoxLayout()
        h_s.addWidget(self.single_img_line)
        h_s.addWidget(btn_browse_single)
        form_s.addRow("Image:", h_s)

        btn_row = QHBoxLayout()
        self.btn_run_single = QPushButton("Run Single Inference")
        self.btn_run_single.clicked.connect(self.on_run_single_inference)
        btn_row.addWidget(self.btn_run_single)

        self.btn_grad_cam = QPushButton("GRAD-CAM Analysis")
        self.btn_grad_cam.clicked.connect(self.on_grad_cam_analysis)
        btn_row.addWidget(self.btn_grad_cam)

        form_s.addRow(btn_row)

        self.grad_cam_display_label = QLabel("Grad-CAM output will appear here.")
        self.grad_cam_display_label.setObjectName("GradCamDisplayLabel")
        self.grad_cam_display_label.setAlignment(Qt.AlignCenter)
        form_s.addRow(self.grad_cam_display_label)

        container_layout.addWidget(self.single_box)

        # Batch group
        self.batch_box = QGroupBox("Batch Inference")
        vb_b = QVBoxLayout(self.batch_box)

        h_fb = QHBoxLayout()
        self.batch_folder_line = QLineEdit()
        btn_folder = QPushButton("Browse Folder...")
        btn_folder.clicked.connect(self.on_browse_batch_folder)
        h_fb.addWidget(self.batch_folder_line)
        h_fb.addWidget(btn_folder)
        vb_b.addLayout(h_fb)

        self.btn_add_images = QPushButton("Add Multiple Images...")
        self.btn_add_images.clicked.connect(self.on_add_batch_images)
        vb_b.addWidget(self.btn_add_images)

        self.batch_images_text = QTextEdit()
        self.batch_images_text.setReadOnly(True)
        vb_b.addWidget(self.batch_images_text)

        self.csv_line = QLineEdit()
        self.csv_line.setPlaceholderText("Optional CSV output, e.g. results.csv")
        vb_b.addWidget(self.csv_line)

        h_runb = QHBoxLayout()
        self.btn_run_batch = QPushButton("Run Batch Inference")
        self.btn_run_batch.clicked.connect(self.on_run_batch_inference)
        self.batch_progress = QProgressBar()
        self.batch_progress.setValue(0)
        h_runb.addWidget(self.btn_run_batch)
        h_runb.addWidget(self.batch_progress)
        vb_b.addLayout(h_runb)

        container_layout.addWidget(self.batch_box)

        # Table for results
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(7)
        self.results_table.setHorizontalHeaderLabels([
            "File", "Predicted",
            "Prob(NoTumor)", "Prob(Pituitary)", "Prob(Meningioma)", "Prob(Glioma)",
            "Error"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        container_layout.addWidget(self.results_table)

        ### NEW BATCH-GRAD-CAM WIDGETS ###
        self.batch_gradcam_box = QGroupBox("Batch Grad-CAM")
        bg_layout = QFormLayout(self.batch_gradcam_box)

        # Method combo
        self.combo_gradcam_method = QComboBox()
        # We'll show these in a user-friendly order. Only activate them if they exist.
        # We'll do a quick check for each import:
        method_list = []
        if GradCAM is not None:
            method_list.append("GradCAM")
        if GradCAMPP is not None:
            method_list.append("GradCAM++")
        if ScoreCAM is not None:
            method_list.append("ScoreCAM")
        if XGradCAM is not None:
            method_list.append("XGradCAM")
        if EigenCAM is not None:
            method_list.append("EigenCAM")
        if not method_list:
            # fallback => no variants installed
            method_list = ["(No Grad-CAM variants installed)"]
        self.combo_gradcam_method.addItems(method_list)
        bg_layout.addRow("Method:", self.combo_gradcam_method)

        # Opacity slider
        self.slider_alpha = QSlider(Qt.Horizontal)
        self.slider_alpha.setRange(0, 100)
        self.slider_alpha.setValue(50)  # 50% overlay
        bg_layout.addRow("Overlay Opacity:", self.slider_alpha)

        # Button
        self.btn_batch_gradcam = QPushButton("Run Batch Grad-CAM (Selected Rows)")
        self.btn_batch_gradcam.clicked.connect(self.on_batch_gradcam)
        bg_layout.addRow(self.btn_batch_gradcam)

        container_layout.addWidget(self.batch_gradcam_box)

        scroll_area.setWidget(container)
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)

        # Internals
        self.current_inference_model = None
        self.batch_images = []
        self.batch_worker = None

        # HPC combos
        self._populate_hpc_combo()
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        self._try_auto_load_first()
        self._on_mode_changed()

    # ----------------------------------------------------------------
    # HPC combos
    # ----------------------------------------------------------------
    def _populate_hpc_combo(self):
        self.model_combo.clear()
        if not self.hpc_combos:
            self.model_combo.addItem("-- No HPC combos found --")
        else:
            for i, rec in enumerate(self.hpc_combos):
                label = f"{i+1}) {rec['combo_str']} (val={rec['val_acc']:.1f}%)"
                self.model_combo.addItem(label)

    def _try_auto_load_first(self):
        if self.hpc_combos:
            self.on_model_changed()

    def on_model_changed(self):
        idx = self.model_combo.currentIndex()
        if not self.hpc_combos or idx < 0 or idx >= len(self.hpc_combos):
            self.current_inference_model = None
            return

        chosen = self.hpc_combos[idx]
        mpath = chosen.get("model_path", "")
        variant = chosen.get("model_variant", "multi")
        if (not mpath) or (not os.path.isfile(mpath)):
            self.current_inference_model = None
            return

        # parse "resize=XYZ" from combo_str, default 128
        resize_val = 128
        for part in chosen["combo_str"].split(","):
            part = part.strip().lower()
            if "resize=" in part:
                try:
                    eqpos = part.index("=")
                    val_s = part[eqpos + 1:].strip()
                    resize_val = int(val_s)
                except:
                    pass

        try:
            self.current_inference_model = InferenceModel(
                checkpoint_path=mpath,
                model_variant=variant,
                input_size=resize_val,
                normalization="zscore",
                device="cuda" if torch.cuda.is_available() else "cpu",
                dropout=0.0
            )
        except Exception as e:
            QMessageBox.critical(self, "Load HPC Model Error", f"Could not load HPC model:\n{e}")
            self.current_inference_model = None

    # ----------------------------------------------------------------
    # HPC info
    # ----------------------------------------------------------------
    def _toggle_hpc_info(self, state):
        if state == Qt.Checked:
            self.info_box.show()
        else:
            self.info_box.hide()

    def show_hpc_info(self):
        if not self.info_box.isVisible():
            return

        self.confmat_canvas.figure.clear()
        self.roc_canvas.figure.clear()
        self.metrics_text.clear()

        idx = self.model_combo.currentIndex()
        if not self.hpc_combos or idx < 0 or idx >= len(self.hpc_combos):
            self.metrics_text.setPlainText("No HPC combos or invalid selection.")
            self.confmat_canvas.draw()
            self.roc_canvas.draw()
            return

        combo = self.hpc_combos[idx]
        confmat = combo.get("confmat", None)
        roc_ = combo.get("roc", None)
        mets = combo.get("metrics", None)

        # ConfMat
        axc = self.confmat_canvas.figure.add_subplot(111)
        if confmat is not None:
            arr = np.array(confmat, dtype=int)
            im = axc.imshow(arr, cmap="Blues")
            axc.set_title("Confusion Matrix", fontsize=10)
            for r in range(arr.shape[0]):
                for c in range(arr.shape[1]):
                    axc.text(c, r, str(arr[r, c]), ha='center', va='center')
            self.confmat_canvas.figure.colorbar(im, ax=axc)
        else:
            axc.text(0.5, 0.5, "No ConfMat found", ha='center', va='center')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.confmat_canvas.figure.tight_layout(pad=0.8)
        self.confmat_canvas.draw()

        # ROC
        axr = self.roc_canvas.figure.add_subplot(111)
        if roc_ and roc_.get("fpr") is not None:
            fpr = roc_["fpr"]
            tpr = roc_["tpr"]
            aucv = roc_.get("auc", 0.0)
            axr.plot(fpr, tpr, color='red', label=f"AUC={aucv:.2f}")
            axr.plot([0, 1], [0, 1], "--", color='gray')
            axr.set_title("ROC Curve", fontsize=10)
            axr.legend(fontsize=8)
        else:
            axr.text(0.5, 0.5, "No ROC (non-binary/double?)", ha='center', va='center')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.roc_canvas.figure.tight_layout(pad=0.8)
        self.roc_canvas.draw()

        # metrics
        if not mets:
            self.metrics_text.setPlainText("No HPC metrics found.")
        else:
            lines = []
            prec = mets.get("precision", [])
            rec_ = mets.get("recall", [])
            f1_ = mets.get("f1", [])
            sup_ = mets.get("support", [])
            for i in range(len(prec)):
                lines.append(f"Class {i}: prec={prec[i]:.3f}, rec={rec_[i]:.3f}, f1={f1_[i]:.3f}, sup={sup_[i]}")
            micro = mets.get("micro_f1", None)
            macro = mets.get("macro_f1", None)
            if micro is not None:
                lines.append(f"\nMicro-F1: {micro:.3f}")
            if macro is not None:
                lines.append(f"Macro-F1: {macro:.3f}")
            self.metrics_text.setPlainText("\n".join(lines))

    # ----------------------------------------------------------------
    # Single vs. Batch
    # ----------------------------------------------------------------
    def _on_mode_changed(self):
        if self.radio_single.isChecked():
            self.single_box.setVisible(True)
            self.batch_box.setVisible(False)
        else:
            self.single_box.setVisible(False)
            self.batch_box.setVisible(True)

    # ----------------------------------------------------------------
    # Single
    # ----------------------------------------------------------------
    def on_browse_single(self):
        dlg = QFileDialog(self, "Select Single Image")
        dlg.setNameFilter("Images (*.png *.jpg *.jpeg)")
        if dlg.exec_():
            self.single_img_line.setText(dlg.selectedFiles()[0])

    def on_run_single_inference(self):
        if not self.current_inference_model:
            QMessageBox.warning(self, "No HPC Model", "Pick HPC combo above.")
            return

        path = self.single_img_line.text().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "Invalid Image", "Please pick a valid image.")
            return

        self.results_table.setRowCount(0)

        try:
            pil_img = Image.open(path).convert('L')
        except Exception as e:
            QMessageBox.critical(self, "Image Error", f"Could not open: {e}")
            return

        try:
            res = self.current_inference_model.predict(pil_img)
            self.results_table.setRowCount(1)
            self._fill_table_row(0, path, res)
        except Exception as e:
            QMessageBox.critical(self, "Inference Error", str(e))

    # ----------------------------------------------------------------
    # Single Grad-CAM
    # ----------------------------------------------------------------
    def on_grad_cam_analysis(self):
        if GradCAM is None or show_cam_on_image is None:
            QMessageBox.warning(self, "Missing grad-cam", "pytorch-grad-cam is not installed or incomplete.")
            return

        if not self.current_inference_model or not hasattr(self.current_inference_model, 'model'):
            QMessageBox.warning(self, "No Model", "No HPC model is currently loaded.")
            return

        img_path = self.single_img_line.text().strip()
        if not img_path or not os.path.isfile(img_path):
            QMessageBox.warning(self, "Invalid Image", "Please specify an image for Grad-CAM.")
            return

        try:
            # We'll just pick standard GradCAM for single analysis
            # (unchanged logic from original)
            cam = self._build_gradcam_variant(
                method_name="GradCAM",
                fallback_ok=True
            )
            if cam is None:
                QMessageBox.warning(self, "No GradCAM", "No Grad-CAM variant is installed.")
                return

            # generate
            visualization = self._run_gradcam_on_path(img_path, cam)
            if visualization is None:
                return

            # display
            vis_uint8 = visualization.astype(np.uint8)
            h2, w2, ch = vis_uint8.shape
            if ch == 3:
                from PyQt5.QtGui import QImage, QPixmap
                q_img = QImage(vis_uint8.data, w2, h2, ch*w2, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img).scaled(
                    self.grad_cam_display_label.width(),
                    self.grad_cam_display_label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.grad_cam_display_label.setPixmap(pixmap)
            else:
                self.grad_cam_display_label.setText("Grad-CAM result not in RGB format.")
        except Exception as e:
            QMessageBox.critical(self, "Grad-CAM Error", f"Error generating Grad-CAM:\n{str(e)}")

    # ----------------------------------------------------------------
    # Batch
    # ----------------------------------------------------------------
    def on_browse_batch_folder(self):
        dlg = QFileDialog(self, "Select Folder for Batch")
        dlg.setFileMode(QFileDialog.Directory)
        if dlg.exec_():
            self.batch_folder_line.setText(dlg.selectedFiles()[0])

    def reload_hpc_combos(self):
        self.hpc_combos = load_hpc_results_json()
        self._populate_hpc_combo()
        if self.hpc_combos:
            self.model_combo.setCurrentIndex(0)
        else:
            self.model_combo.setCurrentIndex(-1)

    def on_add_batch_images(self):
        dlg = QFileDialog(self, "Add Multiple Images")
        dlg.setFileMode(QFileDialog.ExistingFiles)
        dlg.setNameFilter("Images (*.png *.jpg *.jpeg)")
        if dlg.exec_():
            for f in dlg.selectedFiles():
                if f not in self.batch_images:
                    self.batch_images.append(f)
            self._refresh_batch_images()

    def _refresh_batch_images(self):
        self.batch_images_text.clear()
        for p in self.batch_images:
            self.batch_images_text.append(p)

    def on_run_batch_inference(self):
        if not self.current_inference_model:
            QMessageBox.warning(self, "No HPC Model", "Pick HPC combo first.")
            return

        folder = self.batch_folder_line.text().strip()
        if (not folder) and (not self.batch_images):
            QMessageBox.warning(self, "No Input",
                                "Either pick a folder or add images for batch inference.")
            return

        self.results_table.setRowCount(0)
        self.batch_progress.setValue(0)
        export_csv = self.csv_line.text().strip() or None

        self.batch_worker = BatchInferenceWorker(
            inference_model=self.current_inference_model,
            image_paths=self.batch_images if folder == "" else None,
            folder_path=folder if os.path.isdir(folder) else None,
            export_csv=export_csv
        )
        self.batch_worker.progress_msg.connect(self._on_batch_msg)
        self.batch_worker.finished_batch.connect(self._on_batch_finished)
        self.batch_worker.start()

    def _on_batch_msg(self, msg):
        QMessageBox.information(self, "Batch Info", msg)

    def _on_batch_finished(self, results):
        self.batch_progress.setValue(100)
        self.results_table.setRowCount(len(results))
        for i, rdict in enumerate(results):
            fpath = rdict.get("file", "")
            self._fill_table_row(i, fpath, rdict)
        self.batch_worker.deleteLater()
        self.batch_worker = None

    # ----------------------------------------------------------------
    # Fill table
    # ----------------------------------------------------------------
    def _fill_table_row(self, row_idx, file_path, res_dict):
        """
        We assume 4 columns for "NoTumor, Pituitary, Meningioma, Glioma" in that order,
        plus a "Predicted" column.
        """
        file_item = QTableWidgetItem(file_path)
        self.results_table.setItem(row_idx, 0, file_item)

        err_msg = res_dict.get("error", "")
        if err_msg:
            # entire row red
            for col in range(self.results_table.columnCount()):
                item = QTableWidgetItem("" if col != 0 else file_path)
                item.setForeground(QtGui.QColor("red"))
                if col == 6:
                    item.setText(err_msg)
                self.results_table.setItem(row_idx, col, item)
            return

        pred_label, probs_4 = self._interpret_result(res_dict)

        # predicted
        pred_item = QTableWidgetItem(pred_label)
        if pred_label.lower() == "notumor":
            pred_item.setForeground(QtGui.QColor("green"))
        else:
            pred_item.setForeground(QtGui.QColor("darkorange"))
        self.results_table.setItem(row_idx, 1, pred_item)

        # probabilities
        for i in range(4):
            val_s = "N/A"
            if probs_4[i] is not None:
                val_s = f"{probs_4[i] * 100:.1f}%"
            prob_item = QTableWidgetItem(val_s)
            self.results_table.setItem(row_idx, 2 + i, prob_item)

        # error col
        self.results_table.setItem(row_idx, 6, QTableWidgetItem(""))

    def _interpret_result(self, res_dict):
        """
        Return (pred_label, [pNoTumor, pPituitary, pMening, pGlioma]).
        For 'double': 
           "prediction": "no_tumor" or "tumor_subtype_X"
        For 'binary': 
           "class_index": 0..1
        For 'multi/simple': 
           "class_index": 0..3
        """
        out_probs = [None, None, None, None]
        pred_lbl = "Unknown"

        if "error" in res_dict:
            return pred_lbl, out_probs

        # double
        if "prediction" in res_dict:
            # "no_tumor" => index=0 for "NoTumor" in our array
            # "tumor_subtype_0 => Pituitary, 1=>Meningioma, 2=>Glioma"
            pred = res_dict["prediction"].lower()
            if pred == "no_tumor":
                pred_lbl = "notumor"
                bin_p = res_dict.get("binary_probs", [1.0, 0.0])
                out_probs[0] = float(bin_p[0])  # NoTumor => store in index 0
                out_probs[1] = 0.0
                out_probs[2] = 0.0
                out_probs[3] = 0.0
            else:
                # tumor_subtype_X
                idx_s = pred.rsplit("_", 1)[-1]
                try:
                    sub_i = int(idx_s)
                except:
                    sub_i = 0
                # sub_i=0 => Pituitary, sub_i=1 => Meningioma, sub_i=2 => Glioma
                if sub_i == 0:
                    pred_lbl = "pituitary"
                    out_probs[1] = 1.0
                elif sub_i == 1:
                    pred_lbl = "meningioma"
                    out_probs[2] = 1.0
                else:
                    pred_lbl = "glioma"
                    out_probs[3] = 1.0

        elif "class_index" in res_dict and "probabilities" in res_dict:
            cid = int(res_dict["class_index"])
            probs = res_dict["probabilities"]

            if len(probs) == 4:
                # simple => 0=NoTumor, 1=Pituitary, 2=Meningioma, 3=Glioma
                pred_lbl = CLASS_LABELS_4[cid] if 0 <= cid < 4 else "Unknown"
                for i in range(4):
                    out_probs[i] = float(probs[i])
            elif len(probs) == 2:
                # binary => class0=NoTumor, class1=Tumor
                if cid == 0:
                    pred_lbl = "notumor"
                    out_probs[0] = float(probs[0])
                else:
                    pred_lbl = "tumor"
                    out_probs[1] = float(probs[1])  # or out_probs[2], depends on preference
        return pred_lbl, out_probs

    # ----------------------------------------------------------------
    # NEW: BATCH-GRAD-CAM
    # ----------------------------------------------------------------
    def on_batch_gradcam(self):
        """
        Run Grad-CAM on selected rows in the results_table, 
        then display them in a GradCamDialog.
        """
        if GradCAM is None or show_cam_on_image is None:
            QMessageBox.warning(self, "Missing grad-cam", "pytorch-grad-cam is not installed (or partial).")
            return

        if not self.current_inference_model or not hasattr(self.current_inference_model, 'model'):
            QMessageBox.warning(self, "No Model", "No HPC model is loaded.")
            return

        method_name = self.combo_gradcam_method.currentText()
        if method_name.startswith("("):
            QMessageBox.warning(self, "No Grad-CAM", "No Grad-CAM variant installed.")
            return

        rows = self.results_table.selectionModel().selectedRows()
        if not rows:
            QMessageBox.information(self, "No Selection", "Please select one or more rows in the table.")
            return

        alpha_slider_val = self.slider_alpha.value()  # 0..100 => compute alpha fraction
        alpha_fraction = float(alpha_slider_val)/100.0

        # create CAM object
        cam = self._build_gradcam_variant(method_name, fallback_ok=False)
        if cam is None:
            QMessageBox.warning(self, "Unavailable", f"Grad-CAM variant {method_name} is not available.")
            return

        # We'll generate each result, store as pixmap, then show in a QDialog grid
        diag = GradCamDialog(self)
        row_idx = 0
        col_idx = 0

        for sel_row in rows:
            r = sel_row.row()
            file_item = self.results_table.item(r, 0)
            if not file_item:
                continue
            path = file_item.text()
            if not os.path.isfile(path):
                continue

            visualization = self._run_gradcam_on_path(path, cam, alpha=alpha_fraction)
            if visualization is None:
                continue

            # convert to QPixmap
            vis_uint8 = visualization.astype(np.uint8)
            h2, w2, ch = vis_uint8.shape
            if ch != 3:
                # skip
                continue
            from PyQt5.QtGui import QImage, QPixmap
            q_img = QImage(vis_uint8.data, w2, h2, ch*w2, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img).scaled(256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # add to dialog
            title_str = os.path.basename(path)
            diag.add_gradcam_image(pixmap, title=title_str, row=row_idx, col=col_idx)

            col_idx += 1
            if col_idx >= 3:  # 3 images per row, for example
                col_idx = 0
                row_idx += 1

        diag.exec_()

    def _build_gradcam_variant(self, method_name="GradCAM", fallback_ok=True):
        """
        Builds the chosen Grad-CAM variant object. If not found and fallback_ok=True,
        returns a standard GradCAM if possible.
        """
        pytorch_model_instance = self.current_inference_model.model
        pytorch_model_instance.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pytorch_model_instance.to(device)

        # find the final conv layer
        final_conv = None
        for name, module in pytorch_model_instance.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                final_conv = module
        if not final_conv:
            return None

        # pick method
        if method_name == "GradCAM" and GradCAM is not None:
            return GradCAM(model=pytorch_model_instance, target_layers=[final_conv])
        elif method_name == "GradCAM++" and GradCAMPP is not None:
            return GradCAMPP(model=pytorch_model_instance, target_layers=[final_conv])
        elif method_name == "ScoreCAM" and ScoreCAM is not None:
            return ScoreCAM(model=pytorch_model_instance, target_layers=[final_conv])
        elif method_name == "XGradCAM" and XGradCAM is not None:
            return XGradCAM(model=pytorch_model_instance, target_layers=[final_conv])
        elif method_name == "EigenCAM" and EigenCAM is not None:
            return EigenCAM(model=pytorch_model_instance, target_layers=[final_conv])

        if fallback_ok and GradCAM is not None:
            # fallback
            return GradCAM(model=pytorch_model_instance, target_layers=[final_conv])
        return None

    def _run_gradcam_on_path(self, img_path, cam, alpha=0.5):
        """
        Helper that:
         1) opens the image in grayscale
         2) transforms => shape [1,1,H,W]
         3) runs cam(input_tensor)
         4) overlays the mask on the original
         5) returns a float32 array [H,W,3] in range 0..255
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            pil_img = Image.open(img_path).convert('L')
        except Exception as e:
            QMessageBox.critical(self, "Open Error", f"Could not open {img_path}: {e}")
            return None

        model_transform = self.current_inference_model.transform
        input_tensor = model_transform(pil_img).unsqueeze(0).to(device)

        # run
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]  # first batch

        # Convert original to [H,W,3] float32
        gray_arr = np.array(pil_img, dtype=np.float32) / 255.0
        rgb_img_np = np.stack([gray_arr]*3, axis=-1)  # shape [H,W,3]

        # resize CAM if needed
        h, w, _ = rgb_img_np.shape
        import cv2
        cam_resized = cv2.resize(grayscale_cam, (w, h), interpolation=cv2.INTER_LINEAR)

        # use show_cam_on_image with user-chosen alpha
        # show_cam_on_image expects a float32 RGB in [0..1], returns uint8
        overlay = show_cam_on_image(rgb_img_np, cam_resized, use_rgb=True, image_weight=(1.0 - alpha))
        return overlay

