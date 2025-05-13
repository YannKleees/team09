"""
inference_tab.py

An InferenceTab that lists *all* HPC combos from JSON in a QComboBox,
including a synthetic "two-step double" model if we find at least one
binary HPC combo and one threeclass HPC combo. No separate training or
checkpoint is needed; at inference time we use both HPC combos.

Single vs. Batch inference is included, plus optional Grad-CAM. 
Class label fix for "simple" => [0=NoTumor,1=Pituitary,2=Meningioma,3=Glioma].
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

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from gui.widgets import MplCanvas

# Attempt Grad-CAM variants
try:
    from pytorch_grad_cam import GradCAM, ScoreCAM, XGradCAM, EigenCAM
    from pytorch_grad_cam import GradCAMPlusPlus as GradCAMPP
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    GradCAM = None
    ScoreCAM = None
    XGradCAM = None
    EigenCAM = None
    GradCAMPP = None
    show_cam_on_image = None

import numpy as np

# Our inference logic
from logic.inference import InferenceModel
# Two-step pipeline that merges best binary + best threeclass
from logic.inference_two_step import TwoStepDoublePipeline

_HPC_RESULTS_JSON = "hpc_results.json"

# For 4-class "simple" => [0=NoTumor,1=Pituitary,2=Meningioma,3=Glioma]
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


########################################################################
# BatchInferenceWorker
########################################################################
class BatchInferenceWorker(QThread):
    """ Threaded batch inference on multiple images or a folder. """
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


########################################################################
# GradCamDialog
########################################################################
class GradCamDialog(QDialog):
    """ Show multiple Grad-CAM images in a grid, for batch results. """
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
        """Add labeled Grad-CAM image in the grid."""
        lbl_title = QLabel(title)
        lbl_title.setAlignment(Qt.AlignCenter)

        lbl_img = QLabel()
        lbl_img.setPixmap(pixmap)
        lbl_img.setAlignment(Qt.AlignCenter)

        self.container_layout.addWidget(lbl_title, row*2, col)
        self.container_layout.addWidget(lbl_img, row*2+1, col)


########################################################################
# A synthetic two-step "model" that merges best binary + best threeclass
########################################################################
class TwoStepModelWrapper:
    """
    A simple wrapper so it acts like 'inference_model' with .predict() / .predict_batch(),
    but internally uses the TwoStepDoublePipeline.
    """
    def __init__(self, binary_ckpt, three_ckpt, input_size=128, device="cpu"):
        self.pipeline = TwoStepDoublePipeline(
            binary_checkpoint=binary_ckpt,
            three_checkpoint=three_ckpt,
            input_size=input_size,
            device=device,
            dropout=0.0
        )

    def predict_batch(self, image_paths=None, folder_path=None, export_csv=None):
        """
        We'll replicate the InferenceModel’s approach: just run predict_image() on each.
        Return list of dicts: { "file":..., "prediction":..., etc. }
        """
        from PIL import Image
        results = []
        files_list = []
        if image_paths:
            files_list = list(image_paths)
        elif folder_path:
            for f in os.listdir(folder_path):
                ext = f.lower()
                if ext.endswith((".png",".jpg",".jpeg")):
                    files_list.append(os.path.join(folder_path,f))

        for path in files_list:
            if not os.path.isfile(path):
                continue
            try:
                pil_img = Image.open(path).convert("L")
                res = self.predict(pil_img)
                res["file"] = path
                results.append(res)
            except Exception as e:
                results.append({"file":path, "error":f"{e}"})

        # optionally CSV
        if export_csv:
            self._export_results_csv(results, export_csv)
        return results

    def predict(self, pil_img):
        """
        Return a dictionary that mimics the "double" HPC style:
          {
            "prediction": "no_tumor" or "pituitary"/"meningioma"/"glioma",
            "binary_probs": [...],
          }
        """
        res = self.pipeline.predict_image(pil_img)
        if res["final_prediction"].lower() == "notumor":
            return {
                "prediction": "no_tumor",
                "binary_probs": res["binary_prob"],
                "multi_probs": None
            }
        else:
            # (pituitary/meningioma/glioma)
            pred_label = res["final_prediction"].lower()
            return {
                "prediction": f"tumor_subtype_{pred_label}",
                "binary_probs": res["binary_prob"],
                "multi_probs": res["threeclass_prob"]
            }

    def _export_results_csv(self, results, csv_path):
        fieldnames = [
            "file","prediction","binary_probs","multi_probs","error"
        ]
        try:
            import csv
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in results:
                    writer.writerow(row)
        except Exception as e:
            print(f"[WARN] Could not export CSV for two-step: {e}")


########################################################################
# InferenceTab
########################################################################
class InferenceTab(QWidget):
    """
    InferenceTab that lists HPC combos in a QComboBox, so you can pick
    which HPC model to load. Also supports single/batch inference, Grad-CAM,
    and a reload_hpc_combos() method so the HPC combos can be refreshed.
    
    If there's at least one "binary" HPC + one "threeclass" HPC,
    we create an extra combo item => "Double (two-step)".
    Selecting that uses TwoStepModelWrapper to do a two-step pipeline.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

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
        container_layout.setSpacing(8)
        container_layout.setContentsMargins(10, 10, 10, 10)

        # HPC combos
        hpc_box = QGroupBox("Select HPC Model")
        vb_hpc = QVBoxLayout(hpc_box)
        self.hpc_combos = []  # loaded from JSON
        self.model_combo = QComboBox()
        vb_hpc.addWidget(QLabel("Available HPC combos:"))
        vb_hpc.addWidget(self.model_combo)

        self.btn_reload = QPushButton("Reload HPC Combos from JSON")
        self.btn_reload.clicked.connect(self.reload_hpc_combos)
        vb_hpc.addWidget(self.btn_reload)

        container_layout.addWidget(hpc_box)

        # HPC info collapsible
        self.chk_show_info = QCheckBox("Expand HPC Info (ConfMat, ROC, Metrics)")
        self.chk_show_info.setChecked(False)
        self.chk_show_info.stateChanged.connect(self._toggle_hpc_info)
        container_layout.addWidget(self.chk_show_info)

        self.info_box = QGroupBox("HPC Model Info")
        vb_info = QVBoxLayout(self.info_box)

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

        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        vb_info.addWidget(self.metrics_text)

        self.btn_show_info = QPushButton("Refresh HPC Info (Selected Combo)")
        self.btn_show_info.clicked.connect(self.show_hpc_info)
        vb_info.addWidget(self.btn_show_info)

        container_layout.addWidget(self.info_box)
        self.info_box.hide()

        # Single vs. Batch radio
        mode_box = QGroupBox("Inference Mode")
        hb_mode = QHBoxLayout(mode_box)
        self.radio_single = QRadioButton("Single Image")
        self.radio_batch = QRadioButton("Batch")
        self.radio_single.setChecked(True)
        hb_mode.addWidget(self.radio_single)
        hb_mode.addWidget(self.radio_batch)
        container_layout.addWidget(mode_box)
        self.radio_single.toggled.connect(self._on_mode_changed)

        # Single Inference Box
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

        # Batch Inference Box
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

        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(7)
        self.results_table.setHorizontalHeaderLabels([
            "File", "Predicted",
            "Prob(NoTumor)", "Prob(Pituitary)", "Prob(Meningioma)", "Prob(Glioma)",
            "Error"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        container_layout.addWidget(self.results_table)

        # Batch Grad-CAM area
        self.batch_gradcam_box = QGroupBox("Batch Grad-CAM")
        bg_layout = QFormLayout(self.batch_gradcam_box)

        self.combo_gradcam_method = QComboBox()
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
            method_list = ["(No Grad-CAM variants installed)"]
        self.combo_gradcam_method.addItems(method_list)
        bg_layout.addRow("Method:", self.combo_gradcam_method)

        self.slider_alpha = QSlider(Qt.Horizontal)
        self.slider_alpha.setRange(0, 100)
        self.slider_alpha.setValue(50)
        bg_layout.addRow("Overlay Opacity:", self.slider_alpha)

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

        # Load HPC combos initially
        self.reload_hpc_combos()
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)

        # default mode: single
        self._on_mode_changed()

    #####################################################################
    # HPC combos reloading + building "two-step double" if possible
    #####################################################################
    def reload_hpc_combos(self):
        combos = load_hpc_results_json()
        self.hpc_combos = combos[:]  # direct copy

        # Attempt to find best "binary" HPC & best "threeclass" HPC
        best_bin = None
        best_bin_acc = 0.0
        best_three = None
        best_three_acc = 0.0

        for c in combos:
            var = c.get("model_variant","")
            va = c.get("val_acc",0.0)
            if var=="binary" and va>best_bin_acc:
                best_bin = c
                best_bin_acc = va
            elif var=="threeclass" and va>best_three_acc:
                best_three = c
                best_three_acc = va

        # If both found, create a synthetic HPC combo for "double_2step"
        if best_bin and best_three:
            # We'll store a fake combo
            # val_acc = average of the two or just min. We'll do average for show
            double_acc = (best_bin_acc + best_three_acc)/2.0
            synthetic_combo = {
                "combo_str": "Double (two-step from best Binary + Threeclass)",
                "val_acc": double_acc,
                "model_path": "__two_step_fake__",  # dummy
                "model_variant": "double_2step",
                "confmat": None,
                "roc": None,
                "metrics": None
            }
            self.hpc_combos.append(synthetic_combo)

        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        if not self.hpc_combos:
            self.model_combo.addItem("-- No HPC combos found --")
            self.current_inference_model = None
        else:
            for i, rec in enumerate(self.hpc_combos):
                label = f"{i+1}) {rec['combo_str']} (val={rec['val_acc']:.1f}%)"
                self.model_combo.addItem(label)
        self.model_combo.blockSignals(False)

        if self.hpc_combos:
            self.model_combo.setCurrentIndex(0)
            self.on_model_changed()
        else:
            self.info_box.hide()

    def on_model_changed(self):
        row_idx = self.model_combo.currentIndex()
        if not self.hpc_combos or row_idx<0 or row_idx>=len(self.hpc_combos):
            self.current_inference_model = None
            return

        chosen = self.hpc_combos[row_idx]
        variant = chosen.get("model_variant","multi")
        mpath = chosen.get("model_path","")
        # If normal HPC combo
        if variant!="double_2step":
            # normal single HPC checkpoint
            if (not mpath) or (not os.path.isfile(mpath)):
                self.current_inference_model = None
                return

            # Attempt parse "resize=XYZ" from combo_str for input_size
            resize_val = 128
            for part in chosen["combo_str"].split(","):
                part_l = part.strip().lower()
                if "resize=" in part_l:
                    try:
                        eqpos = part_l.index("=")
                        val_s = part_l[eqpos+1:].strip()
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

        else:
            # This is the synthetic two-step combo => build from best binary + best threeclass
            # We already found them in reload_hpc_combos. Let's do it again quickly:
            combos = load_hpc_results_json()
            best_bin = None
            best_bin_acc = 0.0
            best_bin_resize = 128
            best_bin_path = None

            best_thr = None
            best_thr_acc = 0.0
            best_thr_resize = 128
            best_thr_path = None

            for c in combos:
                var = c["model_variant"]
                va = c["val_acc"]
                if var=="binary" and va>best_bin_acc and os.path.isfile(c["model_path"]):
                    best_bin_acc = va
                    best_bin = c
                elif var=="threeclass" and va>best_thr_acc and os.path.isfile(c["model_path"]):
                    best_thr_acc = va
                    best_thr = c

            if best_bin and best_thr:
                # parse out the best bin resize
                for part in best_bin["combo_str"].split(","):
                    pl = part.strip().lower()
                    if "resize=" in pl:
                        try:
                            eq = pl.index("=")
                            rv = pl[eq+1:].strip()
                            best_bin_resize = int(rv)
                        except:
                            pass
                # parse out the best thr resize
                for part in best_thr["combo_str"].split(","):
                    pl = part.strip().lower()
                    if "resize=" in pl:
                        try:
                            eq = pl.index("=")
                            rv = pl[eq+1:].strip()
                            best_thr_resize = int(rv)
                        except:
                            pass

                # We'll pick the smaller or bigger? We'll just pick 128 or the first
                # Actually we can pick e.g. best_bin_resize if you want both consistent
                # For simplicity, pick the bin's resize:
                chosen_resize = best_bin_resize

                try:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    self.current_inference_model = TwoStepModelWrapper(
                        binary_ckpt=best_bin["model_path"],
                        three_ckpt=best_thr["model_path"],
                        input_size=chosen_resize,
                        device=device
                    )
                except Exception as e:
                    QMessageBox.critical(self, "Two-Step Error", f"Could not create two-step pipeline:\n{e}")
                    self.current_inference_model = None
            else:
                self.current_inference_model = None

    #####################################################################
    # HPC Info
    #####################################################################
    def _toggle_hpc_info(self, state):
        if state==Qt.Checked:
            self.info_box.show()
        else:
            self.info_box.hide()

    def show_hpc_info(self):
        if not self.info_box.isVisible():
            return

        self.confmat_canvas.figure.clear()
        self.roc_canvas.figure.clear()
        self.metrics_text.clear()

        row_idx = self.model_combo.currentIndex()
        if not self.hpc_combos or row_idx<0 or row_idx>=len(self.hpc_combos):
            self.metrics_text.setPlainText("No HPC combos or invalid selection.")
            self.confmat_canvas.draw()
            self.roc_canvas.draw()
            return

        combo = self.hpc_combos[row_idx]
        # If it's our synthetic "double_2step", there's no confmat
        if combo["model_variant"]=="double_2step":
            self.metrics_text.setPlainText("This is a two-step synthetic model. No HPC metrics.")
            self.confmat_canvas.draw()
            self.roc_canvas.draw()
            return

        confmat = combo.get("confmat",None)
        roc_ = combo.get("roc",None)
        mets = combo.get("metrics",None)

        # ConfMat
        axc = self.confmat_canvas.figure.add_subplot(111)
        if confmat is not None:
            arr = np.array(confmat, dtype=int)
            im = axc.imshow(arr, cmap="Blues")
            axc.set_title("Confusion Matrix", fontsize=10)
            for r in range(arr.shape[0]):
                for c in range(arr.shape[1]):
                    axc.text(c, r, str(arr[r,c]), ha='center', va='center')
            self.confmat_canvas.figure.colorbar(im, ax=axc)
        else:
            axc.text(0.5,0.5,"No ConfMat found", ha='center', va='center')

        self.confmat_canvas.figure.tight_layout()
        self.confmat_canvas.draw()

        # ROC
        axr = self.roc_canvas.figure.add_subplot(111)
        if roc_ and roc_.get("fpr") is not None:
            fpr = roc_["fpr"]
            tpr = roc_["tpr"]
            aucv = roc_.get("auc",0.0)
            axr.plot(fpr, tpr, color='red', label=f"AUC={aucv:.2f}")
            axr.plot([0,1],[0,1],"--", color='gray')
            axr.set_title("ROC Curve", fontsize=10)
            axr.legend(fontsize=8)
        else:
            axr.text(0.5,0.5,"No ROC (maybe not binary)", ha='center', va='center')

        self.roc_canvas.figure.tight_layout()
        self.roc_canvas.draw()

        # metrics
        if not mets:
            self.metrics_text.setPlainText("No HPC metrics found.")
        else:
            lines = []
            prec = mets.get("precision",[])
            rec_ = mets.get("recall",[])
            f1_ = mets.get("f1",[])
            sup_ = mets.get("support",[])
            for i in range(len(prec)):
                lines.append(f"Class {i}: p={prec[i]:.3f}, r={rec_[i]:.3f}, f1={f1_[i]:.3f}, sup={sup_[i]}")
            micro = mets.get("micro_f1",None)
            macro = mets.get("macro_f1",None)
            if micro is not None:
                lines.append(f"\nMicro-F1: {micro:.3f}")
            if macro is not None:
                lines.append(f"Macro-F1: {macro:.3f}")
            self.metrics_text.setPlainText("\n".join(lines))

    #####################################################################
    # Single vs. Batch
    #####################################################################
    def _on_mode_changed(self):
        if self.radio_single.isChecked():
            self.single_box.setVisible(True)
            self.batch_box.setVisible(False)
        else:
            self.single_box.setVisible(False)
            self.batch_box.setVisible(True)

    #####################################################################
    # Single Inference
    #####################################################################
    def on_browse_single(self):
        dlg = QFileDialog(self, "Select Single Image")
        dlg.setNameFilter("Images (*.png *.jpg *.jpeg)")
        if dlg.exec_():
            self.single_img_line.setText(dlg.selectedFiles()[0])

    def on_run_single_inference(self):
        if not self.current_inference_model:
            QMessageBox.warning(self, "No HPC Model", "Pick HPC combo above (none loaded).")
            return

        path = self.single_img_line.text().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "Invalid Image", "Please pick a valid image file.")
            return

        self.results_table.setRowCount(0)

        try:
            pil_img = Image.open(path).convert('L')
        except Exception as e:
            QMessageBox.critical(self, "Image Error", f"Could not open image:\n{e}")
            return

        try:
            # We'll do a standard dictionary interpretation
            # Some two-step logic returns {"prediction":..., "binary_probs":..., etc.}
            # The single-head logic returns "class_index", ...
            # We'll unify them in _fill_table_row
            if hasattr(self.current_inference_model, "predict"):
                res_dict = self.current_inference_model.predict(pil_img)
            else:
                raise ValueError("Current model does not have a .predict() method")

            res_dict["file"] = path
            self.results_table.setRowCount(1)
            self._fill_table_row(0, path, res_dict)
        except Exception as e:
            QMessageBox.critical(self, "Inference Error", str(e))

    def on_grad_cam_analysis(self):
        """
        Grad-CAM on a single image using the standard “GradCAM” approach.
        For “double_2step,” we won't do a meaningful GradCAM (no single net).
        We'll just show a message. Adjust if needed.
        """
        if GradCAM is None or show_cam_on_image is None:
            QMessageBox.warning(self, "Missing grad-cam", "pytorch-grad-cam not installed.")
            return
        if not self.current_inference_model:
            QMessageBox.warning(self, "No Model", "No HPC model is currently loaded.")
            return

        path = self.single_img_line.text().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "Invalid Image", "Please specify an image for Grad-CAM.")
            return

        # If user selected the synthetic “double_2step,” there's no single CNN to run Grad-CAM on.
        # We'll just show a message. Or skip.
        # For demonstration, we detect that variant:
        row_idx = self.model_combo.currentIndex()
        if row_idx>=0 and row_idx<len(self.hpc_combos):
            if self.hpc_combos[row_idx]["model_variant"]=="double_2step":
                QMessageBox.information(self, "Two-step", "Grad-CAM not supported in two-step pipeline.")
                return

        try:
            # We'll do standard GradCAM if the model is single HPC
            cam = self._build_gradcam_variant(method_name="GradCAM", fallback_ok=True)
            if cam is None:
                QMessageBox.warning(self, "No GradCAM", "No Grad-CAM variant found or final conv not found.")
                return

            visualization = self._run_gradcam_on_path(path, cam)
            if visualization is None:
                return

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
                self.grad_cam_display_label.setText("Grad-CAM result not in RGB.")
        except Exception as e:
            QMessageBox.critical(self, "Grad-CAM Error", f"Error generating Grad-CAM:\n{e}")

    #####################################################################
    # Batch Inference
    #####################################################################
    def on_browse_batch_folder(self):
        dlg = QFileDialog(self, "Select Folder for Batch Inference")
        dlg.setFileMode(QFileDialog.Directory)
        if dlg.exec_():
            self.batch_folder_line.setText(dlg.selectedFiles()[0])

    def on_add_batch_images(self):
        dlg = QFileDialog(self, "Add Multiple Images")
        dlg.setFileMode(QFileDialog.ExistingFiles)
        dlg.setNameFilter("Images (*.png *.jpg *.jpeg)")
        if dlg.exec_():
            selected = dlg.selectedFiles()
            for s in selected:
                if s not in self.batch_images:
                    self.batch_images.append(s)
            self._refresh_batch_images()

    def _refresh_batch_images(self):
        self.batch_images_text.clear()
        for p in self.batch_images:
            self.batch_images_text.append(p)

    def on_run_batch_inference(self):
        if not self.current_inference_model:
            QMessageBox.warning(self, "No HPC Model", "Pick HPC combo first.")
            return

        folder_path = self.batch_folder_line.text().strip()
        if (not folder_path) and (not self.batch_images):
            QMessageBox.warning(self, "No Input", "Provide a folder or add images first.")
            return

        self.results_table.setRowCount(0)
        self.batch_progress.setValue(0)
        export_csv = self.csv_line.text().strip() or None

        self.batch_worker = BatchInferenceWorker(
            inference_model=self.current_inference_model,
            image_paths=self.batch_images if (folder_path == "") else None,
            folder_path=folder_path if os.path.isdir(folder_path) else None,
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
        for i, rec in enumerate(results):
            fpath = rec.get("file", "")
            self._fill_table_row(i, fpath, rec)
        self.batch_worker.deleteLater()
        self.batch_worker = None

    #####################################################################
    # Table
    #####################################################################
    def _fill_table_row(self, row_idx, file_path, res_dict):
        """
        The dictionary might be:
          - single HPC => { "class_index":..., "probabilities":[...] }
          - double HPC => { "prediction":"no_tumor"/"tumor_subtype_0", "binary_probs":[...], ... }
          - two-step => same shape as double HPC => "prediction":"no_tumor"/"tumor_subtype_whatever"

        We'll interpret them into the table columns:
          'File','Predicted', ProbNoTumor, ProbPit, ProbMen, ProbGli, 'Error'
        """
        file_item = QTableWidgetItem(file_path)
        self.results_table.setItem(row_idx, 0, file_item)

        err_msg = res_dict.get("error","")
        if err_msg:
            for col in range(self.results_table.columnCount()):
                item = QTableWidgetItem("" if col!=0 else file_path)
                item.setForeground(QtGui.QColor("red"))
                if col==6:
                    item.setText(err_msg)
                self.results_table.setItem(row_idx,col,item)
            return

        pred_label, probs_4 = self._interpret_result(res_dict)
        pred_item = QTableWidgetItem(pred_label)
        if pred_label.lower()=="notumor":
            pred_item.setForeground(QtGui.QColor("green"))
        else:
            pred_item.setForeground(QtGui.QColor("darkorange"))
        self.results_table.setItem(row_idx, 1, pred_item)

        for i in range(4):
            val_s = "N/A"
            if probs_4[i] is not None:
                val_s = f"{probs_4[i]*100:.1f}%"
            prob_item = QTableWidgetItem(val_s)
            self.results_table.setItem(row_idx, 2+i, prob_item)

        self.results_table.setItem(row_idx, 6, QTableWidgetItem(""))

    def _interpret_result(self, res_dict):
        """
        Return (pred_label, [pNoTum, pPit, pMen, pGli]).

        'double' or 'double_2step' => "prediction": "no_tumor" or "tumor_subtype_..."
           or if "tumor_subtype_pituitary" => we parse "pituitary" => index 1
           or "tumor_subtype_meningioma" => index 2, etc.

        single-head => 'class_index', 'probabilities'.
          If 4-class => 0=NoTumor,1=Pit,2=Men,3=Gli
          If 2-class => 0=NoTumor,1=Tumor
          If 3-class => 0=Pit,1=Men,2=Gli
        """
        out_probs = [None,None,None,None]  # order => [NoTum, Pit, Men, Gli]
        pred_lbl = "Unknown"

        if "error" in res_dict:
            return pred_lbl, out_probs

        # "double" or two-step => "prediction"
        if "prediction" in res_dict:
            # e.g. "no_tumor" or "tumor_subtype_pituitary"
            p = res_dict["prediction"].lower()
            binp = res_dict.get("binary_probs",[1.0,0.0])  # just to fill
            if p=="no_tumor":
                pred_lbl = "notumor"
                out_probs[0] = float(binp[0])
            else:
                # tumor_subtype_??? => pituitary,meningioma,glioma
                if "pituitary" in p:
                    pred_lbl = "pituitary"
                    out_probs[1] = 1.0
                elif "meningioma" in p:
                    pred_lbl = "meningioma"
                    out_probs[2] = 1.0
                else:
                    pred_lbl = "glioma"
                    out_probs[3] = 1.0

        elif "class_index" in res_dict and "probabilities" in res_dict:
            cid = int(res_dict["class_index"])
            probs = res_dict["probabilities"]
            # handle 2-class, 3-class, or 4-class
            if len(probs)==4:
                # multi => 0=NoTum,1=Pit,2=Men,3=Gli
                pred_lbl = CLASS_LABELS_4[cid] if 0<=cid<4 else "unknown"
                for i in range(4):
                    out_probs[i] = float(probs[i])
            elif len(probs)==2:
                # binary => 0=NoTum,1=Tumor
                if cid==0:
                    pred_lbl="notumor"
                    out_probs[0] = float(probs[0])
                else:
                    pred_lbl="tumor"
                    out_probs[1] = float(probs[1])  # just place tumor prob in pit slot
            elif len(probs)==3:
                # threeclass => 0=Pit,1=Men,2=Gli
                if cid==0:
                    pred_lbl="pituitary"
                elif cid==1:
                    pred_lbl="meningioma"
                else:
                    pred_lbl="glioma"
                out_probs[1] = float(probs[0])  # pit
                out_probs[2] = float(probs[1])  # men
                out_probs[3] = float(probs[2])  # gli

        return pred_lbl, out_probs

    #####################################################################
    # Batch Grad-CAM
    #####################################################################
    def on_batch_gradcam(self):
        if GradCAM is None or show_cam_on_image is None:
            QMessageBox.warning(self, "Missing grad-cam", "pytorch-grad-cam is not installed.")
            return
        if not self.current_inference_model:
            QMessageBox.warning(self, "No Model", "No HPC model loaded.")
            return

        # If the user picks the “double_2step” item, we can’t do standard Grad-CAM
        row_idx = self.model_combo.currentIndex()
        if row_idx>=0 and row_idx<len(self.hpc_combos):
            if self.hpc_combos[row_idx]["model_variant"]=="double_2step":
                QMessageBox.information(self, "Two-step", "Grad-CAM not supported for two-step pipeline.")
                return

        method_name = self.combo_gradcam_method.currentText()
        if method_name.startswith("("):
            QMessageBox.warning(self, "No Grad-CAM", "No Grad-CAM variant installed.")
            return

        selected_rows = self.results_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.information(self, "No Selection", "Please select one or more rows first.")
            return

        alpha_fraction = float(self.slider_alpha.value())/100.0
        cam = self._build_gradcam_variant(method_name, fallback_ok=False)
        if cam is None:
            QMessageBox.warning(self, "Unavailable", f"Grad-CAM variant {method_name} is not available or final conv not found.")
            return

        diag = GradCamDialog(self)
        row_count=0
        col_count=0

        for sel_idx in selected_rows:
            r = sel_idx.row()
            file_item = self.results_table.item(r, 0)
            if not file_item:
                continue
            path = file_item.text()
            if not os.path.isfile(path):
                continue

            visualization = self._run_gradcam_on_path(path, cam, alpha=alpha_fraction)
            if visualization is None:
                continue

            vis_uint8 = visualization.astype(np.uint8)
            if vis_uint8.shape[-1]!=3:
                continue
            from PyQt5.QtGui import QImage, QPixmap
            h2,w2,ch = vis_uint8.shape
            q_img = QImage(vis_uint8.data, w2, h2, ch*w2, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img).scaled(
                256,256,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            diag.add_gradcam_image(pixmap, title=os.path.basename(path), row=row_count, col=col_count)

            col_count+=1
            if col_count>=3:
                col_count=0
                row_count+=1

        diag.exec_()

    def _build_gradcam_variant(self, method_name="GradCAM", fallback_ok=True):
        """
        Build a Grad-CAM object for self.current_inference_model.model
        if that model is a single CNN. If it’s not a real HPC single net, return None.
        """
        # If it's an instance of TwoStepModelWrapper, we have no single net
        if not hasattr(self.current_inference_model,"model"):
            # Possibly a "double_2step" or something => skip
            return None

        model_instance = self.current_inference_model.model
        model_instance.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_instance.to(device)

        final_conv = None
        for name,module in model_instance.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                final_conv = module
        if not final_conv:
            return None

        # pick which method:
        if method_name=="GradCAM" and GradCAM is not None:
            return GradCAM(model=model_instance, target_layers=[final_conv])
        elif method_name=="GradCAM++" and GradCAMPP is not None:
            return GradCAMPP(model=model_instance, target_layers=[final_conv])
        elif method_name=="ScoreCAM" and ScoreCAM is not None:
            return ScoreCAM(model=model_instance, target_layers=[final_conv])
        elif method_name=="XGradCAM" and XGradCAM is not None:
            return XGradCAM(model=model_instance, target_layers=[final_conv])
        elif method_name=="EigenCAM" and EigenCAM is not None:
            return EigenCAM(model=model_instance, target_layers=[final_conv])

        if fallback_ok and GradCAM is not None:
            return GradCAM(model=model_instance, target_layers=[final_conv])
        return None

    def _run_gradcam_on_path(self, img_path, cam, alpha=0.5):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            pil_img = Image.open(img_path).convert('L')
        except Exception as e:
            QMessageBox.critical(self, "Open Error", f"Could not open {img_path}: {e}")
            return None

        transform = self.current_inference_model.transform  # type: ignore
        tensor = transform(pil_img).unsqueeze(0).to(device)
        grayscale_cam = cam(input_tensor=tensor, targets=None)
        grayscale_cam = grayscale_cam[0,:]

        import cv2
        arr = np.array(pil_img, dtype=np.float32)/255.0
        rgb_img_np = np.stack([arr]*3, axis=-1)
        h,w,_ = rgb_img_np.shape
        cam_resized = cv2.resize(grayscale_cam, (w,h), interpolation=cv2.INTER_LINEAR)
        if show_cam_on_image is None:
            return None
        overlay = show_cam_on_image(rgb_img_np, cam_resized, use_rgb=True, image_weight=(1.0-alpha))
        return overlay
