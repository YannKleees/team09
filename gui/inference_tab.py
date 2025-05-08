# inference_tab.py

import os
import json
import csv
import warnings
import torch
import torch.nn.functional as F
from PIL import Image

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QComboBox, QLineEdit, QPushButton, QHBoxLayout,
    QFileDialog, QTextEdit, QMessageBox, QGroupBox, QFormLayout, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea, QRadioButton,
    QButtonGroup, QCheckBox, QTableWidgetItem
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5 import QtGui

from gui.widgets import MplCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from logic.inference import InferenceModel

# HPC combos JSON file
_HPC_RESULTS_JSON = "hpc_results.json"

# For "multi" 4 classes: 0=Glioma,1=Meningioma,2=NoTumor,3=Pituitary
CLASS_LABELS_4 = ["Glioma", "Meningioma", "NoTumor", "Pituitary"]


def load_hpc_results_json(path=_HPC_RESULTS_JSON):
    """Load HPC combos from JSON (saved HPC search)."""
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
    Runs inference in background for a batch of images 
    so the UI stays responsive.
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


class InferenceTab(QWidget):
    """
    Upgraded InferenceTab that:
     - Auto-loads HPC combos from JSON
     - Single vs. Batch in separate group boxes (hidden when not in use)
     - Collapsible HPC Info (ConfMat, ROC, metrics)
     - Mild QSS styling & color-coded table rows
     - QThread cleanup
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Basic style to make group boxes, table, etc. look nicer
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
        """)

        main_layout = QVBoxLayout(self)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(10,10,10,10)
        container_layout.setSpacing(8)

        # HPC model selection
        hpc_box = QGroupBox("Select HPC Model")
        vb_hpc = QVBoxLayout(hpc_box)
        self.hpc_combos = load_hpc_results_json()
        self.model_combo = QComboBox()
        vb_hpc.addWidget(QLabel("Available HPC combos:"))
        vb_hpc.addWidget(self.model_combo)
        container_layout.addWidget(hpc_box)

        # Collapsible HPC info
        self.chk_show_info = QCheckBox("Expand HPC Info (ConfMat, ROC, Metrics)")
        self.chk_show_info.setChecked(False)
        self.chk_show_info.stateChanged.connect(self._toggle_hpc_info)
        container_layout.addWidget(self.chk_show_info)

        self.info_box = QGroupBox("HPC Model Info")
        vb_info = QVBoxLayout(self.info_box)

        row_plots = QHBoxLayout()
        # ConfMat
        self.confmat_canvas = MplCanvas(self, width=4, height=3)
        self.confmat_toolbar = NavigationToolbar(self.confmat_canvas, self)
        left_g = QVBoxLayout()
        left_g.addWidget(QLabel("Confusion Matrix"))
        left_g.addWidget(self.confmat_canvas)
        left_g.addWidget(self.confmat_toolbar)
        row_plots.addLayout(left_g)

        # ROC
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

        # Refresh HPC info button
        self.btn_show_info = QPushButton("Refresh HPC Info")
        self.btn_show_info.clicked.connect(self.show_hpc_info)
        vb_info.addWidget(self.btn_show_info)

        container_layout.addWidget(self.info_box)
        self.info_box.hide()  # default collapsed

        # Single vs batch radio
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
        self.single_box = QGroupBox("Single Image Inference")
        form_s = QFormLayout(self.single_box)
        self.single_img_line = QLineEdit()
        btn_browse_single = QPushButton("Browse Image...")
        btn_browse_single.clicked.connect(self.on_browse_single)
        h_s = QHBoxLayout()
        h_s.addWidget(self.single_img_line)
        h_s.addWidget(btn_browse_single)
        form_s.addRow("Image:", h_s)

        self.btn_run_single = QPushButton("Run Single Inference")
        self.btn_run_single.clicked.connect(self.on_run_single_inference)
        form_s.addRow(self.btn_run_single)

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

        # Table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(7)
        self.results_table.setHorizontalHeaderLabels([
            "File", "Predicted",
            "Prob(Glioma)", "Prob(Mening)", "Prob(NoTumor)", "Prob(Pituitary)",
            "Error"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        container_layout.addWidget(self.results_table)

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
        self._on_mode_changed()  # set initial hide/show

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
        if not self.hpc_combos or idx<0 or idx>=len(self.hpc_combos):
            self.current_inference_model = None
            return

        chosen = self.hpc_combos[idx]
        mpath = chosen.get("model_path","")
        variant = chosen.get("model_variant","multi")
        if not mpath or not os.path.isfile(mpath):
            self.current_inference_model = None
            return

        # parse resize from combo_str
        resize_val = 128
        for part in chosen["combo_str"].split(","):
            part = part.strip().lower()
            if "resize=" in part:
                try:
                    eqpos = part.index("=")
                    val_s = part[eqpos+1:].strip()
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
        """Refresh HPC info if expanded."""
        if not self.info_box.isVisible():
            return

        self.confmat_canvas.figure.clear()
        self.roc_canvas.figure.clear()
        self.metrics_text.clear()

        idx = self.model_combo.currentIndex()
        if not self.hpc_combos or idx<0 or idx>=len(self.hpc_combos):
            self.metrics_text.setPlainText("No HPC combos or invalid selection.")
            self.confmat_canvas.draw()
            self.roc_canvas.draw()
            return

        combo = self.hpc_combos[idx]
        confmat = combo.get("confmat", None)
        roc_ = combo.get("roc", None)
        mets = combo.get("metrics", None)

        # Confmat
        axc = self.confmat_canvas.figure.add_subplot(111)
        if confmat:
            import numpy as np
            arr = np.array(confmat, dtype=int)
            im = axc.imshow(arr, cmap="Blues")
            axc.set_title("Confusion Matrix", fontsize=10)
            for r in range(arr.shape[0]):
                for c in range(arr.shape[1]):
                    axc.text(c, r, str(arr[r,c]), ha='center', va='center')
            self.confmat_canvas.figure.colorbar(im, ax=axc)
        else:
            axc.text(0.5,0.5,"No ConfMat found", ha='center', va='center')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.confmat_canvas.figure.tight_layout(pad=0.8)
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
            axr.text(0.5,0.5,"No ROC (non-binary/double?)", ha='center', va='center')

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
                lines.append(f"Class {i}: p={prec[i]:.3f}, r={rec_[i]:.3f}, f1={f1_[i]:.3f}, sup={sup_[i]}")
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
    # Batch
    # ----------------------------------------------------------------
    def on_browse_batch_folder(self):
        dlg = QFileDialog(self, "Select Folder for Batch")
        dlg.setFileMode(QFileDialog.Directory)
        if dlg.exec_():
            self.batch_folder_line.setText(dlg.selectedFiles()[0])

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
            image_paths=self.batch_images if folder=="" else None,
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
    # Fill Table
    # ----------------------------------------------------------------
    def _fill_table_row(self, row_idx, file_path, res_dict):
        """
        Columns:
          0: File
          1: Predicted
          2: Prob(Glioma)
          3: Prob(Mening)
          4: Prob(NoTumor)
          5: Prob(Pituitary)
          6: Error
        We'll also color-code the row:
          - Error => entire row red
          - "NoTumor" => green
          - "Tumor/Glioma/Meningioma/Pituitary" => orange
        """
        # File
        file_item = QTableWidgetItem(file_path)
        self.results_table.setItem(row_idx, 0, file_item)

        # Check error
        err_msg = res_dict.get("error", "")
        if err_msg:
            # entire row red
            for col in range(self.results_table.columnCount()):
                item = QTableWidgetItem("" if col!=0 else file_path)
                item.setForeground(QtGui.QColor("red"))
                if col == 6:  # error col
                    item.setText(err_msg)
                self.results_table.setItem(row_idx, col, item)
            return

        # otherwise interpret result
        pred_label, probs_4 = self._interpret_result(res_dict)

        # Pred item
        pred_item = QTableWidgetItem(pred_label)
        # color code
        if pred_label.lower()=="notumor":
            pred_item.setForeground(QtGui.QColor("green"))
        else:
            # might be "tumor", "glioma", "meningioma", "pituitary"
            pred_item.setForeground(QtGui.QColor("darkorange"))

        self.results_table.setItem(row_idx, 1, pred_item)

        # Prob columns
        for i in range(4):
            val_s = "N/A"
            if probs_4[i] is not None:
                val_s = f"{probs_4[i]*100:.1f}%"
            prob_item = QTableWidgetItem(val_s)
            self.results_table.setItem(row_idx, 2+i, prob_item)

        # error col is empty
        self.results_table.setItem(row_idx, 6, QTableWidgetItem(""))

    def _interpret_result(self, res_dict):
        """
        Return (pred_label, [pGlioma, pMening, pNoTumor, pPitu]).
        - 'double' => "prediction": "no_tumor" or "tumor_subtype_#"
        - 'binary' => "class_index": 0..1, "probabilities": length=2
        - 'multi' => "class_index": 0..3, "probabilities": length=4
        """
        out_probs = [None,None,None,None]
        pred_lbl = "Unknown"

        if "error" in res_dict:
            return pred_lbl, out_probs

        # double
        if "prediction" in res_dict:
            # e.g. "no_tumor" => 2=NoTumor
            # or "tumor_subtype_0" => 0=Glioma
            pred = res_dict["prediction"].lower()
            if pred == "no_tumor":
                pred_lbl = "NoTumor"
                bin_p = res_dict.get("binary_probs",[1.0,0.0])
                out_probs[2] = float(bin_p[0])  # noTumor prob
                out_probs[0] = 0.0
                out_probs[1] = 0.0
                out_probs[3] = 0.0
            else:
                # "tumor_subtype_X"
                idx_s = pred.rsplit("_",1)[-1]
                try:
                    sub_i = int(idx_s)
                except:
                    sub_i = 0
                # sub_i => 0=Glioma,1=Mening,2=Pituitary
                if sub_i==0:
                    pred_lbl = "Glioma"
                elif sub_i==1:
                    pred_lbl = "Meningioma"
                else:
                    pred_lbl = "Pituitary"
                mp = res_dict.get("multi_probs",[0.0,0.0,0.0])
                out_probs[0] = float(mp[0]) if len(mp)>0 else 0.0
                out_probs[1] = float(mp[1]) if len(mp)>1 else 0.0
                out_probs[2] = 0.0
                out_probs[3] = float(mp[2]) if len(mp)>2 else 0.0

        # single-head => "class_index", "probabilities"
        elif "class_index" in res_dict and "probabilities" in res_dict:
            cid = int(res_dict["class_index"])
            probs = res_dict["probabilities"]
            # multi => 4 classes
            if len(probs)==4:
                if 0<=cid<4:
                    pred_lbl = CLASS_LABELS_4[cid]
                for i in range(4):
                    out_probs[i] = float(probs[i])
            elif len(probs)==2:
                # binary => class0=NoTumor, class1=Tumor => map tumor => "Glioma"
                if cid==0:
                    pred_lbl = "NoTumor"
                else:
                    pred_lbl = "Tumor"
                out_probs[0] = float(probs[1]) if cid==1 else 0.0  # "tumor" => glioma col
                out_probs[1] = 0.0
                out_probs[2] = float(probs[0])  # noTumor
                out_probs[3] = 0.0

        return pred_lbl, out_probs
