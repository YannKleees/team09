import os
import random
import math
from PIL import Image
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QFormLayout, QHBoxLayout,
    QPushButton, QCheckBox, QSlider, QFileDialog, QMessageBox, QSpinBox,
    QTextEdit, QProgressBar, QTabWidget, QGridLayout, QSpacerItem, QSizePolicy,
    QScrollArea, QLineEdit, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from gui.widgets import MplCanvas
from logic.augmentation import preview_augmentation, bulk_augment_dataset
import numpy as np


def count_distribution(root_dir):
    """
    Return a dict:
      {
        "Training/ClassA": countA,
        "Training/ClassB": countB,
        "Testing/ClassA": ...
      }
    """
    dist = {}
    if not os.path.isdir(root_dir):
        return dist
    for subset in os.listdir(root_dir):
        subset_path = os.path.join(root_dir, subset)
        if os.path.isdir(subset_path):
            classes = [
                d for d in os.listdir(subset_path)
                if os.path.isdir(os.path.join(subset_path, d))
            ]
            for cls_ in classes:
                cpath = os.path.join(subset_path, cls_)
                images = [
                    f for f in os.listdir(cpath)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
                dist[f"{subset}/{cls_}"] = len(images)
    return dist


class AugmentationWorker(QThread):
    """
    Bulk augmentation in background, using a single `final_target` for each class.
    """
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(int, list)  # (total_new_images, error_log)

    def __init__(
        self,
        src_root,            # <--- add source root
        dst_root,            # <--- add destination root
        final_target=2000,
        rotate_deg=15,
        do_rotate=True,
        do_bc=True,
        do_gamma=True,
        do_flip=False,
        do_perspective=False,
        do_crop=False,
        b_min=0.85,
        b_max=1.15,
        c_min=0.85,
        c_max=1.15,
        g_min=0.85,
        g_max=1.15,
        parent=None
    ):
        super().__init__(parent)
        self.src_root = src_root
        self.dst_root = dst_root
        self.final_target = final_target
        self.rotate_deg = rotate_deg
        self.do_rotate = do_rotate
        self.do_bc = do_bc
        self.do_gamma = do_gamma
        self.do_flip = do_flip
        self.do_perspective = do_perspective
        self.do_crop = do_crop
        self.b_min = b_min
        self.b_max = b_max
        self.c_min = c_min
        self.c_max = c_max
        self.g_min = g_min
        self.g_max = g_max

    def run(self):
        def callback(pval, msg):
            self.progress.emit(pval, msg)

        try:
            total_new_images, error_log = bulk_augment_dataset(
                src_root=self.src_root,
                dst_root=self.dst_root,
                final_target=self.final_target,
                rotate_deg=self.rotate_deg,
                do_rotate=self.do_rotate,
                do_brightness_contrast=self.do_bc,
                do_gamma=self.do_gamma,
                do_flip=self.do_flip,
                do_perspective=self.do_perspective,
                do_crop=self.do_crop,
                brightness_range=(self.b_min, self.b_max),
                contrast_range=(self.c_min, self.c_max),
                gamma_range=(self.g_min, self.g_max),
                progress_callback=callback
            )
            self.finished.emit(total_new_images, error_log)
        except Exception as e:
            msg = f"Augmentation error: {e}"
            self.progress.emit(0, msg)
            self.finished.emit(0, [msg])


class QDoubleSpinBoxRange(QDoubleSpinBox):
    """
    Helper widget for double spin boxes to quickly set:
      - range
      - default value
      - single step
      - decimals
      - a tooltip
    """
    def __init__(self, minv, maxv, default, label_name="", parent=None):
        super().__init__(parent)
        self.setRange(minv, maxv)
        self.setSingleStep(0.05)
        self.setValue(default)
        self.setDecimals(2)
        self.setToolTip(label_name)


class AugmentationTab(QWidget):
    """
    Two subtabs:
      - Tab 1: Preview (multi-preview)
      - Tab 2: Bulk Augmentation => each class up to 'final_target'
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        main_layout = QVBoxLayout(self)

        intro_label = QLabel("<h3>Data Augmentation (Fixed Target per Class)</h3>")
        intro_label.setWordWrap(True)
        main_layout.addWidget(intro_label)

        self.sub_tabs = QTabWidget()
        self.sub_tabs.setTabPosition(QTabWidget.North)

        preview_tab = self._create_preview_tab()
        bulk_tab = self._create_bulk_tab()

        self.sub_tabs.addTab(preview_tab, "Preview")
        self.sub_tabs.addTab(bulk_tab, "Bulk Augmentation")

        main_layout.addWidget(self.sub_tabs)
        self.setLayout(main_layout)

        self.worker = None
        self.original_img = None

    # ----------------------------------------------------------------
    # PREVIEW TAB
    # ----------------------------------------------------------------
    def _create_preview_tab(self):
        preview_widget = QWidget()
        layout = QVBoxLayout(preview_widget)

        preview_box = QGroupBox("Preview Configuration")
        form = QFormLayout()

        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.on_load_image)

        self.rotate_chk = QCheckBox("Rotate?")
        self.rotate_chk.setChecked(True)
        self.rotate_slider = QSlider(Qt.Horizontal)
        self.rotate_slider.setRange(0, 60)
        self.rotate_slider.setValue(15)

        self.bc_chk = QCheckBox("Brightness/Contrast?")
        self.bc_chk.setChecked(True)
        self.b_min_spin = QDoubleSpinBoxRange(0.1, 2.0, 0.85, "Brightness Min:")
        self.b_max_spin = QDoubleSpinBoxRange(0.1, 2.0, 1.15, "Brightness Max:")
        self.c_min_spin = QDoubleSpinBoxRange(0.1, 2.0, 0.85, "Contrast Min:")
        self.c_max_spin = QDoubleSpinBoxRange(0.1, 2.0, 1.15, "Contrast Max:")

        self.gamma_chk = QCheckBox("Gamma?")
        self.gamma_chk.setChecked(True)
        self.g_min_spin = QDoubleSpinBoxRange(0.5, 2.0, 0.85, "Gamma Min:")
        self.g_max_spin = QDoubleSpinBoxRange(0.5, 2.0, 1.15, "Gamma Max:")

        self.flip_chk = QCheckBox("Flip?")
        self.perspective_chk = QCheckBox("Perspective Warp?")
        self.crop_chk = QCheckBox("Random Crop?")

        self.minmax_chk = QCheckBox("MinMax Preview?")

        self.multi_preview_spin = QSpinBox()
        self.multi_preview_spin.setRange(1, 10)
        self.multi_preview_spin.setValue(1)

        self.prev_btn = QPushButton("Generate Preview(s)")
        self.prev_btn.clicked.connect(self.on_preview)

        form.addRow(self.load_btn)

        rot_layout = QHBoxLayout()
        rot_layout.addWidget(self.rotate_chk)
        rot_layout.addWidget(self.rotate_slider)
        form.addRow("Rotation:", rot_layout)

        form.addRow(self.bc_chk)
        bc_layout = QHBoxLayout()
        bc_layout.addWidget(self.b_min_spin)
        bc_layout.addWidget(self.b_max_spin)
        form.addRow("Brightness Range:", bc_layout)

        ct_layout = QHBoxLayout()
        ct_layout.addWidget(self.c_min_spin)
        ct_layout.addWidget(self.c_max_spin)
        form.addRow("Contrast Range:", ct_layout)

        form.addRow(self.gamma_chk)
        g_layout = QHBoxLayout()
        g_layout.addWidget(self.g_min_spin)
        g_layout.addWidget(self.g_max_spin)
        form.addRow("Gamma Range:", g_layout)

        adv_layout = QHBoxLayout()
        adv_layout.addWidget(self.flip_chk)
        adv_layout.addWidget(self.perspective_chk)
        adv_layout.addWidget(self.crop_chk)
        form.addRow("Advanced:", adv_layout)

        row2 = QHBoxLayout()
        row2.addWidget(self.minmax_chk)
        row2.addWidget(QLabel("Multi Previews:"))
        row2.addWidget(self.multi_preview_spin)
        form.addRow(row2)

        form.addRow(self.prev_btn)
        preview_box.setLayout(form)
        layout.addWidget(preview_box)

        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        layout.addWidget(self.preview_scroll)

        return preview_widget

    def on_load_image(self):
        dlg = QFileDialog(self)
        dlg.setNameFilter("Images (*.png *.jpg *.jpeg)")
        if dlg.exec_():
            selected = dlg.selectedFiles()[0]
            try:
                self.original_img = Image.open(selected).convert('L')
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not load image:\n{e}")

    def on_preview(self):
        if self.original_img is None:
            QMessageBox.warning(self, "No Image", "Load an image first.")
            return

        n_previews = self.multi_preview_spin.value()

        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(8)
        row = 0

        # Original
        orig_lbl = QLabel("Original")
        grid_layout.addWidget(orig_lbl, row, 0, 1, 1, alignment=Qt.AlignHCenter)
        orig_canvas = MplCanvas(grid_widget, width=3, height=3)
        self._show_pil_on_canvas(orig_canvas, self.original_img)
        grid_layout.addWidget(orig_canvas, row+1, 0)

        # Previews
        for i in range(n_previews):
            col = i + 1
            lbl = QLabel(f"Preview {i+1}")
            grid_layout.addWidget(lbl, row, col, alignment=Qt.AlignHCenter)
            canvas = MplCanvas(grid_widget, width=3, height=3)
            aug_tens = preview_augmentation(
                pil_image=self.original_img,
                rotate_deg=self.rotate_slider.value(),
                do_rotate=self.rotate_chk.isChecked(),
                do_brightness_contrast=self.bc_chk.isChecked(),
                do_gamma=self.gamma_chk.isChecked(),
                do_minmax=self.minmax_chk.isChecked(),
                do_flip=self.flip_chk.isChecked(),
                do_perspective=self.perspective_chk.isChecked(),
                do_crop=self.crop_chk.isChecked(),
                brightness_range=(self.b_min_spin.value(), self.b_max_spin.value()),
                contrast_range=(self.c_min_spin.value(), self.c_max_spin.value()),
                gamma_range=(self.g_min_spin.value(), self.g_max_spin.value())
            )
            arr = aug_tens.squeeze().numpy()
            self._show_arr_on_canvas(canvas, arr)
            grid_layout.addWidget(canvas, row+1, col)

        grid_layout.addItem(
            QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Expanding),
            row+2, 0
        )
        self.preview_scroll.setWidget(grid_widget)

    def _show_pil_on_canvas(self, canvas, pil_img):
        canvas.figure.clear()
        ax = canvas.figure.subplots()
        ax.imshow(pil_img, cmap='gray')
        ax.axis('off')
        canvas.draw()

    def _show_arr_on_canvas(self, canvas, arr):
        canvas.figure.clear()
        ax = canvas.figure.subplots()
        ax.imshow(arr, cmap='gray')
        ax.axis('off')
        canvas.draw()

    # ----------------------------------------------------------------
    # BULK AUGMENTATION TAB
    # ----------------------------------------------------------------
    def _create_bulk_tab(self):
        """
        Wraps the bulk augmentation area in a QScrollArea so it can expand or scroll.
        """
        # Core widget for the bulk area
        bulk_widget = QWidget()
        layout = QVBoxLayout(bulk_widget)

        # --- Bulk config box
        bulk_box = QGroupBox("Bulk Augmentation Configuration")
        bform = QFormLayout()

        self.src_folder_edit = QLineEdit("cleaned_data")
        self.btn_src_browse = QPushButton("Browse...")
        self.btn_src_browse.clicked.connect(self.on_browse_src)
        h_src = QHBoxLayout()
        h_src.addWidget(self.src_folder_edit)
        h_src.addWidget(self.btn_src_browse)
        bform.addRow("Source Folder:", h_src)

        self.dst_folder_edit = QLineEdit("augmented_data")
        self.btn_dst_browse = QPushButton("Browse...")
        self.btn_dst_browse.clicked.connect(self.on_browse_dst)
        h_dst = QHBoxLayout()
        h_dst.addWidget(self.dst_folder_edit)
        h_dst.addWidget(self.btn_dst_browse)
        bform.addRow("Destination Folder:", h_dst)

        self.target_spin = QSpinBox()
        self.target_spin.setRange(1, 99999)
        self.target_spin.setValue(3000)

        self.bulk_rotate_chk = QCheckBox("Rotate?")
        self.bulk_rotate_chk.setChecked(True)
        self.bulk_rotate_slider = QSlider(Qt.Horizontal)
        self.bulk_rotate_slider.setRange(0, 60)
        self.bulk_rotate_slider.setValue(15)

        self.bulk_bc_chk = QCheckBox("Brightness/Contrast?")
        self.bulk_bc_chk.setChecked(True)
        self.bulk_b_min = QDoubleSpinBoxRange(0.1, 2.0, 0.85, "Bmin")
        self.bulk_b_max = QDoubleSpinBoxRange(0.1, 2.0, 1.15, "Bmax")
        self.bulk_c_min = QDoubleSpinBoxRange(0.1, 2.0, 0.85, "Cmin")
        self.bulk_c_max = QDoubleSpinBoxRange(0.1, 2.0, 1.15, "Cmax")

        self.bulk_gamma_chk = QCheckBox("Gamma?")
        self.bulk_gamma_chk.setChecked(True)
        self.bulk_g_min = QDoubleSpinBoxRange(0.1, 2.0, 0.85, "Gmin")
        self.bulk_g_max = QDoubleSpinBoxRange(0.1, 2.0, 1.15, "Gmax")

        self.bulk_flip_chk = QCheckBox("Flip?")
        self.bulk_perspective_chk = QCheckBox("Perspective?")
        self.bulk_crop_chk = QCheckBox("Crop?")

        self.bulk_btn = QPushButton("Start Bulk Augmentation")
        self.bulk_btn.clicked.connect(self.on_bulk_augment)

        self.aug_progress = QProgressBar()
        self.aug_progress.setValue(0)
        self.aug_log = QTextEdit()
        self.aug_log.setReadOnly(True)

        bform.addRow("Final Target (images per class):", self.target_spin)

        rot_layout = QHBoxLayout()
        rot_layout.addWidget(self.bulk_rotate_chk)
        rot_layout.addWidget(self.bulk_rotate_slider)
        bform.addRow("Rotation:", rot_layout)

        bc_line = QHBoxLayout()
        bc_line.addWidget(self.bulk_bc_chk)
        bform.addRow("Bright/Contrast?", bc_line)

        bc2_line = QHBoxLayout()
        bc2_line.addWidget(self.bulk_b_min)
        bc2_line.addWidget(self.bulk_b_max)
        bform.addRow("Brightness Range:", bc2_line)

        bc3_line = QHBoxLayout()
        bc3_line.addWidget(self.bulk_c_min)
        bc3_line.addWidget(self.bulk_c_max)
        bform.addRow("Contrast Range:", bc3_line)

        g_line = QHBoxLayout()
        g_line.addWidget(self.bulk_gamma_chk)
        bform.addRow("Gamma?", g_line)

        g2_line = QHBoxLayout()
        g2_line.addWidget(self.bulk_g_min)
        g2_line.addWidget(self.bulk_g_max)
        bform.addRow("Gamma Range:", g2_line)

        adv_line = QHBoxLayout()
        adv_line.addWidget(self.bulk_flip_chk)
        adv_line.addWidget(self.bulk_perspective_chk)
        adv_line.addWidget(self.bulk_crop_chk)
        bform.addRow("Advanced:", adv_line)

        bform.addRow(self.bulk_btn)
        bform.addRow(self.aug_progress)
        bform.addRow(self.aug_log)

        bulk_box.setLayout(bform)
        layout.addWidget(bulk_box)

        # --- Distribution box
        dist_box = QGroupBox("Distribution Dashboard")
        dist_layout = QVBoxLayout(dist_box)
        info_label = QLabel(
            "Shows how class counts changed from the chosen source folder "
            "to the chosen destination folder."
        )
        info_label.setWordWrap(True)
        dist_layout.addWidget(info_label)

        self.dist_btn = QPushButton("Show Distribution Comparison")
        self.dist_btn.clicked.connect(self.on_show_distribution)
        dist_layout.addWidget(self.dist_btn)

        self.dist_canvas = MplCanvas(self, width=6, height=4)
        dist_toolbar = NavigationToolbar(self.dist_canvas, self)
        dist_layout.addWidget(self.dist_canvas)
        dist_layout.addWidget(dist_toolbar)

        layout.addWidget(dist_box)

        # Wrap everything in a scroll area so it can expand or scroll
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(500)  # adjust to taste
        scroll_area.setWidget(bulk_widget)

        return scroll_area

    def on_browse_src(self):
        dlg = QFileDialog(self)
        dlg.setFileMode(QFileDialog.Directory)
        if dlg.exec_():
            self.src_folder_edit.setText(dlg.selectedFiles()[0])

    def on_browse_dst(self):
        dlg = QFileDialog(self)
        dlg.setFileMode(QFileDialog.Directory)
        if dlg.exec_():
            self.dst_folder_edit.setText(dlg.selectedFiles()[0])

    def on_bulk_augment(self):
        self.aug_log.clear()
        self.aug_progress.setValue(0)

        src_folder = self.src_folder_edit.text().strip()
        dst_folder = self.dst_folder_edit.text().strip()

        if not os.path.isdir(src_folder):
            QMessageBox.warning(self, "Invalid Source", f"'{src_folder}' is not a valid directory.")
            return

        if not dst_folder:
            QMessageBox.warning(self, "Invalid Destination", "Please specify a valid destination folder.")
            return

        final_target = self.target_spin.value()
        rotate_deg = self.bulk_rotate_slider.value() if self.bulk_rotate_chk.isChecked() else 0
        do_rotate = self.bulk_rotate_chk.isChecked()
        do_bc = self.bulk_bc_chk.isChecked()
        do_gamma = self.bulk_gamma_chk.isChecked()
        do_flip = self.bulk_flip_chk.isChecked()
        do_perspective = self.bulk_perspective_chk.isChecked()
        do_crop = self.bulk_crop_chk.isChecked()

        b_min = self.bulk_b_min.value()
        b_max = self.bulk_b_max.value()
        c_min = self.bulk_c_min.value()
        c_max = self.bulk_c_max.value()
        g_min = self.bulk_g_min.value()
        g_max = self.bulk_g_max.value()

        self.worker = AugmentationWorker(
            src_root=src_folder,
            dst_root=dst_folder,
            final_target=final_target,
            rotate_deg=rotate_deg,
            do_rotate=do_rotate,
            do_bc=do_bc,
            do_gamma=do_gamma,
            do_flip=do_flip,
            do_perspective=do_perspective,
            do_crop=do_crop,
            b_min=b_min,
            b_max=b_max,
            c_min=c_min,
            c_max=c_max,
            g_min=g_min,
            g_max=g_max
        )
        self.worker.progress.connect(self.on_aug_progress)
        self.worker.finished.connect(self.on_aug_finished)
        self.worker.start()

    def on_aug_progress(self, val, msg):
        self.aug_progress.setValue(val)
        self.aug_progress.setFormat(msg)

    def on_aug_finished(self, total_new_images, error_log):
        self.aug_progress.setValue(100)
        self.aug_progress.setFormat("Finished")
        self.aug_log.append(f"Total new images created: {total_new_images}")
        if error_log:
            self.aug_log.append("<b>Errors:</b>")
            for e in error_log:
                self.aug_log.append(e)

    def on_show_distribution(self):
        # Compare distribution from the chosen src folder to the chosen dst folder
        src_folder = self.src_folder_edit.text().strip()
        dst_folder = self.dst_folder_edit.text().strip()

        cleaned_dist = count_distribution(src_folder)
        aug_dist = count_distribution(dst_folder)

        # 1) Collect all possible subset/class combos
        all_keys = sorted(set(cleaned_dist.keys()).union(aug_dist.keys()))

        # 2) **New**: Filter out anything that isn't "Training/...".
        all_keys = [k for k in all_keys if k.startswith("Training/")]

        cleaned_vals = [cleaned_dist.get(k, 0) for k in all_keys]
        aug_vals = [aug_dist.get(k, 0) for k in all_keys]

        self.dist_canvas.figure.clear()
        ax = self.dist_canvas.figure.subplots()

        x_positions = np.arange(len(all_keys))
        width = 0.35

        bars_cleaned = ax.bar(x_positions - width/2, cleaned_vals, width, label="Source")
        bars_aug = ax.bar(x_positions + width/2, aug_vals, width, label="Destination")

        # Annotate bars (optional)
        for bar in bars_cleaned:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2, height + 0.5,
                f"{int(height)}", ha='center', va='bottom',
                color="blue", fontsize=8
            )

        for bar in bars_aug:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2, height + 0.5,
                f"{int(height)}", ha='center', va='bottom',
                color="orange", fontsize=8
            )

        ax.set_ylabel("Image Count")
        ax.set_title("Source vs. Destination Distribution (Training only)")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(all_keys, rotation=45, ha='right')
        ax.legend()

        self.dist_canvas.figure.tight_layout()
        self.dist_canvas.draw()

        # Show logs in the text area if desired:
        self.aug_log.append("<b>Distribution comparison (Training only):</b>")
        for k in all_keys:
            c_val = cleaned_dist.get(k, 0)
            a_val = aug_dist.get(k, 0)
            self.aug_log.append(f"{k}: source={c_val}  destination={a_val}")
"""
auto_search_tab.py

Updated HPC search that removes the old "disable_aug" and "adv_aug" parameters,
removes any references to "resize", and also removes "double" from the HPC model variants
so we don't train it. We only keep 'binary', 'simple', 'threeclass'.
Includes a set of checkboxes for model variants, plus typical HPC space
(dropout, weight_decay, earlystop, class_weight).
Everything else (progress display, final training, HPC info) is the same.
"""

import os
import json
import random
import copy
import torch
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QGroupBox, QFormLayout,
    QLabel, QPushButton, QComboBox, QLineEdit, QSpinBox,
    QProgressBar, QTextEdit, QMessageBox, QHBoxLayout, QTableWidget,
    QTableWidgetItem, QHeaderView, QFileDialog, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from gui.widgets import MplCanvas

from src.experiment import Experiment

_HPC_RESULTS_JSON = "hpc_results.json"


def save_hpc_results_json(all_results, path=_HPC_RESULTS_JSON):
    """
    Saves HPC combos (confmat/ROC/metrics) to a JSON file so they persist across sessions.
    """
    import json
    data_to_save = []
    for rec in all_results:
        # Convert confusion matrix, if any
        confmat_tensor = rec.get("confmat", None)
        if isinstance(confmat_tensor, torch.Tensor):
            cmat_list = confmat_tensor.cpu().tolist()
        else:
            cmat_list = None

        # Convert any ROC arrays to lists
        roc_obj = rec.get("roc", None)
        if roc_obj:
            fpr = roc_obj.get("fpr", [])
            tpr = roc_obj.get("tpr", [])
            if isinstance(fpr, np.ndarray):
                fpr = fpr.tolist()
            if isinstance(tpr, np.ndarray):
                tpr = tpr.tolist()
            roc_obj = {
                "fpr": fpr,
                "tpr": tpr,
                "auc": roc_obj.get("auc", None)
            }
        else:
            roc_obj = None

        data_to_save.append({
            "combo_str":     rec.get("combo_str", ""),
            "val_acc":       rec.get("val_acc", 0.0),
            "model_path":    rec.get("model_path", ""),
            "model_variant": rec.get("model_variant", ""),
            "confmat_list":  cmat_list,
            "roc":           roc_obj,
            "metrics":       rec.get("metrics", None)
        })

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2)
    except Exception as e:
        print(f"[WARN] Could not save HPC results to {path}: {e}")


def load_hpc_results_json(path=_HPC_RESULTS_JSON):
    if not os.path.isfile(path):
        return []
    if os.path.getsize(path) == 0:
        print(f"[WARN] {path} is empty. Return no HPC combos.")
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                print(f"[WARN] {path} is empty (or whitespace). Return no HPC combos.")
                return []
            data = json.loads(content)
    except Exception as e:
        print(f"[WARN] Could not load HPC combos from {path}: {e}")
        return []

    results = []
    for item in data:
        cmat_list = item.get("confmat_list", None)
        if cmat_list is not None:
            cmat_tensor = torch.tensor(cmat_list, dtype=torch.int32)
        else:
            cmat_tensor = None

        results.append({
            "combo_str":     item.get("combo_str", ""),
            "val_acc":       item.get("val_acc", 0.0),
            "model_path":    item.get("model_path", ""),
            "model_variant": item.get("model_variant", ""),
            "confmat":       cmat_tensor,
            "roc":           item.get("roc", None),
            "metrics":       item.get("metrics", None)
        })
    return results


class HPCSearchWorker(QThread):
    """
    Conduct HPC with partial training for each combo (no "double" variant).
    We store best val_acc, confmat, ROC, metrics, plus the saved model path.
    """
    progress = pyqtSignal(int)
    log_msg = pyqtSignal(str)
    combo_epoch_update = pyqtSignal(int, int, float, float)  # combo_idx, epoch, train_acc, val_acc
    finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_root = "cleaned_data"
        self.model_variants = []
        self.dropout_list = []
        self.weight_decay_list = []
        self.earlystop_list = []
        self.class_weight_list = []
        self.batch_size = 16
        self.lr = 0.001
        self.mini_epochs = 2
        self.device = "cpu"

        self.best_acc = -1.0
        self.best_params_str = ""
        self.all_results = []

    def setup_hpc(
        self,
        data_root,
        model_variants,
        dropout_list,
        weight_decay_list,
        earlystop_list,
        class_weight_list,
        batch_size,
        lr,
        mini_epochs,
        device="cpu"
    ):
        self.data_root = data_root
        self.model_variants = model_variants  # e.g. ["binary","simple","threeclass"] (no double)
        self.dropout_list = dropout_list
        self.weight_decay_list = weight_decay_list
        self.earlystop_list = earlystop_list
        self.class_weight_list = class_weight_list
        self.batch_size = batch_size
        self.lr = lr
        self.mini_epochs = mini_epochs
        self.device = device

        self.best_acc = -1.0
        self.best_params_str = ""
        self.all_results = []

    def run(self):
        import itertools
        param_space = list(itertools.product(
            self.model_variants,
            self.dropout_list,
            self.weight_decay_list,
            self.earlystop_list,
            self.class_weight_list
        ))
        total_combos = len(param_space)
        if total_combos == 0:
            self.log_msg.emit("No HPC combos specified!")
            self.finished.emit()
            return

        self.log_msg.emit(f"Starting HPC with {total_combos} combos, each ~{self.mini_epochs} epoch(s)...")

        os.makedirs("hpc_checkpoints", exist_ok=True)
        from src.experiment import Experiment

        for combo_idx, combo in enumerate(param_space):
            model_var   = combo[0]
            drop        = combo[1]
            wdec        = combo[2]
            estop       = combo[3]
            cw          = combo[4]

            combo_str = (
                f"model={model_var}, drop={drop}, "
                f"wdec={wdec}, eStop={estop}, cw={cw}"
            )
            self.log_msg.emit(f"\n[Combo {combo_idx+1}/{total_combos}] => {combo_str}")

            try:
                exp = Experiment(
                    data_root=self.data_root,
                    model_variant=model_var,
                    input_size=128,
                    batch_size=self.batch_size,
                    lr=self.lr,
                    epochs=self.mini_epochs,
                    disable_augment=True,   # removed references to adv_aug
                    adv_aug=False,          # removed references to adv_aug
                    dropout=drop,
                    weight_decay=wdec,
                    earlystop_patience=estop,
                    use_class_weight=cw,
                    device=self.device
                )
                exp.load_data()
                if hasattr(exp, "train_dataset") and getattr(exp.train_dataset, "is_empty", False):
                    self.log_msg.emit("  -> Skipping: dataset is EMPTY.")
                    self.all_results.append({
                        "combo_str": combo_str,
                        "val_acc": 0.0,
                        "confmat": None,
                        "roc": None,
                        "metrics": None,
                        "model_path": None,
                        "model_variant": model_var
                    })
                    continue

                exp._setup_optimizer_criterion()

            except Exception as e:
                self.log_msg.emit(f"Error building experiment: {e}")
                self.all_results.append({
                    "combo_str": combo_str,
                    "val_acc": 0.0,
                    "confmat": None,
                    "roc": None,
                    "metrics": None,
                    "model_path": None,
                    "model_variant": model_var
                })
                continue

            best_val_for_combo = 0.0
            best_state_dict = None
            best_confmat = None
            best_roc = None
            best_metrics = None
            no_improve_count = 0

            for ep in range(self.mini_epochs):
                train_loss, train_acc, _ = exp._train_one_epoch()
                val_metrics = exp._evaluate()
                val_acc = val_metrics["accuracy"]

                if val_acc > best_val_for_combo:
                    best_val_for_combo = val_acc
                    import copy
                    best_state_dict = copy.deepcopy(exp.model.state_dict())
                    best_confmat = val_metrics.get("confusion_matrix")
                    best_roc = {
                        "fpr": val_metrics.get("roc_fpr"),
                        "tpr": val_metrics.get("roc_tpr"),
                        "auc": val_metrics.get("roc_auc")
                    }
                    best_metrics = exp._compute_class_metrics(val_metrics)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if estop > 0 and no_improve_count >= estop:
                        self.log_msg.emit(f"  -> EarlyStop triggered at epoch {ep+1}")
                        break

                self.combo_epoch_update.emit(combo_idx, ep+1, train_acc, val_acc)
                self.log_msg.emit(f"  Ep {ep+1}/{self.mini_epochs}: train={train_acc:.1f}%, val={val_acc:.1f}%")

            if best_state_dict is not None:
                model_path = os.path.join("hpc_checkpoints", f"hpc_model_combo_{combo_idx+1}.pth")
                torch.save(best_state_dict, model_path)
            else:
                model_path = None

            self.all_results.append({
                "combo_str": combo_str,
                "val_acc": best_val_for_combo,
                "confmat": best_confmat,
                "roc": best_roc,
                "metrics": best_metrics,
                "model_path": model_path,
                "model_variant": model_var
            })

            if best_val_for_combo > self.best_acc:
                self.best_acc = best_val_for_combo
                self.best_params_str = combo_str

            pct = int(100*(combo_idx+1) / total_combos)
            self.progress.emit(pct)

        self.log_msg.emit(f"\nHPC DONE. Best val_acc={self.best_acc:.2f}% => {self.best_params_str}")
        self.finished.emit()


class FinalTrainWorker(QThread):
    """
    Performs final training on the best HPC combo.
    """
    finishedDetailed = pyqtSignal(float, dict, object, dict)
    progress = pyqtSignal(int)
    log_msg = pyqtSignal(str)
    epoch_update = pyqtSignal(int, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.best_params_str = ""
        self.data_root = "cleaned_data"
        self.epochs = 10
        self.batch = 32
        self.lr = 0.001
        self.folds = 1
        self.device = "cpu"

    def setup_final(self, best_params_str, data_root, epochs, batch, lr, folds, device="cpu"):
        self.best_params_str = best_params_str
        self.data_root = data_root
        self.epochs = epochs
        self.batch = batch
        self.lr = lr
        self.folds = folds
        self.device = device

    def run(self):
        if not self.best_params_str:
            self.log_msg.emit("No HPC combo chosen for final training.")
            self.finishedDetailed.emit(0.0, {}, None, {})
            return

        # parse param string, e.g. "model=..., drop=..., wdec=..., eStop=..., cw=..."
        parts = self.best_params_str.split(",")
        param_dict = {}
        for p in parts:
            if "=" in p:
                kv = p.split("=")
                if len(kv) == 2:
                    k, v = kv
                    param_dict[k.strip()] = v.strip()

        model_var = param_dict.get("model", "binary")
        drop       = float(param_dict.get("drop", "0.0"))
        wdec       = float(param_dict.get("wdec", "0.0"))
        estop      = int(param_dict.get("eStop", "0"))
        cw         = (param_dict.get("cw", "False") == "True")

        self.log_msg.emit(f"[Final Training] Using => {self.best_params_str}")
        self.log_msg.emit(f"Folds={self.folds}, epochs={self.epochs}, batch={self.batch}, lr={self.lr}")

        try:
            exp = Experiment(
                data_root=self.data_root,
                model_variant=model_var,
                input_size=128,
                batch_size=self.batch,
                lr=self.lr,
                epochs=self.epochs,
                disable_augment=True,
                adv_aug=False,
                dropout=drop,
                weight_decay=wdec,
                earlystop_patience=estop,
                use_class_weight=cw,
                device=self.device,
                folds=self.folds
            )
            exp.set_logger(self.log_msg.emit)
        except Exception as e:
            self.log_msg.emit(f"Failed building final experiment: {e}")
            self.progress.emit(0)
            self.finishedDetailed.emit(0.0, {}, None, {})
            return

        exp.load_data()
        if hasattr(exp, "train_dataset") and getattr(exp.train_dataset, "is_empty", False):
            self.log_msg.emit("Train dataset is empty => abort.")
            self.finishedDetailed.emit(0.0, {}, None, {})
            return

        exp._setup_optimizer_criterion()

        final_acc = 0.0
        if self.folds <= 1:
            best_val = 0.0
            no_improve = 0
            for ep in range(self.epochs):
                tr_loss, tr_acc, _ = exp._train_one_epoch()
                val_metrics = exp._evaluate()
                val_acc = val_metrics["accuracy"]

                if val_acc > best_val:
                    best_val = val_acc
                    no_improve = 0
                else:
                    no_improve += 1

                self.epoch_update.emit(ep+1, tr_acc, val_acc)
                pct = int(100*(ep+1)/self.epochs)
                self.progress.emit(pct)
                self.log_msg.emit(f"Ep {ep+1}: train={tr_acc:.1f}%, val={val_acc:.1f}%")

                if estop>0 and no_improve>=estop:
                    self.log_msg.emit(f"[EarlyStop at ep {ep+1}]")
                    break

            final_acc = best_val
        else:
            avg_acc, _, _, _ = exp.run_kfold(
                folds=self.folds,
                epochs=self.epochs
            )
            final_acc = avg_acc
            self.log_msg.emit(f"K-Fold => avg_val_acc={avg_acc:.2f}%")
            self.progress.emit(100)

        final_roc = exp.final_roc
        final_confmat = exp.final_confmat
        final_metrics = exp.final_class_metrics
        self.finishedDetailed.emit(final_acc, final_roc, final_confmat, final_metrics)


class AutoSearchTab(QWidget):
    """
    HPC search for (binary, simple, threeclass) but NOT double,
    plus final training of the best combo, storing confusion matrix & ROC.
    """

    hpc_updated = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setStyleSheet("""
            QWidget#AutoSearchTab {
                background-color: #f9f9f9;
            }
            QGroupBox {
                border: 1px solid #cccccc;
                border-radius: 6px;
                margin-top: 10px;
                padding: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 4px 10px;
                color: #0078D7;
            }
            QLabel {
                font-size: 10pt;
            }
            QPushButton {
                background-color: #0078d7;
                color: #ffffff;
                border: none;
                padding: 6px 14px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #005fa3;
            }
            QLineEdit, QTextEdit, QSpinBox, QComboBox {
                font-size: 9pt;
                padding: 4px;
            }
            QProgressBar {
                height: 20px;
                border-radius: 4px;
                text-align: center;
            }
            QTableWidget {
                font-size: 9pt;
            }
        """)
        self.setObjectName("AutoSearchTab")

        main_layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        cont_layout = QVBoxLayout(container)
        cont_layout.setContentsMargins(15,15,15,15)
        cont_layout.setSpacing(20)

        # HPC group
        hpc_group = QGroupBox("Hyperparameter Search (HPC)")
        hpc_form = QFormLayout(hpc_group)

        # Dataset folder
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["cleaned_data", "unprocessed_data", "augmented_data"])
        hpc_form.addRow("Dataset Folder:", self.dataset_combo)

        # Model variants => checkboxes, but NO double.
        self.model_ck_binary = QCheckBox("binary")
        self.model_ck_simple  = QCheckBox("simple")
        self.model_ck_threeclass = QCheckBox("threeclass")

        model_ck_layout = QHBoxLayout()
        model_ck_layout.addWidget(self.model_ck_binary)
        model_ck_layout.addWidget(self.model_ck_simple)
        model_ck_layout.addWidget(self.model_ck_threeclass)
        hpc_form.addRow(QLabel("Model Variants:"), model_ck_layout)

        # Dropout
        self.dropout_edit = QLineEdit("0.0,0.3")
        hpc_form.addRow("Dropout List:", self.dropout_edit)

        # Weight decay
        self.wdec_edit = QLineEdit("0.0,0.0001")
        hpc_form.addRow("Weight Decay List:", self.wdec_edit)

        # Early stop
        self.es_edit = QLineEdit("0,3")
        hpc_form.addRow("EarlyStop List:", self.es_edit)

        # Class Weight
        self.cw_edit = QLineEdit("False,True")
        hpc_form.addRow("Class Weight (List):", self.cw_edit)

        # batch size
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1,512)
        self.batch_spin.setValue(16)
        hpc_form.addRow("Batch Size:", self.batch_spin)

        # LR
        self.lr_edit = QLineEdit("0.001")
        hpc_form.addRow("Learning Rate:", self.lr_edit)

        # HPC mini-epochs
        self.mini_epoch_spin = QSpinBox()
        self.mini_epoch_spin.setRange(1,20)
        self.mini_epoch_spin.setValue(2)
        hpc_form.addRow("Mini-Epochs (partial):", self.mini_epoch_spin)

        self.btn_hpc = QPushButton("Run HPC Search")
        self.btn_hpc.clicked.connect(self.on_run_hpc)
        hpc_form.addRow(self.btn_hpc)

        cont_layout.addWidget(hpc_group)

        # HPC progress & log
        self.hpc_progress = QProgressBar()
        self.hpc_progress.setValue(0)
        self.hpc_log = QTextEdit()
        self.hpc_log.setReadOnly(True)
        self.hpc_log.setMinimumHeight(180)

        cont_layout.addWidget(self.hpc_progress)
        cont_layout.addWidget(self.hpc_log)

        self.hpc_canvas = MplCanvas(self, width=8, height=4, dpi=100)
        self.hpc_canvas.setMinimumSize(800, 400)
        self.hpc_toolbar = NavigationToolbar(self.hpc_canvas, self)

        cont_layout.addWidget(QLabel("<b>Current HPC Combo: Train vs. Val Accuracy</b>"))
        cont_layout.addWidget(self.hpc_canvas)
        cont_layout.addWidget(self.hpc_toolbar)

        # HPC Results Overview
        results_group = QGroupBox("HPC Results Overview")
        rg_layout = QVBoxLayout(results_group)

        self.hpc_table = QTableWidget()
        self.hpc_table.setColumnCount(4)
        self.hpc_table.setHorizontalHeaderLabels(["Rank","Combo","Val Acc","Model?"])
        self.hpc_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.hpc_table.setMinimumHeight(800)  # <--- bigger table
        rg_layout.addWidget(self.hpc_table)

        h_btn_row = QHBoxLayout()
        self.btn_show_info = QPushButton("Show HPC Info (Selected)")
        self.btn_show_info.clicked.connect(self.on_show_hpc_info)
        h_btn_row.addWidget(self.btn_show_info)

        self.btn_delete_model = QPushButton("Delete HPC Model (Selected)")
        self.btn_delete_model.clicked.connect(self.on_delete_model)
        h_btn_row.addWidget(self.btn_delete_model)

        rg_layout.addLayout(h_btn_row)

        self.hpc_info_text = QTextEdit()
        self.hpc_info_text.setReadOnly(True)
        self.hpc_info_text.setMinimumHeight(200)  # <--- bigger info text
        rg_layout.addWidget(self.hpc_info_text)

        self.hpc_bar_canvas = MplCanvas(self, width=8, height=3, dpi=100)
        self.hpc_bar_canvas.setMinimumSize(800, 450)  # <--- larger
        self.hpc_bar_toolbar = NavigationToolbar(self.hpc_bar_canvas, self)
        rg_layout.addWidget(QLabel("<b>HPC Combos: Validation Accuracy</b>"))
        rg_layout.addWidget(self.hpc_bar_canvas)
        rg_layout.addWidget(self.hpc_bar_toolbar)

        self.hpc_auc_canvas = MplCanvas(self, width=8, height=3, dpi=100)
        self.hpc_auc_canvas.setMinimumSize(800, 450)  # <--- larger
        self.hpc_auc_toolbar = NavigationToolbar(self.hpc_auc_canvas, self)
        rg_layout.addWidget(QLabel("<b>HPC Combos: ROC AUC (Binary Only)</b>"))
        rg_layout.addWidget(self.hpc_auc_canvas)
        rg_layout.addWidget(self.hpc_auc_toolbar)

        cont_layout.addWidget(results_group)

        # Final Training group
        final_box = QGroupBox("Final Training (Use HPC Combo)")
        final_form = QFormLayout(final_box)

        self.final_epochs_spin = QSpinBox()
        self.final_epochs_spin.setRange(1,200)
        self.final_epochs_spin.setValue(10)
        final_form.addRow("Final Epochs:", self.final_epochs_spin)

        self.final_batch_spin = QSpinBox()
        self.final_batch_spin.setRange(1,512)
        self.final_batch_spin.setValue(32)
        final_form.addRow("Final Batch Size:", self.final_batch_spin)

        self.final_lr_edit = QLineEdit("0.001")
        final_form.addRow("Final LR:", self.final_lr_edit)

        self.final_folds_spin = QSpinBox()
        self.final_folds_spin.setRange(1,10)
        self.final_folds_spin.setValue(1)
        final_form.addRow("Folds (K-Fold):", self.final_folds_spin)

        self.btn_final = QPushButton("Train Final Model")
        self.btn_final.setEnabled(False)
        self.btn_final.clicked.connect(self.on_train_final)
        final_form.addRow(self.btn_final)

        cont_layout.addWidget(final_box)

        self.final_progress = QProgressBar()
        self.final_progress.setValue(0)
        self.final_log = QTextEdit()
        self.final_log.setReadOnly(True)
        self.final_log.setMinimumHeight(100)
        cont_layout.addWidget(self.final_progress)
        cont_layout.addWidget(self.final_log)

        self.final_canvas = MplCanvas(container, width=8, height=4)
        self.final_canvas.setMinimumSize(800, 400)
        self.final_toolbar = NavigationToolbar(self.final_canvas, self)
        cont_layout.addWidget(QLabel("<b>Final Training: Train vs. Val Accuracy</b>"))
        cont_layout.addWidget(self.final_canvas)
        cont_layout.addWidget(self.final_toolbar)

        final_res_g = QGroupBox("Final Results: Confusion Matrix & ROC")
        fres_layout = QHBoxLayout(final_res_g)

        self.confmat_canvas = MplCanvas(self, width=4, height=3)
        self.confmat_canvas.setMinimumSize(400, 300)
        self.confmat_toolbar = NavigationToolbar(self.confmat_canvas, self)
        v_left = QVBoxLayout()
        v_left.addWidget(QLabel("<b>Confusion Matrix</b>"))
        v_left.addWidget(self.confmat_canvas)
        v_left.addWidget(self.confmat_toolbar)
        fres_layout.addLayout(v_left)

        self.roc_canvas = MplCanvas(self, width=4, height=3)
        self.roc_canvas.setMinimumSize(400, 300)
        self.roc_toolbar = NavigationToolbar(self.roc_canvas, self)
        v_right = QVBoxLayout()
        v_right.addWidget(QLabel("<b>ROC Curve (Binary/Double)</b>"))
        v_right.addWidget(self.roc_canvas)
        v_right.addWidget(self.roc_toolbar)
        fres_layout.addLayout(v_right)

        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        fres_layout.addWidget(self.metrics_text)

        cont_layout.addWidget(final_res_g)

        scroll.setWidget(container)
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

        self.hpc_worker = None
        self.current_combo_idx = -1
        self.hpc_train_acc = []
        self.hpc_val_acc = []
        self.best_params_str = ""
        self.all_results = []

        self.final_worker = None
        self.final_epoch_list = []
        self.final_train_acc_list = []
        self.final_val_acc_list = []
        self.final_roc_data = None
        self.final_confmat_data = None
        self.final_metrics_data = None

        # load existing HPC combos from JSON
        self._load_existing_hpc_results()

    def _load_existing_hpc_results(self):
        loaded = load_hpc_results_json()
        if loaded:
            self.all_results = []
            for item in loaded:
                rec = {
                    "combo_str":     item.get("combo_str",""),
                    "val_acc":       item.get("val_acc",0.0),
                    "model_path":    item.get("model_path",""),
                    "model_variant": item.get("model_variant",""),
                    "metrics":       item.get("metrics",None),
                    "roc":           item.get("roc",None),
                    "confmat":       item.get("confmat",None)
                }
                self.all_results.append(rec)

            self._update_hpc_table()
            self._plot_hpc_combo_chart()
            self._plot_hpc_auc_chart()

            best = max(self.all_results, key=lambda r: r["val_acc"], default=None)
            if best:
                self.best_params_str = best["combo_str"]
                self.hpc_log.append(f"Loaded HPC combos from JSON. Best => {self.best_params_str}")
                self.btn_final.setEnabled(True)

    def _save_current_hpc_results(self):
        save_hpc_results_json(self.all_results)

    def on_run_hpc(self):
        self.hpc_log.clear()
        self.hpc_progress.setValue(0)
        self.hpc_info_text.clear()
        self.hpc_table.clearContents()
        self.hpc_table.setRowCount(0)
        self.hpc_train_acc.clear()
        self.hpc_val_acc.clear()
        self._plot_hpc_training_chart()
        self._plot_hpc_combo_chart()
        self._plot_hpc_auc_chart()

        data_root = self.dataset_combo.currentText().strip()
        if not os.path.isdir(data_root):
            QMessageBox.warning(self, "Invalid Folder", f"'{data_root}' is not valid.")
            return

        chosen_models = []
        if self.model_ck_binary.isChecked():
            chosen_models.append("binary")
        if self.model_ck_simple.isChecked():
            chosen_models.append("simple")
        if self.model_ck_threeclass.isChecked():
            chosen_models.append("threeclass")
        if not chosen_models:
            QMessageBox.warning(self, "No Models", "Please check at least one model variant.")
            return

        def parse_csv_floats(text):
            res = []
            for t in text.split(","):
                t = t.strip()
                try:
                    val = float(t)
                    res.append(val)
                except:
                    pass
            return res

        def parse_csv_ints(text):
            res = []
            for t in text.split(","):
                t = t.strip()
                try:
                    val = int(t)
                    res.append(val)
                except:
                    pass
            return res

        def parse_csv_bools(text):
            res = []
            for t in text.split(","):
                t = t.strip().lower()
                if t in ("true","1"):
                    res.append(True)
                elif t in ("false","0"):
                    res.append(False)
            return res

        dropout_list = parse_csv_floats(self.dropout_edit.text())
        wdec_list = parse_csv_floats(self.wdec_edit.text())
        es_list = parse_csv_ints(self.es_edit.text())
        cw_list = parse_csv_bools(self.cw_edit.text())

        if not (dropout_list and wdec_list and es_list and cw_list):
            QMessageBox.critical(self, "Input Error", "One of the HPC lists is empty or invalid.")
            return

        batch_size = self.batch_spin.value()
        try:
            lr = float(self.lr_edit.text().strip())
        except:
            QMessageBox.critical(self, "LR Error", "LR must be float.")
            return

        mep = self.mini_epoch_spin.value()

        from .auto_search_tab import HPCSearchWorker
        self.hpc_worker = HPCSearchWorker()
        self.hpc_worker.setup_hpc(
            data_root=data_root,
            model_variants=chosen_models,
            dropout_list=dropout_list,
            weight_decay_list=wdec_list,
            earlystop_list=es_list,
            class_weight_list=cw_list,
            batch_size=batch_size,
            lr=lr,
            mini_epochs=mep,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.hpc_worker.progress.connect(self._on_hpc_progress)
        self.hpc_worker.log_msg.connect(self._on_hpc_log)
        self.hpc_worker.combo_epoch_update.connect(self._on_hpc_combo_epoch)
        self.hpc_worker.finished.connect(self._on_hpc_finished)

        self.btn_hpc.setEnabled(False)
        self.hpc_worker.start()

    def _on_hpc_progress(self, val):
        self.hpc_progress.setValue(val)

    def _on_hpc_log(self, msg):
        self.hpc_log.append(msg)

    def _on_hpc_combo_epoch(self, combo_idx, epoch_idx, tr_acc, val_acc):
        if combo_idx != self.current_combo_idx:
            self.current_combo_idx = combo_idx
            self.hpc_train_acc.clear()
            self.hpc_val_acc.clear()

        self.hpc_train_acc.append(tr_acc)
        self.hpc_val_acc.append(val_acc)
        self._plot_hpc_training_chart()

    def _plot_hpc_training_chart(self):
        self.hpc_canvas.figure.clear()
        ax = self.hpc_canvas.figure.add_subplot(111)
        ax.set_title("HPC Combo Training (Partial)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True)

        e_range = range(1, len(self.hpc_train_acc)+1)
        ax.plot(e_range, self.hpc_train_acc, marker='o', label="Train")
        ax.plot(e_range, self.hpc_val_acc, marker='s', label="Val")
        ax.legend()

        self.hpc_canvas.figure.tight_layout()
        self.hpc_canvas.draw()

    def _on_hpc_finished(self):
        self.hpc_log.append("=== HPC Finished ===")
        self.hpc_progress.setValue(100)
        self.btn_hpc.setEnabled(True)

        if self.hpc_worker:
            new_results = self.hpc_worker.all_results
            best_params_str = self.hpc_worker.best_params_str
            best_acc = self.hpc_worker.best_acc

            old_list = load_hpc_results_json()
            combined = old_list + new_results
            deduped = []
            seen = set()
            for item in combined:
                key = (item.get("combo_str",""), item.get("model_path",""))
                if key not in seen:
                    seen.add(key)
                    deduped.append(item)

            self.all_results = deduped
            self._save_current_hpc_results()

            self.best_params_str = best_params_str
            self.hpc_log.append(f"Best HPC => {best_params_str} (val_acc ~ {best_acc:.1f}%)")

            self._update_hpc_table()
            self._plot_hpc_combo_chart()
            self._plot_hpc_auc_chart()

            self.hpc_updated.emit()
            if self.best_params_str:
                self.btn_final.setEnabled(True)

        self.hpc_worker = None

    def _update_hpc_table(self):
        sorted_list = sorted(self.all_results, key=lambda x: x["val_acc"], reverse=True)
        self.hpc_table.setRowCount(len(sorted_list))
        for i, rec in enumerate(sorted_list):
            rank_item = QTableWidgetItem(str(i+1))
            combo_item = QTableWidgetItem(rec["combo_str"])
            acc_item = QTableWidgetItem(f"{rec['val_acc']:.2f}")
            has_model = "Yes" if rec["model_path"] else "No"
            model_item = QTableWidgetItem(has_model)

            self.hpc_table.setItem(i, 0, rank_item)
            self.hpc_table.setItem(i, 1, combo_item)
            self.hpc_table.setItem(i, 2, acc_item)
            self.hpc_table.setItem(i, 3, model_item)

    def _plot_hpc_combo_chart(self):
        self.hpc_bar_canvas.figure.clear()
        ax = self.hpc_bar_canvas.figure.add_subplot(111)
        if not self.all_results:
            ax.text(0.5, 0.5, "No HPC results yet", ha='center', va='center')
        else:
            xvals = range(len(self.all_results))
            acc_vals = [r["val_acc"] for r in self.all_results]
            best_idx = max(range(len(acc_vals)), key=lambda i: acc_vals[i])
            colors = ["steelblue"] * len(acc_vals)
            colors[best_idx] = "red"
            ax.bar(xvals, acc_vals, color=colors)
            ax.set_title("HPC Combos Validation Accuracy")
            ax.set_xlabel("Combo Index")
            ax.set_ylabel("Val Acc (%)")
            ax.set_xticks(xvals)
            ax.set_xticklabels([str(i+1) for i in xvals])
            ax.grid(axis='y', alpha=0.3)

        self.hpc_bar_canvas.figure.tight_layout()
        self.hpc_bar_canvas.draw()

    def _plot_hpc_auc_chart(self):
        self.hpc_auc_canvas.figure.clear()
        ax = self.hpc_auc_canvas.figure.add_subplot(111)
        if not self.all_results:
            ax.text(0.5, 0.5, "No HPC results yet", ha='center', va='center')
        else:
            auc_vals = []
            for rec in self.all_results:
                roc_obj = rec.get("roc", None)
                if roc_obj and roc_obj.get("auc", None) is not None:
                    auc_vals.append(float(roc_obj["auc"]))
                else:
                    auc_vals.append(0.0)

            xvals = range(len(auc_vals))
            best_idx = max(range(len(auc_vals)), key=lambda i: auc_vals[i])
            colors = ["purple"] * len(auc_vals)
            colors[best_idx] = "orange"
            ax.bar(xvals, auc_vals, color=colors)
            ax.set_title("HPC Combos ROC AUC (Binary Only)")
            ax.set_xlabel("Combo Index")
            ax.set_ylabel("AUC")
            ax.set_xticks(xvals)
            ax.set_xticklabels([str(i+1) for i in xvals])
            ax.grid(axis='y', alpha=0.3)

        self.hpc_auc_canvas.figure.tight_layout()
        self.hpc_auc_canvas.draw()

    def on_show_hpc_info(self):
        row = self.hpc_table.currentRow()
        if row < 0:
            self.hpc_info_text.setPlainText("No HPC row selected.")
            return

        combo_str = self.hpc_table.item(row, 1).text()
        val_acc_s = self.hpc_table.item(row, 2).text()
        try:
            val_acc = float(val_acc_s)
        except:
            val_acc = 0.0

        found = None
        for rec in self.all_results:
            if rec["combo_str"] == combo_str and abs(rec["val_acc"]-val_acc) < 1e-4:
                found = rec
                break

        if not found:
            self.hpc_info_text.setPlainText("Could not find HPC record.")
            return

        cm_str = self._format_confmat(found["confmat"])
        roc_str = self._format_roc(found["roc"])
        mets_str = self._format_metrics(found["metrics"])

        html = f"""
        <div style="border:2px solid #444;padding:10px;border-radius:6px;">
          <h3 style="margin-top:0;">HPC Info Card</h3>
          <p><b>Hyperparams:</b> {combo_str}</p>
          <p><b>Val Accuracy:</b> {val_acc:.2f}%</p>
          <p><b>Confusion Matrix:</b><br>{cm_str}</p>
          <p><b>ROC Info:</b><br>{roc_str}</p>
          <p><b>Class Metrics:</b><br>{mets_str}</p>
          <hr><em>Model file: {found['model_path'] if found['model_path'] else 'No checkpoint'}</em>
        </div>
        """
        self.hpc_info_text.setHtml(html)

    def _format_confmat(self, confmat):
        if confmat is None:
            return "N/A"
        arr = confmat.cpu().numpy()
        rows = []
        for row in arr:
            row_s = " ".join(f"{int(x):3d}" for x in row)
            rows.append(row_s)
        return f"<pre>{chr(10).join(rows)}</pre>"

    def _format_roc(self, roc_dict):
        if not roc_dict or roc_dict.get("fpr") is None:
            return "N/A"
        auc_v = roc_dict.get("auc", None)
        if auc_v is None:
            return "AUC = N/A"
        return f"AUC = {auc_v:.3f}"

    def _format_metrics(self, metrics):
        if not metrics:
            return "N/A"
        lines = []
        p = metrics.get("precision", [])
        r = metrics.get("recall", [])
        f1 = metrics.get("f1", [])
        sup = metrics.get("support", [])
        for i in range(len(p)):
            lines.append(f"Class {i}: p={p[i]:.3f}, r={r[i]:.3f}, f1={f1[i]:.3f}, sup={sup[i]}")
        micro = metrics.get("micro_f1",None)
        macro = metrics.get("macro_f1",None)
        if micro is not None:
            lines.append(f"Micro-F1: {micro:.3f}")
        if macro is not None:
            lines.append(f"Macro-F1: {macro:.3f}")
        return f"<pre>{chr(10).join(lines)}</pre>"

    def on_delete_model(self):
        row = self.hpc_table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "No row selected", "Select an HPC row first.")
            return

        combo_str = self.hpc_table.item(row, 1).text()
        val_acc_s = self.hpc_table.item(row, 2).text()
        try:
            val_acc = float(val_acc_s)
        except:
            val_acc = 0.0

        found_index = None
        for i, rec in enumerate(self.all_results):
            if rec["combo_str"] == combo_str and abs(rec["val_acc"] - val_acc) < 1e-5:
                found_index = i
                break

        if found_index is None:
            QMessageBox.information(self, "Not Found", "No matching HPC combo in memory.")
            return

        found = self.all_results[found_index]
        mpath = found["model_path"]
        if mpath and os.path.isfile(mpath):
            try:
                os.remove(mpath)
                self.hpc_log.append(f"Deleted HPC model file: {mpath}")
            except OSError as e:
                QMessageBox.critical(self, "Error", f"Could not delete: {e}")

        self.all_results.pop(found_index)
        self.hpc_log.append("Removed HPC combo from list.")

        self._save_current_hpc_results()
        self._update_hpc_table()
        self._plot_hpc_combo_chart()
        self._plot_hpc_auc_chart()

    def on_train_final(self):
        self.final_log.clear()
        self.final_progress.setValue(0)
        self.final_epoch_list.clear()
        self.final_train_acc_list.clear()
        self.final_val_acc_list.clear()
        self._plot_final_chart()

        self.final_roc_data = None
        self.final_confmat_data = None
        self.final_metrics_data = None
        self.metrics_text.clear()
        self._plot_final_confmat()
        self._plot_final_roc()

        if not self.best_params_str:
            QMessageBox.information(self, "No HPC Combo", "No best HPC combo to train.")
            return

        root_folder = self.dataset_combo.currentText().strip()
        if not os.path.isdir(root_folder):
            QMessageBox.warning(self, "Invalid Folder", f"{root_folder} not valid.")
            return

        ep = self.final_epochs_spin.value()
        bs = self.final_batch_spin.value()
        try:
            lr = float(self.final_lr_edit.text().strip())
        except:
            QMessageBox.critical(self, "LR Error", "Final LR must be float.")
            return
        folds = self.final_folds_spin.value()

        from .auto_search_tab import FinalTrainWorker
        self.final_worker = FinalTrainWorker()
        self.final_worker.setup_final(
            best_params_str=self.best_params_str,
            data_root=root_folder,
            epochs=ep,
            batch=bs,
            lr=lr,
            folds=folds,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.final_worker.log_msg.connect(self._on_final_log)
        self.final_worker.progress.connect(self._on_final_progress)
        self.final_worker.epoch_update.connect(self._on_final_epoch)
        self.final_worker.finishedDetailed.connect(self._on_final_finished)

        self.btn_final.setEnabled(False)
        self.final_worker.start()

    def _on_final_log(self, msg):
        self.final_log.append(msg)

    def _on_final_progress(self, val):
        self.final_progress.setValue(val)

    def _on_final_epoch(self, ep, train_acc, val_acc):
        self.final_epoch_list.append(ep)
        self.final_train_acc_list.append(train_acc)
        self.final_val_acc_list.append(val_acc)
        self._plot_final_chart()

    def _on_final_finished(self, final_acc, roc_dict, confmat, metrics):
        self.final_log.append(f"Final training done => acc={final_acc:.2f}%")
        self.final_progress.setValue(100)
        self.btn_final.setEnabled(True)

        self.final_roc_data = roc_dict
        self.final_confmat_data = confmat
        self.final_metrics_data = metrics

        self._plot_final_confmat()
        self._plot_final_roc()
        self._show_final_metrics()

        self.final_worker = None

    def _plot_final_chart(self):
        self.final_canvas.figure.clear()
        ax = self.final_canvas.figure.add_subplot(111)
        ax.set_title("Final Train vs. Val Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True)
        ax.plot(self.final_epoch_list, self.final_train_acc_list, marker='o', label="Train")
        ax.plot(self.final_epoch_list, self.final_val_acc_list, marker='s', label="Val")
        ax.legend()
        self.final_canvas.figure.tight_layout()
        self.final_canvas.draw()

    def _plot_final_confmat(self):
        self.confmat_canvas.figure.clear()
        ax = self.confmat_canvas.figure.add_subplot(111)
        if self.final_confmat_data is None:
            ax.text(0.5, 0.5, "No ConfMat found", ha='center', va='center')
        else:
            cm = self.final_confmat_data.cpu().numpy().astype(float)
            im = ax.imshow(cm, cmap="Blues")
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(int(cm[i,j])), ha='center', va='center')
            self.confmat_canvas.figure.colorbar(im, ax=ax)
        self.confmat_canvas.figure.tight_layout()
        self.confmat_canvas.draw()

    def _plot_final_roc(self):
        self.roc_canvas.figure.clear()
        ax = self.roc_canvas.figure.add_subplot(111)
        if not self.final_roc_data or self.final_roc_data.get("fpr") is None:
            ax.text(0.5, 0.5, "ROC not available", ha='center', va='center')
        else:
            fpr = self.final_roc_data["fpr"]
            tpr = self.final_roc_data["tpr"]
            auc_v = self.final_roc_data["auc"] if self.final_roc_data["auc"] else 0.0
            ax.plot(fpr, tpr, color='red', label=f"AUC={auc_v:.2f}")
            ax.plot([0,1],[0,1],"--", color='gray')
            ax.set_title("ROC Curve")
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.legend()
        self.roc_canvas.figure.tight_layout()
        self.roc_canvas.draw()

    def _show_final_metrics(self):
        self.metrics_text.clear()
        if not self.final_metrics_data:
            self.metrics_text.append("No class metrics found.")
            return
        prec = self.final_metrics_data.get("precision",[])
        rec = self.final_metrics_data.get("recall",[])
        f1s = self.final_metrics_data.get("f1",[])
        sup = self.final_metrics_data.get("support",[])
        self.metrics_text.append("<b>Class Metrics</b> (Precision/Recall/F1/Support)")
        for i in range(len(prec)):
            self.metrics_text.append(
                f"Class {i}: p={prec[i]:.3f}, r={rec[i]:.3f}, f1={f1s[i]:.3f}, sup={sup[i]}"
            )
        micro = self.final_metrics_data.get("micro_f1",None)
        macro = self.final_metrics_data.get("macro_f1",None)
        if micro is not None:
            self.metrics_text.append(f"\nMicro-F1: {micro:.3f}")
        if macro is not None:
            self.metrics_text.append(f"Macro-F1: {macro:.3f}")
# File: gui/compare_tab.py

import os
import glob
import numpy as np
from PIL import Image

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QHBoxLayout,
    QPushButton, QComboBox, QTextEdit, QMessageBox, QScrollArea
)
from PyQt5.QtCore import Qt

from gui.widgets import MplCanvas
from logic.comparison import load_image_as_grayscale


class CompareTab(QWidget):
    """
    Lets you browse images in a chosen dataset folder, then tries to find the
    same filename in other sets (raw, cleaned, augmented). Now shows:
      - The Original, Cleaned, plus *all* augmented variants side by side
      - Below each image, a histogram with mean/std/min/max/median
      - Color-coded borders for each type
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLayout(QVBoxLayout())

        info_label = QLabel(
            "<h3>Compare: Raw vs. Cleaned vs. Augmented (All Variants)</h3>"
            "Browse to see how one filename appears across raw, cleaned, and multiple augmentations."
        )
        info_label.setWordWrap(True)
        self.layout().addWidget(info_label)

        # -----------------------------------
        # Browsing controls
        # -----------------------------------
        browse_group = QGroupBox("Image Navigation")
        browse_layout = QHBoxLayout()

        self.dataset_combo = QComboBox()
        self.dataset_combo.setToolTip("Choose which dataset folder to scan.")
        self.dataset_combo.addItems([
            "data/archive/Training",
            "cleaned_data/Training",
            "augmented_data/Training"
        ])

        self.load_btn = QPushButton("Load Image List")
        self.load_btn.setToolTip("Find all images in the selected folder (recursively).")
        self.load_btn.clicked.connect(self.on_load_images)

        self.prev_btn = QPushButton("Previous")
        self.prev_btn.setToolTip("Go to the previous image in the list.")
        self.prev_btn.clicked.connect(self.on_prev_image)

        self.next_btn = QPushButton("Next")
        self.next_btn.setToolTip("Go to the next image in the list.")
        self.next_btn.clicked.connect(self.on_next_image)

        browse_layout.addWidget(self.dataset_combo)
        browse_layout.addWidget(self.load_btn)
        browse_layout.addWidget(self.prev_btn)
        browse_layout.addWidget(self.next_btn)
        browse_group.setLayout(browse_layout)
        self.layout().addWidget(browse_group)

        # -----------------------------------
        # ScrollArea to hold the big figure
        # -----------------------------------
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.layout().addWidget(self.scroll_area)

        # Log area
        self.log_box = QTextEdit()
        self.log_box.setToolTip("Shows which files were found plus info about augmented images.")
        self.log_box.setReadOnly(True)
        self.layout().addWidget(self.log_box)

        # Internals
        self.all_images = []
        self.curr_index = -1

    def on_load_images(self):
        chosen_path = self.dataset_combo.currentText().strip()
        if not os.path.isdir(chosen_path):
            QMessageBox.warning(self, "Error", f"{chosen_path} is not a valid folder.")
            return

        pattern_list = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            pattern_list.extend(glob.glob(os.path.join(chosen_path, "**", ext), recursive=True))

        self.all_images = sorted(pattern_list)
        self.curr_index = 0 if self.all_images else -1

        self.log_box.clear()
        if not self.all_images:
            self.log_box.append("No images found in this folder.")
        else:
            self.log_box.append(f"Loaded {len(self.all_images)} images from {chosen_path}.")
            self.show_comparison()

    def on_prev_image(self):
        if not self.all_images:
            return
        self.curr_index -= 1
        if self.curr_index < 0:
            self.curr_index = len(self.all_images) - 1
        self.show_comparison()

    def on_next_image(self):
        if not self.all_images:
            return
        self.curr_index += 1
        if self.curr_index >= len(self.all_images):
            self.curr_index = 0
        self.show_comparison()

    def show_comparison(self):
        if not self.all_images or self.curr_index < 0:
            return

        curr_image_path = self.all_images[self.curr_index]
        base_name = os.path.basename(curr_image_path)

        # We'll try to locate the same file in the raw, cleaned, and augmented sets
        root_orig = "data/archive/Training"
        root_clean = "cleaned_data/Training"
        root_aug = "augmented_data/Training"

        path_orig = self._locate_image(root_orig, base_name)
        path_clean = self._locate_image(root_clean, base_name)
        aug_paths = self._locate_all_aug_images(root_aug, base_name)

        # Build an ordered list for plotting
        images_and_labels = []
        if path_orig:
            images_and_labels.append((path_orig, "Original"))
        if path_clean:
            images_and_labels.append((path_clean, "Cleaned"))
        for i, ap in enumerate(aug_paths):
            images_and_labels.append((ap, f"Aug{i+1}"))

        from gui.widgets import MplCanvas
        new_canvas = MplCanvas(self, width=5, height=4)
        fig = new_canvas.figure

        if not images_and_labels:
            # No images found => placeholder
            fig.clear()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No matching images found in raw/cleaned/aug sets.", ha='center', va='center')
            ax.axis('off')
            new_canvas.draw()

            self.scroll_area.setWidget(new_canvas)
            self.log_box.clear()
            self.log_box.append(f"No matching images for '{base_name}' in raw/cleaned/aug.")
            return

        n_images = len(images_and_labels)

        # Fix for older Matplotlib: can't pass 'figsize' into subplots(), so do:
        fig.clear()
        fig.set_size_inches(4*n_images, 6)
        axes = fig.subplots(nrows=2, ncols=n_images)

        # If there's only 1 image, 'axes' won't be 2D => reshape to (2,1)
        if n_images == 1:
            axes = np.array([axes]).reshape(2,1)

        # Helper to color the spines
        def color_spines(ax, color="black", lw=2):
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(lw)

        for col_idx, (img_path, label_str) in enumerate(images_and_labels):
            ax_img = axes[0, col_idx]
            ax_hist = axes[1, col_idx]

            arr = load_image_as_grayscale(img_path)
            if arr is None:
                ax_img.text(0.5, 0.5, "Error reading", ha='center', va='center', color='red')
                ax_img.axis('off')
                ax_hist.text(0.5, 0.5, "No data", ha='center', va='center', color='red')
                ax_hist.axis('off')
                continue

            ax_img.imshow(arr, cmap='gray')
            ax_img.axis('off')
            ax_img.set_title(label_str, fontsize=10)

            # Color-code the border
            if label_str.lower().startswith("original"):
                color_spines(ax_img, "green")
            elif label_str.lower().startswith("clean"):
                color_spines(ax_img, "blue")
            else:
                # any "Aug" => orange
                color_spines(ax_img, "darkorange")

            # Build histogram
            flat = np.array(arr).ravel().astype(np.float32)
            ax_hist.hist(flat, bins=50, color='blue', alpha=0.6)
            ax_hist.set_xlabel("Intensity", fontsize=8)
            ax_hist.set_ylabel("Count", fontsize=8)

            # Compute stats
            mean_val = float(flat.mean())
            std_val = float(flat.std())
            mn_val = float(flat.min())
            mx_val = float(flat.max())
            med_val = float(np.median(flat))

            ax_hist.set_title(
                f"Mean={mean_val:.1f}, Std={std_val:.1f}\n"
                f"Min={mn_val:.0f}, Max={mx_val:.0f}, Med={med_val:.1f}",
                fontsize=9
            )

        fig.tight_layout()
        new_canvas.draw()
        self.scroll_area.setWidget(new_canvas)

        # Update the log box
        self.log_box.clear()
        self.log_box.append(f"<b>Comparing '{base_name}' across sets:</b>")
        self.log_box.append(f"Index {self.curr_index+1}/{len(self.all_images)} in {self.dataset_combo.currentText()}")
        self.log_box.append(f"Original path:   {path_orig if path_orig else 'Not found'}")
        self.log_box.append(f"Cleaned path:    {path_clean if path_clean else 'Not found'}")
        if aug_paths:
            self.log_box.append(f"Found {len(aug_paths)} augmented variant(s).")
            for i, ap in enumerate(aug_paths):
                self.log_box.append(f"   Aug{i+1}: {ap}")
        else:
            self.log_box.append("No augmented variants found for this file.")

    def _locate_image(self, root_dir, base_name):
        """
        Return the full path for base_name if found in root_dir (recursively).
        """
        if not os.path.isdir(root_dir):
            return None
        for subdir, dirs, files in os.walk(root_dir):
            if base_name in files:
                return os.path.join(subdir, base_name)
        return None

    def _locate_all_aug_images(self, root_dir, base_name):
        """
        Return a list of all augmented images (e.g. baseName_aug0.jpg),
        searching inside `root_dir`.
        We'll guess the 'stem' is base_name minus extension,
        then anything that starts with that + "_aug" + same ext.
        """
        if not os.path.isdir(root_dir):
            return []
        stem, ext = os.path.splitext(base_name)
        matches = []
        for subdir, dirs, files in os.walk(root_dir):
            for f in files:
                # check if it starts with <stem>_aug and ends with same extension
                if f.lower().startswith(stem.lower() + "_aug") and f.lower().endswith(ext.lower()):
                    matches.append(os.path.join(subdir, f))
        matches.sort()
        return matches
import os
import random
import numpy as np
from PIL import Image
import cv2  # For advanced histogram eq if needed

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QFormLayout,
    QPushButton, QSpinBox, QTextEdit, QMessageBox, QScrollArea,
    QSizePolicy, QHBoxLayout, QCheckBox, QComboBox, QDialog,
    QGridLayout
)
from PyQt5.QtCore import Qt, QPoint
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar
)
from matplotlib.backend_bases import MouseButton, PickEvent
from matplotlib.patches import Rectangle

from gui.widgets import MplCanvas


########################################################################
## HELPER FUNCTIONS
########################################################################
def gather_subfolders(root_folder):
    """
    Return a list of subfolder names under `root_folder` that themselves
    contain images or class folders. Typically: ["Training","Testing","Validation"].
    """
    if not os.path.isdir(root_folder):
        return []
    subfolders = []
    for d in os.listdir(root_folder):
        dpath = os.path.join(root_folder, d)
        if os.path.isdir(dpath):
            # check if it has images or sub-subfolders
            subfolders.append(d)
    return sorted(subfolders)


def gather_images_in_folder(folder_path, max_samples=None):
    """
    Return a list of all .png/.jpg/.jpeg images inside `folder_path` (recursively).
    If max_samples is set, limit the total count.
    """
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            ext = f.lower()
            if ext.endswith('.png') or ext.endswith('.jpg') or ext.endswith('.jpeg'):
                all_files.append(os.path.join(root, f))

    if max_samples and len(all_files) > max_samples:
        random.shuffle(all_files)
        return all_files[:max_samples]
    return all_files


def load_grayscale_float32(image_path):
    """
    Load a single image as grayscale float32. Returns a flattened array.
    On error, returns None.
    """
    try:
        from PIL import Image
        img = Image.open(image_path).convert('L')
        arr = np.array(img, dtype=np.float32)
        return arr.flatten()  # flatten for intensity stats
    except:
        return None


def gather_class_distribution(root_folder, subfolders):
    """
    For each subset (e.g. "Training", "Testing"), gather how many images
    are in each class subfolder (e.g. 'NoTumor', 'Pituitary', etc.).
    
    Returns a dict of shape:
      {
        "Training": {
           "NoTumor": 123,
           "Pituitary": 111,
           ...
        },
        "Testing": { ... }
      }
    """
    dist_info = {}
    for subset in subfolders:
        dist_info[subset] = {}
        subset_path = os.path.join(root_folder, subset)
        if not os.path.isdir(subset_path):
            continue

        # For each class folder in this subset:
        for cls_ in os.listdir(subset_path):
            cpath = os.path.join(subset_path, cls_)
            if os.path.isdir(cpath):
                # Count how many images
                cnt = 0
                for f in os.listdir(cpath):
                    lowf = f.lower()
                    if lowf.endswith('.png') or lowf.endswith('.jpg') or lowf.endswith('.jpeg'):
                        cnt += 1
                dist_info[subset][cls_] = cnt

    return dist_info


########################################################################
## DashboardTab (Upgraded)
########################################################################
class DashboardTab(QWidget):
    """
    Upgraded "DashboardTab" with:
      1. Multi-Subset Comparisons
      2. Advanced Stats (quartiles, box plots)
      3. Interactive histogram exploration
      4. Original sampling + distribution
      5. Compare Normalizations
      6. (NEW) Display class distribution counts in each folder
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Apply a unified stylesheet for a cleaner, modern look
        self.setStyleSheet("""
            QWidget#DashboardTab {
                background-color: #F7F7F7;
                font-family: "Segoe UI", Roboto, Arial, sans-serif;
            }

            QLabel {
                color: #333333;
                font-size: 11pt;
            }

            QGroupBox {
                border: 1px solid #CCCCCC;
                border-radius: 6px;
                margin-top: 8px;
                background-color: #FFFFFF;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                color: #0078D7;
                padding: 4px 10px;
                font-weight: bold;
            }

            QPushButton {
                background-color: #0078D7;
                color: white;
                font-size: 11pt;
                font-weight: 600;
                padding: 8px 16px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #005CB5;
            }
            QPushButton:disabled {
                background-color: #9FBADF;
            }

            QSpinBox, QLineEdit, QComboBox, QTextEdit {
                font-size: 10pt;
                padding: 4px;
            }
            QCheckBox {
                font-size: 10pt;
            }

            QTextEdit {
                background-color: #FFFFFF;
                border: 1px solid #CCCCCC;
                border-radius: 4px;
            }

            QScrollArea {
                border: none;
            }
        """)

        self.setObjectName("DashboardTab")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Scroll area for all content
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)

        # Container widget + layout
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(15)

        # Title and short description
        title_label = QLabel("<h2 style='color: #0078D7; margin-bottom: 0;'>Data Overview & Dashboard</h2>")
        title_label.setWordWrap(True)
        content_layout.addWidget(title_label)

        desc_label = QLabel(
            "Analyze class distribution, sample intensities, compare normalizations,\n"
            "and advanced features: multi-subset comparisons, box plots, interactive histogram exploration.\n"
            "Also see how many images exist in each folder/class."
        )
        desc_label.setWordWrap(True)
        content_layout.addWidget(desc_label)

        # --- Folder Selection & Analysis group ---
        select_group = QGroupBox("Folder Selection & Analysis")
        form_layout = QFormLayout()
        form_layout.setSpacing(10)

        self.folder_edit_label = QLabel("data/archive")  # just a label, or you can use QLineEdit
        self.folder_edit_label.setToolTip("Path containing subfolders like 'Training', 'Testing', etc.")
        self.folder_edit_label.setStyleSheet("font-weight: 600; color: #444444;")

        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(1, 200)
        self.sample_spin.setValue(30)
        self.sample_spin.setToolTip("How many images to sample for intensity stats (each subset).")

        # Button row
        self.btn_analyze = QPushButton("Analyze Folder")
        self.btn_analyze.setToolTip("Compute distribution & intensity stats for subsets, then plot.")
        self.btn_analyze.clicked.connect(self.on_analyze)

        h_buttons = QHBoxLayout()
        h_buttons.addWidget(self.btn_analyze)

        form_layout.addRow("Dataset Folder:", self.folder_edit_label)
        form_layout.addRow("Images to Sample:", self.sample_spin)
        form_layout.addRow(h_buttons)

        select_group.setLayout(form_layout)
        content_layout.addWidget(select_group)

        # Log box
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(150)
        content_layout.addWidget(self.log_box)

        # --- (NEW) Class Distribution group ---
        self.distribution_group = QGroupBox("Folder Info & Class Distribution")
        dist_layout = QVBoxLayout(self.distribution_group)
        dist_layout.setSpacing(10)

        dist_desc = QLabel(
            "Displays how many images exist in each subset folder\n"
            "and each class subfolder."
        )
        dist_desc.setWordWrap(True)
        dist_layout.addWidget(dist_desc)

        self.distribution_text = QTextEdit()
        self.distribution_text.setReadOnly(True)
        self.distribution_text.setMinimumHeight(220)  # make it taller
        dist_layout.addWidget(self.distribution_text)

        content_layout.addWidget(self.distribution_group)

        # Plot area (Histogram of intensities)
        self.plot_group = QGroupBox("Histogram of Sampled Intensities")
        plot_layout = QVBoxLayout()

        self.canvas = MplCanvas(self, width=5, height=4)
        self.canvas_toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.canvas)
        plot_layout.addWidget(self.canvas_toolbar)

        self.plot_group.setLayout(plot_layout)
        content_layout.addWidget(self.plot_group)

        # --- Advanced Stats group: box plots, quartiles, multi-subset selection ---
        self.adv_group = QGroupBox("Advanced Stats & Multi-Subset Comparison")
        adv_layout = QVBoxLayout(self.adv_group)
        adv_layout.setSpacing(10)

        # We'll dynamically find subfolders in the root. The user can pick which to compare.
        self.subset_checkboxes = []
        self.subset_box_layout = QVBoxLayout()
        adv_layout.addWidget(QLabel("Select subsets to compare (Training, Testing, Validation, etc.):"))
        adv_layout.addLayout(self.subset_box_layout)

        self.adv_run_btn = QPushButton("Compare Selected Subsets")
        self.adv_run_btn.clicked.connect(self.on_compare_subsets)
        adv_layout.addWidget(self.adv_run_btn)

        # Boxplot canvas
        self.box_canvas = MplCanvas(self, width=5, height=3)
        self.box_toolbar = NavigationToolbar(self.box_canvas, self)
        adv_layout.addWidget(self.box_canvas)
        adv_layout.addWidget(self.box_toolbar)

        # We'll have a text box for advanced stats
        self.adv_stats_box = QTextEdit()
        self.adv_stats_box.setReadOnly(True)
        adv_layout.addWidget(self.adv_stats_box)

        content_layout.addWidget(self.adv_group)

        # Footer note
        footer_label = QLabel(
            "<i>Note: This demonstration attempts to read images from the selected folder or the default <b>data/archive</b> if not changed.</i>"
        )
        footer_label.setWordWrap(True)
        content_layout.addWidget(footer_label)

        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)

        # Internal variables to store intensities for each subset
        self.subset_to_intensities = {}
        # For interactive bin clicking:
        self._current_hist_data = {}

    # --------------------------------------------------------------------------
    # Compare Normalization (placeholder - not heavily used in this demo)
    # --------------------------------------------------------------------------
    def on_compare_normalization(self):
        root_folder = self.folder_edit_label.text().strip()
        if not os.path.isdir(root_folder):
            QMessageBox.warning(self, "Invalid Folder", f"Path '{root_folder}' does not exist.")
            return
        # Placeholder for a normalization comparison dialog
        QMessageBox.information(self, "Normalization Comparison", "Dialog not implemented in this example.")

    # --------------------------------------------------------------------------
    # Main "Analyze Folder" => Finds subsets, samples intensities, plots them
    # + (NEW) Show class distribution in distribution_text
    # --------------------------------------------------------------------------
    def on_analyze(self):
        """
        1) Detect which subfolders exist (e.g. "Training","Testing","Validation") 
        2) For each subfolder, gather intensities from up to 'sample_spin' images
        3) Show them in a single combined histogram + show stats in the log
        4) Update the checkboxes in the "Advanced Stats" area for subset selection
        5) (NEW) Show how many images are in each folder/class in distribution_text
        """
        root_folder = self.folder_edit_label.text().strip()
        if not os.path.isdir(root_folder):
            QMessageBox.warning(self, "Invalid Folder", "Selected dataset folder not found.")
            return

        # Clear old data
        self.log_box.clear()
        self.subset_to_intensities.clear()
        self.distribution_text.clear()

        # 1) Find subfolders
        subfolders = gather_subfolders(root_folder)
        if not subfolders:
            self.log_box.append("No subfolders found (e.g. no 'Training','Testing'?). Aborting.")
            return

        # (NEW) Show class distribution
        dist_info = gather_class_distribution(root_folder, subfolders)
        self._show_class_distribution(dist_info)

        # 2) For each subfolder, gather random images -> intensities
        sample_count = self.sample_spin.value()
        for subset in subfolders:
            subset_path = os.path.join(root_folder, subset)
            all_imgs = gather_images_in_folder(subset_path, max_samples=sample_count)

            intensities = []
            error_count = 0
            for path in all_imgs:
                arr_f = load_grayscale_float32(path)
                if arr_f is None:
                    error_count += 1
                    continue
                intensities.extend(arr_f)

            self.subset_to_intensities[subset] = np.array(intensities, dtype=np.float32)
            self.log_box.append(f"<b>{subset}:</b> sampled {len(intensities)} intensities ({error_count} errors)")

        # 3) Combined histogram plot
        self.plot_combined_histogram()

        # 4) Update subset checkboxes
        #    Clear old checkboxes
        for cb in self.subset_checkboxes:
            self.subset_box_layout.removeWidget(cb)
            cb.deleteLater()
        self.subset_checkboxes.clear()

        for subset in subfolders:
            cb = QCheckBox(subset)
            cb.setChecked(True)
            self.subset_box_layout.addWidget(cb)
            self.subset_checkboxes.append(cb)

        self.log_box.append("\nAnalysis complete. Check 'Advanced Stats & Multi-Subset Comparison' for more.\n")

    def _show_class_distribution(self, dist_info):
        """
        Display the class/folder distribution in the distribution_text box.
        dist_info is a dict: { "Training": {"NoTumor": 100, ...}, "Testing": {...}, ...}
        """
        self.distribution_text.clear()
        self.distribution_text.append("<b>Class Distribution in Each Subfolder:</b>\n")
        for subset, class_map in dist_info.items():
            self.distribution_text.append(f"<u>{subset}</u>:")
            if not class_map:
                self.distribution_text.append("  (No classes or images found)")
            else:
                for cls_name, count in class_map.items():
                    self.distribution_text.append(f"  {cls_name}: {count}")
            self.distribution_text.append("")  # spacing

    def plot_combined_histogram(self):
        """
        Plot an overlapping histogram with a different color for each subset.
        We'll also enable pick events so user can click bars (interactive).
        """
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)

        colors = ["blue", "green", "red", "orange", "purple", "teal"]
        color_idx = 0
        self._current_hist_data.clear()  # store bin info for interactive clicks

        for subset, data in self.subset_to_intensities.items():
            if len(data) == 0:
                continue
            c = colors[color_idx % len(colors)]
            color_idx += 1
            # Plot histogram with alpha
            counts, bins, patches = ax.hist(
                data,
                bins=50,
                alpha=0.4,
                label=subset,
                color=c,
                picker=True  # enable picking
            )
            # We'll store the bin edges & counts so we can handle pick events
            for p in patches:
                p.set_picker(True)

        ax.set_title("Combined Histogram of Sampled Intensities")
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Count")
        ax.legend()

        # Connect pick event
        self.canvas.mpl_connect("pick_event", self.on_hist_pick)

        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def on_hist_pick(self, event: PickEvent):
        """
        Called when user clicks on a bar (patch) in the histogram.
        We'll attempt to identify the bin range, then show images in that range if feasible.
        """
        artist = event.artist
        if not isinstance(artist, Rectangle):
            return

        # Extract bin range from rectangle
        x_left = artist.get_x()
        width = artist.get_width()
        bin_min = x_left
        bin_max = x_left + width

        self.log_box.append(f"<b>Clicked bin:</b> {bin_min:.1f} to {bin_max:.1f}<br>")

        # Just a simple info dialog with the counts in that bin for each subset
        count_in_range = {}
        for subset, intensities in self.subset_to_intensities.items():
            sub_vals = intensities[(intensities >= bin_min) & (intensities < bin_max)]
            count_in_range[subset] = len(sub_vals)

        lines = []
        for subset, cnt in count_in_range.items():
            lines.append(f"{subset}: {cnt} intensities in that bin.")
        detail_str = "\n".join(lines)

        msg = f"You clicked bin {bin_min:.1f}–{bin_max:.1f}.\n\n{detail_str}"
        QMessageBox.information(self, "Histogram Bin Info", msg)

    # --------------------------------------------------------------------------
    # "Compare Subsets" button => do box plots, advanced stats
    # --------------------------------------------------------------------------
    def on_compare_subsets(self):
        """
        1) Gather whichever subsets are checked.
        2) Show a box plot for intensities in each subset side by side.
        3) Display quartiles, min, max, outliers in adv_stats_box.
        4) Possibly highlight if one subset is systematically darker or lighter.
        """
        chosen_subsets = []
        for cb in self.subset_checkboxes:
            if cb.isChecked():
                chosen_subsets.append(cb.text())

        if not chosen_subsets:
            QMessageBox.warning(self, "No subsets selected", "Please check at least one subset.")
            return

        # Prepare data for box plot
        data_for_boxplot = []
        subset_labels = []
        for s in chosen_subsets:
            arr = self.subset_to_intensities.get(s, None)
            if arr is not None and len(arr) > 0:
                data_for_boxplot.append(arr)
                subset_labels.append(s)

        if not data_for_boxplot:
            QMessageBox.information(self, "No Data", "No intensities found for the selected subsets.")
            return

        self.box_canvas.figure.clear()
        ax = self.box_canvas.figure.add_subplot(111)

        # Box plot
        ax.boxplot(data_for_boxplot, labels=subset_labels, showfliers=True)
        ax.set_title("Subset Comparison - Box Plot of Intensities")
        ax.set_ylabel("Pixel Intensity")

        self.box_canvas.figure.tight_layout()
        self.box_canvas.draw()

        # Advanced stats
        self.adv_stats_box.clear()
        self.adv_stats_box.append("<b>Advanced Stats</b> (min, 25%, median, 75%, max)\n")

        all_medians = []
        for i, subset_name in enumerate(subset_labels):
            arr = data_for_boxplot[i]
            stats = np.percentile(arr, [0, 25, 50, 75, 100])
            linestr = (
                f"{subset_name} => "
                f"min={stats[0]:.1f}, Q1={stats[1]:.1f}, median={stats[2]:.1f}, "
                f"Q3={stats[3]:.1f}, max={stats[4]:.1f}"
            )
            self.adv_stats_box.append(linestr)
            all_medians.append(stats[2])

        avg_median = np.mean(all_medians)
        self.adv_stats_box.append(f"\nAverage median across chosen subsets: {avg_median:.1f}")

        threshold = 20
        for i, med in enumerate(all_medians):
            if abs(med - avg_median) >= threshold:
                subset_name = subset_labels[i]
                self.adv_stats_box.append(
                    f"<b>Warning:</b> {subset_name} median differs from average by >= {threshold}!"
                )
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
        self.results_table.setMinimumHeight(1000)
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
# File: gui/info_tab.py

import os
import random
import numpy as np
from PIL import Image

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QPushButton,
    QTextEdit, QMessageBox, QHBoxLayout, QSlider, QScrollArea,
    QComboBox, QLineEdit, QSpinBox, QFileDialog, QFormLayout
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# If you’re using your own MplCanvas:
from gui.widgets import MplCanvas

# Logic modules
from logic.advanced_eda import (
    gather_random_images,
    collect_resolutions,
    histogram_equalize,
    gather_subset_resolutions
)
# Note: Removed imports for detect_poor_quality and augmentation preview.

class InfoTab(QWidget):
    """
    An “ultimate guide” tab that:
      1) Explains each application tab (Preprocessing, etc.).
      2) Demonstrates how they work with small interactive demos:
         - resolution stats
         - histogram equalization
         - gamma correction
         - subset comparisons
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Master layout: Use a scroll area to comfortably handle the many sections
        main_layout = QVBoxLayout(self)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)

        # Container for all content
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(12)

        # ------------------- Title & Intro -------------------
        title_label = QLabel("<h2>Ultimate Guide & Demonstrations</h2>")
        title_label.setWordWrap(True)
        content_layout.addWidget(title_label)

        intro_label = QLabel(
            "This tab provides an all-in-one overview of the other tabs, "
            "plus interactive EDA demos to illustrate how your data changes "
            "during Preprocessing and more."
        )
        intro_label.setWordWrap(True)
        content_layout.addWidget(intro_label)

        # ------------------- Overview of Tabs -------------------
        overview_group = QGroupBox("Overview of Other Tabs")
        overview_layout = QVBoxLayout(overview_group)

        overview_layout.addWidget(self._make_label(
            "<b>Dashboard Tab:</b> Offers general data overviews—like class distributions or basic stats."
        ))
        overview_layout.addWidget(self._make_label(
            "<b>Info Tab (current):</b> This instructive tab with live EDA demos."
        ))
        overview_layout.addWidget(self._make_label(
            "<b>Preprocessing Tab:</b> Cleans raw data, removes duplicates, applies blur/sharpen, histogram eq, etc."
        ))
        overview_layout.addWidget(self._make_label(
            "<b>Augmentation Tab:</b> Generates synthetic images (rotate, flip, brightness, etc.), optionally balancing classes."
        ))
        overview_layout.addWidget(self._make_label(
            "<b>Comparison Tab:</b> Let’s you see the same file in raw, cleaned, or augmented sets side by side."
        ))
        # Removed the HPC line from the overview
        overview_layout.addWidget(self._make_label(
            "<b>Inference Tab:</b> Loads a final trained model to classify a single image."
        ))

        content_layout.addWidget(overview_group)

        # ------------------- Analysis Controls -------------------
        controls_box = QGroupBox("Analysis Controls & Settings")
        controls_form = QFormLayout(controls_box)

        # Root folder
        self.root_edit = QLineEdit("data/archive")
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._on_browse_folder)
        h_line = QHBoxLayout()
        h_line.addWidget(self.root_edit)
        h_line.addWidget(btn_browse)
        controls_form.addRow("Root Folder:", h_line)

        # Max images
        self.max_images_spin = QSpinBox()
        self.max_images_spin.setRange(1, 1000)
        self.max_images_spin.setValue(200)
        controls_form.addRow("Max Images to Sample:", self.max_images_spin)

        # Removed the poor-quality detection checkbox

        # Resolution bin slider
        self.res_bin_slider = QSlider(Qt.Horizontal)
        self.res_bin_slider.setRange(5, 50)
        self.res_bin_slider.setValue(20)
        controls_form.addRow("Resolution Histogram Bins:", self.res_bin_slider)

        # hist eq image
        self.eq_img_edit = QLineEdit()
        self.eq_img_edit.setPlaceholderText("Optional image for histogram eq; or leave blank for random.")
        eq_btn = QPushButton("Pick...")
        eq_btn.clicked.connect(self._on_pick_eq_image)
        eq_h = QHBoxLayout()
        eq_h.addWidget(self.eq_img_edit)
        eq_h.addWidget(eq_btn)
        controls_form.addRow("Histogram EQ Image:", eq_h)

        # Button to run all demos
        self.run_demos_btn = QPushButton("Run Demos")
        self.run_demos_btn.clicked.connect(self.on_run_demos)
        controls_form.addRow(self.run_demos_btn)

        content_layout.addWidget(controls_box)

        # ------------------- Resolution Demo -------------------
        self.res_group = QGroupBox("Resolution Demo")
        res_layout = QVBoxLayout(self.res_group)

        self.res_text = QTextEdit()
        self.res_text.setReadOnly(True)
        res_layout.addWidget(self.res_text)

        self.res_canvas = MplCanvas(self, width=7, height=5)  # bigger chart
        self.res_toolbar = NavigationToolbar(self.res_canvas, self)
        res_layout.addWidget(self.res_canvas)
        res_layout.addWidget(self.res_toolbar)

        content_layout.addWidget(self.res_group)

        # ------------------- Histogram EQ Demo -------------------
        self.eq_group = QGroupBox("Histogram Equalization Demo")
        eq_layout = QVBoxLayout(self.eq_group)

        eq_label = QLabel("Shows a PIL grayscale image (left) and the CLAHE-equalized version (right).")
        eq_label.setWordWrap(True)
        eq_layout.addWidget(eq_label)

        self.eq_canvas = MplCanvas(self, width=7, height=5)
        self.eq_toolbar = NavigationToolbar(self.eq_canvas, self)
        eq_layout.addWidget(self.eq_canvas)
        eq_layout.addWidget(self.eq_toolbar)

        content_layout.addWidget(self.eq_group)

        # ------------------- Gamma Demo -------------------
        self.gamma_group = QGroupBox("Gamma Correction Demo")
        gamma_layout = QVBoxLayout(self.gamma_group)

        gamma_label = QLabel(
            "Move the slider to lighten/darken the image. (<1 => lighten, >1 => darken). "
            "Similar transformations appear in the Augmentation Tab."
        )
        gamma_label.setWordWrap(True)
        gamma_layout.addWidget(gamma_label)

        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(50, 200)  # 0.5..2.0 scaled by 100
        self.gamma_slider.setValue(100)
        self.gamma_slider.valueChanged.connect(self.on_gamma_slider_changed)
        gamma_layout.addWidget(self.gamma_slider)

        self.gamma_canvas = MplCanvas(self, width=7, height=5)
        self.gamma_toolbar = NavigationToolbar(self.gamma_canvas, self)
        gamma_layout.addWidget(self.gamma_canvas)
        gamma_layout.addWidget(self.gamma_toolbar)

        content_layout.addWidget(self.gamma_group)

        # ------------------- Subset Comparison Demo -------------------
        self.subset_group = QGroupBox("Subset Comparison Demo")
        subset_layout = QVBoxLayout(self.subset_group)

        subset_label = QLabel(
            "Compare resolution distributions across subsets (Training, Testing, Both)."
        )
        subset_label.setWordWrap(True)
        subset_layout.addWidget(subset_label)

        self.subset_combo = QComboBox()
        self.subset_combo.addItems(["Training only", "Testing only", "Both"])
        subset_layout.addWidget(self.subset_combo)

        self.subset_btn = QPushButton("Compare Subsets")
        self.subset_btn.clicked.connect(self.on_compare_subsets)
        subset_layout.addWidget(self.subset_btn)

        self.subset_canvas = MplCanvas(self, width=7, height=5)
        self.subset_toolbar = NavigationToolbar(self.subset_canvas, self)
        subset_layout.addWidget(self.subset_canvas)
        subset_layout.addWidget(self.subset_toolbar)

        content_layout.addWidget(self.subset_group)

        # Removed the Augmentation Preview Demo block
        # Removed the HPC Mini-Demo block

        # ------------------- Log / Output -------------------
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        content_layout.addWidget(self.log_area)

        # Footer
        footer_label = QLabel(
            "<i>All demos above are simplified examples. For the real pipeline, see each dedicated tab.</i>"
        )
        footer_label.setWordWrap(True)
        content_layout.addWidget(footer_label)

        scroll_area.setWidget(content_widget)
        self.setLayout(main_layout)

        # Internals
        self.gamma_sample_img = None
        self.selected_eq_image = None

    # -----------------------------------------
    # Helper: a word-wrapped QLabel
    # -----------------------------------------
    def _make_label(self, text):
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        return lbl

    # -----------------------------------------
    # Folder & EQ image pick
    # -----------------------------------------
    def _on_browse_folder(self):
        dlg = QFileDialog(self, "Select Root Folder")
        dlg.setFileMode(QFileDialog.Directory)
        if dlg.exec_():
            self.root_edit.setText(dlg.selectedFiles()[0])

    def _on_pick_eq_image(self):
        dlg = QFileDialog(self, "Pick an image for histogram eq", self.root_edit.text() or ".")
        dlg.setNameFilter("Images (*.png *.jpg *.jpeg)")
        if dlg.exec_():
            chosen = dlg.selectedFiles()[0]
            self.eq_img_edit.setText(chosen)
            self.selected_eq_image = chosen

    # -----------------------------------------
    # “Run Demos” => resolution stats, eq, gamma
    # -----------------------------------------
    def on_run_demos(self):
        root_folder = self.root_edit.text().strip()
        if not root_folder or not os.path.isdir(root_folder):
            QMessageBox.warning(self, "Invalid Folder", "Please select a valid folder containing images.")
            return

        # Clear logs
        self.log_area.clear()
        self.res_text.clear()

        # Gather random images
        subfolders = ["Training", "Testing"]
        max_imgs = self.max_images_spin.value()
        all_imgs = gather_random_images(root_folder, subfolders, max_samples=max_imgs)
        if not all_imgs:
            self.log_area.append("No images found in selected folder.")
            return

        # 1) resolution stats
        widths, heights, err_count = collect_resolutions(all_imgs)
        if err_count > 0:
            self.log_area.append(f"Error reading {err_count} images.\n")
        if widths and heights:
            wmin, wmax = min(widths), max(widths)
            hmin, hmax = min(heights), max(heights)
            wmean, hmean = np.mean(widths), np.mean(heights)
            self.res_text.append(f"<b>Resolution Summary (up to {max_imgs} images)</b>\n")
            self.res_text.append(
                f"Width => min:{wmin}, max:{wmax}, mean:{wmean:.1f}\n"
                f"Height => min:{hmin}, max:{hmax}, mean:{hmean:.1f}\n"
            )
        else:
            self.res_text.append("No valid images for resolution stats.\n")

        # 2) plot resolution distribution
        bins = self.res_bin_slider.value()
        self._plot_res_distributions(widths, heights, bins)

        # 3) hist eq => pick user-chosen or random
        eq_path = self.eq_img_edit.text().strip()
        if eq_path and os.path.isfile(eq_path):
            self.selected_eq_image = eq_path
        else:
            self.selected_eq_image = random.choice(all_imgs)
        self._show_hist_eq_demo(self.selected_eq_image)

        # 4) gamma => pick random
        self.gamma_sample_img = random.choice(all_imgs)
        self._update_gamma_demo()

        self.log_area.append("\nDemos complete.\n")

    def _plot_res_distributions(self, widths, heights, bins):
        self.res_canvas.figure.clear()
        fig = self.res_canvas.figure

        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        ax1.hist(widths, bins=bins, color='blue', alpha=0.6)
        ax1.set_title("Widths")
        ax1.set_xlabel("Pixels")
        ax1.set_ylabel("Count")

        ax2.hist(heights, bins=bins, color='green', alpha=0.6)
        ax2.set_title("Heights")
        ax2.set_xlabel("Pixels")

        ax3.scatter(widths, heights, alpha=0.3)
        ax3.set_title("Width vs. Height")
        ax3.set_xlabel("Width")
        ax3.set_ylabel("Height")

        fig.tight_layout()
        self.res_canvas.draw()

    def _show_hist_eq_demo(self, path):
        self.eq_canvas.figure.clear()
        if not path:
            return
        try:
            with Image.open(path).convert('L') as im:
                ax1, ax2 = self.eq_canvas.figure.subplots(1, 2)

                ax1.imshow(im, cmap='gray')
                ax1.set_title("Original")
                ax1.axis('off')

                eq_img = histogram_equalize(im)
                ax2.imshow(eq_img, cmap='gray')
                ax2.set_title("Equalized")
                ax2.axis('off')

                self.eq_canvas.figure.tight_layout()
                self.eq_canvas.draw()

                self.log_area.append(f"Histogram EQ on: {os.path.basename(path)}\n")

        except Exception as e:
            self.log_area.append(f"Histogram EQ error: {e}\n")

    # -----------------------------------------
    # Gamma Correction
    # -----------------------------------------
    def on_gamma_slider_changed(self, val):
        if self.gamma_sample_img:
            self._update_gamma_demo()

    def _update_gamma_demo(self):
        self.gamma_canvas.figure.clear()
        gamma_val = self.gamma_slider.value() / 100.0

        try:
            with Image.open(self.gamma_sample_img).convert('L') as im:
                lut = [pow(i / 255.0, gamma_val)*255 for i in range(256)]
                out_img = im.point(lut)

                ax = self.gamma_canvas.figure.subplots(1, 1)
                ax.imshow(out_img, cmap='gray')
                ax.set_title(f"Gamma = {gamma_val:.2f}")
                ax.axis('off')

                self.gamma_canvas.figure.tight_layout()
                self.gamma_canvas.draw()

        except Exception as e:
            self.log_area.append(f"Gamma error: {e}\n")

    # -----------------------------------------
    # Subset Comparison
    # -----------------------------------------
    def on_compare_subsets(self):
        root_folder = self.root_edit.text().strip()
        if not root_folder or not os.path.isdir(root_folder):
            QMessageBox.warning(self, "Invalid Folder", f"Folder '{root_folder}' not found.")
            return

        choice = self.subset_combo.currentText()
        if choice == "Training only":
            sub_list = ["Training"]
        elif choice == "Testing only":
            sub_list = ["Testing"]
        else:
            sub_list = ["Training", "Testing"]

        w_dict, h_dict = gather_subset_resolutions(root_folder, sub_list)
        self._plot_subset_comparison(w_dict, h_dict)

    def _plot_subset_comparison(self, subset_w, subset_h):
        self.subset_canvas.figure.clear()
        fig = self.subset_canvas.figure

        subs = sorted(subset_w.keys())
        if not subs:
            self.log_area.append("No subsets found to compare.\n")
            return

        if len(subs) == 1:
            # single subset => top: widths, bottom: heights
            ax1 = fig.add_subplot(2, 1, 1)
            wdata = subset_w[subs[0]]
            ax1.hist(wdata, bins=30, alpha=0.6, color='blue')
            ax1.set_title(f"{subs[0]} - Widths")

            ax2 = fig.add_subplot(2, 1, 2)
            hdata = subset_h[subs[0]]
            ax2.hist(hdata, bins=30, alpha=0.6, color='green')
            ax2.set_title(f"{subs[0]} - Heights")
        else:
            # overlapped
            ax1 = fig.add_subplot(2, 1, 1)
            for s in subs:
                ax1.hist(subset_w[s], bins=30, alpha=0.4, label=s)
            ax1.set_title("Widths (Overlapped)")
            ax1.legend()

            ax2 = fig.add_subplot(2, 1, 2)
            for s in subs:
                ax2.hist(subset_h[s], bins=30, alpha=0.4, label=s)
            ax2.set_title("Heights (Overlapped)")
            ax2.legend()

        fig.tight_layout()
        self.subset_canvas.draw()
import sys
from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QStyleFactory, QApplication
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

# Import existing tabs
from gui.dashboard_tab import DashboardTab
from gui.preprocessing_tab import PreprocessingTab
from gui.augmentation_tab import AugmentationTab
from gui.compare_tab import CompareTab
from gui.auto_search_tab import AutoSearchTab
from gui.info_tab import InfoTab
from gui.inference_tab import InferenceTab

class MainWindow(QMainWindow):
    """
    A single-application window that houses all the tabs:
      - Dashboard
      - Info
      - Preprocessing
      - Augmentation
      - Comparison
      - Auto Search
      - Inference
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Brain Tumor MRI Toolkit")
        self.setWindowIcon(QIcon())  # If you have an icon, set it or remove

        self.resize(1200, 800)
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)

        # Add tabs
        self.tab_widget.addTab(DashboardTab(), "Dashboard")
        self.tab_widget.addTab(InfoTab(), "Info")
        self.tab_widget.addTab(PreprocessingTab(), "Preprocessing")
        self.tab_widget.addTab(AugmentationTab(), "Augmentation")
        self.tab_widget.addTab(CompareTab(), "Comparison")

        # HPC (AutoSearch)
        self.auto_search_tab = AutoSearchTab()
        self.tab_widget.addTab(self.auto_search_tab, "Auto Search")

        # Inference
        self.inference_tab = InferenceTab()
        self.tab_widget.addTab(self.inference_tab, "Inference")

        # CONNECT HPC's "hpc_updated" => Inference tab reload
        self.auto_search_tab.hpc_updated.connect(self.inference_tab.reload_hpc_combos)

        self.setCentralWidget(self.tab_widget)
        self._setup_modern_style()

    def _setup_modern_style(self):
        self.setStyle(QStyleFactory.create("Fusion"))
        qss = """
        QMainWindow { background-color: #f9f9f9; }
        QWidget { font-family: "Segoe UI"; background-color: #ffffff; font-size: 10pt; }
        QTabWidget::pane { background-color: #ffffff; border: 1px solid #ccc; }
        QTabBar::tab {
            background: #f3f3f3; border: 1px solid #ccc; padding: 8px 14px; margin-right: 2px;
            border-top-left-radius: 6px; border-top-right-radius: 6px; min-width: 120px;
        }
        QTabBar::tab:selected { background: #ffffff; border-bottom: 1px solid #ffffff; }
        QTabBar::tab:hover { background-color: #e8e8e8; }
        QGroupBox {
            border: 1px solid #ddd; border-radius: 8px; margin-top: 10px; padding: 10px;
        }
        QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 8px; }
        QPushButton {
            background-color: #0078d7; color: #ffffff; border: none;
            padding: 8px 14px; border-radius: 6px;
        }
        QPushButton:hover { background-color: #005fa3; }
        QProgressBar {
            border: 1px solid #aaa; border-radius: 4px; text-align: center; height: 20px;
        }
        QProgressBar::chunk { background-color: #0078d7; width: 10px; }
        QTextEdit, QLineEdit, QSpinBox, QComboBox {
            border: 1px solid #ccc; border-radius: 4px; padding: 4px;
        }
        QLabel { font-size: 10pt; }
        """
        self.setStyleSheet(qss)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
# File: gui/preprocessing_tab.py

import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QFormLayout, QCheckBox,
    QLineEdit, QPushButton, QProgressBar, QMessageBox, QTextEdit,
    QHBoxLayout, QFileDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from logic.preprocessing import preprocess_entire_dataset


class PreprocessingWorker(QThread):
    """
    QThread to handle copying, filtering (blur/sharpen), histogram eq,
    resizing, etc. without blocking the UI.
    """
    progress = pyqtSignal(int, str)
    # We'll now emit 5 items:
    #  dict -> dist_info
    #  list -> error_log
    #  int  -> total_processed
    #  int  -> dups_removed
    #  float-> elapsed_sec
    finished = pyqtSignal(dict, list, int, int, float)

    def __init__(
        self,
        root_folder,
        output_folder,
        resize_val,
        delete_duplicates=False,
        apply_hist_eq=False,
        filter_mode="none",
        interpolation_mode="BILINEAR",
        parent=None
    ):
        super().__init__(parent)
        self.root_folder = root_folder
        self.output_folder = output_folder
        self.resize_val = resize_val
        self.delete_duplicates = delete_duplicates
        self.apply_hist_eq = apply_hist_eq
        self.filter_mode = filter_mode
        self.interpolation_mode = interpolation_mode

    def run(self):
        try:
            def progress_callback(current, total):
                pct = int((current / total) * 100)
                msg = f"Processing image {current} of {total}"
                self.progress.emit(pct, msg)

            self.progress.emit(0, "Starting Preprocessing...")
            (dist_info,
             error_log,
             total_processed,
             dups_removed,
             elapsed_sec) = preprocess_entire_dataset(
                root_dir=self.root_folder,
                dst_root=self.output_folder,
                resize_val=self.resize_val,
                delete_duplicates=self.delete_duplicates,
                apply_hist_eq=self.apply_hist_eq,
                filter_mode=self.filter_mode,
                interpolation=self.interpolation_mode,
                progress_fn=progress_callback
            )
            self.progress.emit(100, "All done!")
            self.finished.emit(dist_info, error_log, total_processed, dups_removed, elapsed_sec)
        except Exception as e:
            msg = f"Error: {str(e)}"
            self.progress.emit(0, msg)
            # For consistency, pass placeholders if an error occurs:
            self.finished.emit({}, [msg], 0, 0, 0.0)


class PreprocessingTab(QWidget):
    """
    Allows the user to:
      1. Pick a raw dataset folder
      2. Pick an output folder
      3. (Optional) Remove duplicates
      4. (Optional) Apply histogram eq
      5. (Optional) Filter mode (blur/sharpen)
      6. Resize images with chosen interpolation
      7. Produce a 'cleaned' version of the data
      8. Advanced Logging
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QVBoxLayout(self)

        doc_label = QLabel(
            "<h3>Dataset Preprocessing</h3>"
            "Steps:<br>"
            "1) Copy the raw dataset (Training & Testing).<br>"
            "2) (Optional) Remove duplicates.<br>"
            "3) (Optional) Blur/Sharpen filters.<br>"
            "4) (Optional) Histogram Equalization (CLAHE).<br>"
            "5) Resize images with chosen interpolation.<br>"
            "Your original data remains untouched."
        )
        doc_label.setWordWrap(True)
        main_layout.addWidget(doc_label)

        # Folder selection group
        folder_box = QGroupBox("Data Folders")
        folder_layout = QFormLayout()

        # Source folder
        self.root_edit = QLineEdit("data/archive")
        self.root_edit.setToolTip("Folder containing 'Training' & 'Testing' subfolders.")
        btn_src_browse = QPushButton("Browse...")
        btn_src_browse.clicked.connect(self.on_browse_src)

        h_src = QHBoxLayout()
        h_src.addWidget(self.root_edit)
        h_src.addWidget(btn_src_browse)
        folder_layout.addRow("Source Folder:", h_src)

        # Output folder
        self.out_edit = QLineEdit("cleaned_data")
        self.out_edit.setToolTip("Destination folder where cleaned data is placed.")
        btn_out_browse = QPushButton("Browse...")
        btn_out_browse.clicked.connect(self.on_browse_out)

        h_out = QHBoxLayout()
        h_out.addWidget(self.out_edit)
        h_out.addWidget(btn_out_browse)
        folder_layout.addRow("Output Folder:", h_out)

        folder_box.setLayout(folder_layout)
        main_layout.addWidget(folder_box)

        # BASIC SETTINGS group
        basic_box = QGroupBox("Basic Settings")
        basic_form = QFormLayout()

        self.resize_combo = QLineEdit("128")  # let user type or pick dimension
        basic_form.addRow("Resize (px):", self.resize_combo)

        # Interpolation mode
        from PyQt5.QtWidgets import QComboBox
        self.interp_combo = QComboBox()
        self.interp_combo.addItems(["BILINEAR", "BICUBIC", "LANCZOS"])
        basic_form.addRow("Interpolation:", self.interp_combo)

        basic_box.setLayout(basic_form)
        main_layout.addWidget(basic_box)

        # ADVANCED SETTINGS group
        adv_box = QGroupBox("Advanced")
        adv_form = QFormLayout()

        self.dup_chk = QCheckBox("Delete duplicates?")
        self.eq_chk = QCheckBox("Apply Histogram Equalization?")

        # Filter combo
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["none", "blur", "sharpen"])
        self.filter_combo.setToolTip("Apply blur or sharpen filter to each image")

        adv_form.addRow(self.dup_chk)
        adv_form.addRow(self.eq_chk)
        adv_form.addRow("Filter Mode:", self.filter_combo)

        # Advanced logging checkbox
        self.adv_log_chk = QCheckBox("Enable Advanced Logging")
        adv_form.addRow(self.adv_log_chk)

        adv_box.setLayout(adv_form)
        main_layout.addWidget(adv_box)

        # Action + Progress
        self.btn_preprocess = QPushButton("Start Preprocessing")
        self.btn_preprocess.setToolTip("Copies dataset, optional eq, filtering, duplicates, etc.")
        self.btn_preprocess.clicked.connect(self.on_start_preprocessing)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        main_layout.addWidget(self.btn_preprocess)
        main_layout.addWidget(self.progress_bar)

        # Log area
        self.log_box = QTextEdit()
        self.log_box.setToolTip("Distribution info & any errors appear here.")
        self.log_box.setReadOnly(True)
        main_layout.addWidget(self.log_box)

        self.setLayout(main_layout)
        self.worker = None

    def on_browse_src(self):
        dlg = QFileDialog(self)
        dlg.setFileMode(QFileDialog.Directory)
        if dlg.exec_():
            folder = dlg.selectedFiles()[0]
            self.root_edit.setText(folder)

    def on_browse_out(self):
        dlg = QFileDialog(self)
        dlg.setFileMode(QFileDialog.Directory)
        if dlg.exec_():
            folder = dlg.selectedFiles()[0]
            self.out_edit.setText(folder)

    def on_start_preprocessing(self):
        root_folder = self.root_edit.text().strip()
        if not os.path.isdir(root_folder):
            QMessageBox.warning(self, "Invalid Folder", f"'{root_folder}' is not a valid directory.")
            return

        output_folder = self.out_edit.text().strip()
        if not output_folder:
            QMessageBox.warning(self, "No Output Folder", "Please specify an output folder.")
            return

        try:
            resize_val = int(self.resize_combo.text().strip())
        except:
            QMessageBox.warning(self, "Invalid Resize", "Resize must be an integer.")
            return

        delete_dup = self.dup_chk.isChecked()
        apply_eq = self.eq_chk.isChecked()
        filter_mode = self.filter_combo.currentText()
        interp_mode = self.interp_combo.currentText()

        self.log_box.clear()
        self.progress_bar.setValue(0)

        self.worker = PreprocessingWorker(
            root_folder=root_folder,
            output_folder=output_folder,
            resize_val=resize_val,
            delete_duplicates=delete_dup,
            apply_hist_eq=apply_eq,
            filter_mode=filter_mode,
            interpolation_mode=interp_mode
        )
        self.worker.progress.connect(self.on_worker_progress)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_worker_progress(self, val, msg):
        self.progress_bar.setValue(val)
        self.progress_bar.setFormat(msg)

    def on_worker_finished(
        self,
        dist_info: dict,
        error_log: list,
        total_processed: int,
        dups_removed: int,
        elapsed_sec: float
    ):
        """
        dist_info: final distribution of images
        error_log: any errors encountered
        total_processed: how many images were processed
        dups_removed: how many duplicates removed
        elapsed_sec: total time spent
        """
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Finished")

        # Basic logs always
        self.log_box.append(f"<b>Total images processed:</b> {total_processed}")

        if dups_removed > 0:
            # "in one phrase" => you can keep it short
            self.log_box.append(f"<b>Duplicates removed in one go:</b> {dups_removed}")

        # Display distribution
        if dist_info:
            self.log_box.append("<br><b>Final distribution in output folder:</b>")
            for subset, sub_dict in dist_info.items():
                sub_total = sum(sub_dict.values())
                self.log_box.append(f"[{subset}] => {sub_total} images total")
                for cls, cnt in sub_dict.items():
                    self.log_box.append(f"   {cls}: {cnt}")
                self.log_box.append("")

        # Show any errors
        if error_log:
            self.log_box.append("<b>Errors during processing:</b>")
            for e in error_log:
                self.log_box.append(e)

        self.log_box.append(f"<i>Elapsed time: {elapsed_sec:.2f} seconds</i>\n")

        # If advanced logging is enabled, add extra explanation
        if self.adv_log_chk.isChecked():
            self.log_box.append("<hr><b>Advanced Logging Summary:</b>")
            self.log_box.append("1) We first copied the original dataset structure into the output folder.")
            if self.dup_chk.isChecked():
                self.log_box.append(
                    "2) We removed duplicates by comparing file hashes, which prevents repeated samples "
                    "from biasing our dataset."
                )
            if self.eq_chk.isChecked():
                self.log_box.append(
                    "3) We applied CLAHE-based Histogram Equalization to enhance contrast for each image."
                )
            fmode = self.filter_combo.currentText()
            if fmode != "none":
                self.log_box.append(
                    f"4) We applied the '{fmode}' filter to each image for potential smoothing or sharpening."
                )
            rz = self.resize_combo.text().strip()
            interp = self.interp_combo.currentText()
            self.log_box.append(
                f"5) Lastly, we resized all images to {rz}x{rz} using {interp} interpolation "
                "so that training uses a consistent input size."
            )
            self.log_box.append("Processing is complete. The dataset is now 'cleaned' and ready.\n")
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

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MplCanvas(FigureCanvas):
    """
    A reusable Matplotlib canvas for embedding in PyQt.
    Used by any tab that needs side-by-side images or charts.
    """
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(fig)
        self.setParent(parent)
# File: logic/advanced_eda.py

import os
import random
import numpy as np
from PIL import Image
import cv2  # For CLAHE if using histogram_equalize
from PIL import ImageOps

def gather_random_images(root_folder, subfolders, max_samples=200):
    """
    Collect up to 'max_samples' images from each specified subfolder,
    including top-level (directly under subset) AND sub-subfolders.

    Example: subfolders = ["Training", "Testing"].
    We do:
       - Look in data/archive/Training -> gather images in that folder
       - Also gather images in subfolders (e.g. "ClassA", "ClassB") if they exist
       - Do the same for data/archive/Testing
    Return a random subset of these images up to 'max_samples' total.
    """
    imgs = []
    for subset in subfolders:
        subset_path = os.path.join(root_folder, subset)
        if not os.path.isdir(subset_path):
            continue

        # 1) Gather images directly in subset_path
        top_level_images = [
            os.path.join(subset_path, f)
            for f in os.listdir(subset_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        imgs.extend(top_level_images)

        # 2) Also gather images in subfolders (like "ClassA")
        for d in os.listdir(subset_path):
            dpath = os.path.join(subset_path, d)
            if os.path.isdir(dpath):
                sub_imgs = [
                    os.path.join(dpath, f)
                    for f in os.listdir(dpath)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
                imgs.extend(sub_imgs)

    random.shuffle(imgs)
    return imgs[:max_samples]

def gather_all_images_anywhere(root_folder, subfolders):
    """
    Similar to gather_random_images, but returns ALL images (no random limit).
    Great for building a 'gallery' of all images in Training/Testing.

    NOTE: Potentially large if your dataset is big. Use with caution or impose a limit.
    """
    imgs = []
    for subset in subfolders:
        subset_path = os.path.join(root_folder, subset)
        if not os.path.isdir(subset_path):
            continue

        # top-level images
        for f in os.listdir(subset_path):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                imgs.append(os.path.join(subset_path, f))

        # subfolders
        for d in os.listdir(subset_path):
            dpath = os.path.join(subset_path, d)
            if os.path.isdir(dpath):
                for fname in os.listdir(dpath):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        imgs.append(os.path.join(dpath, fname))
    return imgs


def collect_resolutions(image_paths):
    """
    Return lists of widths, heights, plus a count of read errors.
    """
    widths = []
    heights = []
    error_count = 0
    for path in image_paths:
        try:
            with Image.open(path) as im:
                w, h = im.size
                widths.append(w)
                heights.append(h)
        except:
            error_count += 1
    return widths, heights, error_count

def detect_poor_quality(image_paths):
    """
    Return a list of poor-quality image paths (basic checks):
      - extremely small dimension (<50px)
      - very low variance (<5.0)
    """
    poor_quality = []
    for path in image_paths:
        try:
            with Image.open(path) as im:
                w, h = im.size
                if w < 50 or h < 50:
                    poor_quality.append(path)
                    continue
                arr = np.array(im.convert('L'), dtype=np.float32)
                if arr.size > 0 and arr.std() < 5.0:
                    poor_quality.append(path)
        except:
            pass
    return poor_quality

def histogram_equalize(pil_img):
    """
    Now uses Adaptive Histogram Equalization (CLAHE) instead of global eq.
    """
    arr = np.array(pil_img, dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq_arr = clahe.apply(arr)
    out_pil = Image.fromarray(eq_arr)
    return out_pil

def detect_extended_outliers(
    image_paths,
    brightness_low=15.0,
    brightness_high=240.0,
    dimension_threshold=50,
    var_threshold=5.0
):
    """
    Extended outlier detection with dimension, variance, brightness checks.
    """
    extended_outliers = []
    for path in image_paths:
        try:
            with Image.open(path) as im:
                w, h = im.size
                arr = np.array(im.convert('L'), dtype=np.float32)
                if arr.size == 0:
                    extended_outliers.append(path)
                    continue

                mean_val = arr.mean()
                std_val = arr.std()

                # dimension check
                if w < dimension_threshold or h < dimension_threshold:
                    extended_outliers.append(path)
                    continue

                # variance check
                if std_val < var_threshold:
                    extended_outliers.append(path)
                    continue

                # brightness check
                if mean_val < brightness_low or mean_val > brightness_high:
                    extended_outliers.append(path)
                    continue

        except:
            extended_outliers.append(path)
    return extended_outliers

def gather_subset_resolutions(root_folder, subsets):
    """
    For each subset in subsets (e.g. ["Training","Testing"]),
    gather all images and return (subset_to_widths, subset_to_heights).
    """
    subset_to_widths = {}
    subset_to_heights = {}

    for sub in subsets:
        sub_path = os.path.join(root_folder, sub)
        if not os.path.isdir(sub_path):
            subset_to_widths[sub] = []
            subset_to_heights[sub] = []
            continue

        w_list = []
        h_list = []

        # gather images in sub_path directly
        for fname in os.listdir(sub_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                fullp = os.path.join(sub_path, fname)
                try:
                    with Image.open(fullp) as im:
                        w, h = im.size
                        w_list.append(w)
                        h_list.append(h)
                except:
                    pass

        # gather images in subfolders
        class_list = [
            d for d in os.listdir(sub_path)
            if os.path.isdir(os.path.join(sub_path, d))
        ]
        for c in class_list:
            cpath = os.path.join(sub_path, c)
            for fname in os.listdir(cpath):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    fullp = os.path.join(cpath, fname)
                    try:
                        with Image.open(fullp) as im:
                            w, h = im.size
                            w_list.append(w)
                            h_list.append(h)
                    except:
                        pass

        subset_to_widths[sub] = w_list
        subset_to_heights[sub] = h_list

    return subset_to_widths, subset_to_heights
import os
import math
import random
import shutil
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from logic.duplicates import rotate_image_pil


def preview_augmentation(
    pil_image: Image.Image,
    rotate_deg: float = 15,
    do_rotate: bool = True,
    do_brightness_contrast: bool = True,
    do_gamma: bool = True,
    do_minmax: bool = False,
    do_flip: bool = False,
    do_perspective: bool = False,
    do_crop: bool = False,
    brightness_range=(0.85, 1.15),
    contrast_range=(0.85, 1.15),
    gamma_range=(0.85, 1.15)
):
    """
    Preview a single random augmentation on one image
    (used by the Preview tab).
    """
    base_resize = T.Resize((224, 224), interpolation=InterpolationMode.BILINEAR)
    base_t = T.ToTensor()(base_resize(pil_image))
    temp_pil = T.ToPILImage()(base_t).convert('L')

    if do_rotate:
        angle = random.uniform(-rotate_deg, rotate_deg)
        temp_pil = rotate_image_pil(temp_pil, angle=angle, resample=Image.BILINEAR)

    if do_flip:
        if random.random() < 0.5:
            temp_pil = temp_pil.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            temp_pil = temp_pil.transpose(Image.FLIP_TOP_BOTTOM)

    if do_perspective:
        distort = random.uniform(0.1, 0.4)
        perspective = T.RandomPerspective(distortion_scale=distort, p=1.0)
        temp_pil = perspective(temp_pil)

    if do_crop:
        cropper = T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0))
        temp_pil = cropper(temp_pil)

    if do_brightness_contrast:
        b_min, b_max = brightness_range
        c_min, c_max = contrast_range
        color_jit = T.ColorJitter(
            brightness=(b_min, b_max),
            contrast=(c_min, c_max)
        )
        temp_pil = color_jit(temp_pil)

    if do_gamma:
        g_min, g_max = gamma_range
        g = random.uniform(g_min, g_max)
        temp_pil = F.adjust_gamma(temp_pil, gamma=g, gain=1)

    out_tens = T.ToTensor()(temp_pil)
    if do_minmax:
        mn, mx = out_tens.min(), out_tens.max()
        if (mx - mn) > 1e-5:
            out_tens = (out_tens - mn) / (mx - mn)

    return out_tens


def bulk_augment_dataset(
    src_root: str,
    dst_root: str,
    final_target: int,
    rotate_deg=15,
    do_rotate=True,
    do_brightness_contrast=True,
    do_gamma=True,
    do_flip=False,
    do_perspective=False,
    do_crop=False,
    brightness_range=(0.85, 1.15),
    contrast_range=(0.85, 1.15),
    gamma_range=(0.85, 1.15),
    progress_callback=None
):
    """
    Creates an augmented dataset in `dst_root`, ensuring for each
    class (under the "Training" or other subsets) we have at least `final_target` images.

    Steps:
      1) Copy all original images from src_root -> dst_root.
      2) Count how many each class has. If already >= final_target, skip augmentation.
      3) If < final_target, do a round-robin approach using the source images.
      4) Stop once you have exactly `final_target` for that class.
    """
    if not os.path.isdir(src_root):
        raise ValueError(f"Source folder '{src_root}' does not exist.")

    error_log = []
    total_new_images = 0

    # Gather all subfolders (commonly "Training", "Testing", etc.)
    subsets = []
    for name in os.listdir(src_root):
        s_path = os.path.join(src_root, name)
        if os.path.isdir(s_path):
            subsets.append(name)

    # Prepare lists for each subset & class
    class_info = {}
    total_ops = 0

    # Create the same subsets in dst_root
    for subset in subsets:
        src_subset = os.path.join(src_root, subset)
        dst_subset = os.path.join(dst_root, subset)
        os.makedirs(dst_subset, exist_ok=True)

        classes = [
            d for d in os.listdir(src_subset)
            if os.path.isdir(os.path.join(src_subset, d))
        ]
        for cls_ in classes:
            src_cls_dir = os.path.join(src_subset, cls_)
            dst_cls_dir = os.path.join(dst_subset, cls_)
            os.makedirs(dst_cls_dir, exist_ok=True)

            # Gather all images in src
            src_images = [
                f for f in os.listdir(src_cls_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            base_count = len(src_images)

            # We'll copy them all first => 'base_count' copy ops
            # Then see how many we need to augment
            needed = max(0, final_target - base_count)

            class_key = f"{subset}/{cls_}"
            class_info[class_key] = {
                "src_images": src_images,
                "base_count": base_count,
                "needed": needed
            }
            total_ops += base_count  # each original is one copy op
            total_ops += needed      # plus the needed new images

    if total_ops == 0:
        if progress_callback:
            progress_callback(100, "No images found or no classes in source.")
        return 0, []

    # (i) Copy all originals
    done_ops = 0

    def update_progress(msg):
        nonlocal done_ops, total_ops, progress_callback
        done_ops += 1
        pct = int((done_ops / total_ops) * 100)
        if progress_callback:
            progress_callback(pct, msg)

    for class_key, info in class_info.items():
        subset_name, cls_name = class_key.split("/", 1)
        src_subset = os.path.join(src_root, subset_name, cls_name)
        dst_subset = os.path.join(dst_root, subset_name, cls_name)

        for img_name in info["src_images"]:
            src_path = os.path.join(src_subset, img_name)
            dst_path = os.path.join(dst_subset, img_name)
            try:
                shutil.copy2(src_path, dst_path)
                update_progress(f"Copy {class_key}")
            except Exception as e:
                err_msg = f"[CopyError] {src_path} => {str(e)}"
                error_log.append(err_msg)

    # (ii) Now do round-robin augmentation where needed
    def build_augment_func():
        def augment_func(pil_img: Image.Image):
            temp = pil_img
            if do_rotate:
                angle = random.uniform(-rotate_deg, rotate_deg)
                temp = rotate_image_pil(temp, angle=angle, resample=Image.BILINEAR)

            if do_flip:
                if random.random() < 0.5:
                    temp = temp.transpose(Image.FLIP_LEFT_RIGHT)
                else:
                    temp = temp.transpose(Image.FLIP_TOP_BOTTOM)

            if do_perspective:
                distort = random.uniform(0.1, 0.4)
                persp = T.RandomPerspective(distortion_scale=distort, p=1.0)
                temp = persp(temp)

            if do_crop:
                w, h = temp.size
                cropper = T.RandomResizedCrop(size=(h, w), scale=(0.8, 1.0))
                temp = cropper(temp)

            if do_brightness_contrast:
                b_min, b_max = brightness_range
                c_min, c_max = contrast_range
                color_jit = T.ColorJitter(
                    brightness=(b_min, b_max),
                    contrast=(c_min, c_max)
                )
                temp = color_jit(temp)

            if do_gamma:
                g_min, g_max = gamma_range
                g_val = random.uniform(g_min, g_max)
                temp = F.adjust_gamma(temp, gamma=g_val, gain=1)

            return temp
        return augment_func

    augment_func = build_augment_func()

    for class_key, info in class_info.items():
        needed = info["needed"]
        if needed <= 0:
            continue

        subset_name, cls_name = class_key.split("/", 1)
        dst_subset = os.path.join(dst_root, subset_name, cls_name)
        src_subset = os.path.join(src_root, subset_name, cls_name)
        base_count = info["base_count"]
        src_images = info["src_images"]

        idx = 0
        for i in range(needed):
            src_name = src_images[idx]
            idx = (idx + 1) % base_count  # loop around if needed
            src_path = os.path.join(src_subset, src_name)
            try:
                with Image.open(src_path).convert('L') as pil_img:
                    aug_pil = augment_func(pil_img)
                    base_name, ext = os.path.splitext(src_name)
                    out_name = f"{base_name}_aug{i}{ext}"
                    out_path = os.path.join(dst_subset, out_name)
                    aug_pil.save(out_path)
                    total_new_images += 1
                    update_progress(f"Augment {class_key}")
            except Exception as e:
                err_msg = f"[AugError] {src_path} => {str(e)}"
                error_log.append(err_msg)

    if progress_callback:
        progress_callback(100, "Augmentation done.")
    return total_new_images, error_log
# File: logic/auto_experiment.py

import itertools
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from src.experiment import Experiment


class AutoSearchWorker(QThread):
    """
    A QThread that runs a hyperparameter search on the specified HPC ranges.
    After finishing, it exposes `best_params`, `best_acc`, and a full log of combos in `searcher`.
    """

    progress = pyqtSignal(int)   # emits integer for progress [0..100]
    log_msg = pyqtSignal(str)    # emits str messages
    finished = pyqtSignal()      # emitted when search is done

    def __init__(self, parent=None):
        super().__init__(parent)
        self.searcher = None  # We store HPC results here

        # The user-provided HPC space
        self.data_root = "cleaned_data"
        self.resize_list = [128]
        self.disable_augment_list = [False]
        self.adv_aug_list = [False]
        self.dropout_list = [0.0]
        self.weight_decay_list = [0.0]
        self.earlystop_list = [0]
        self.class_weight_list = [False]
        self.batch_size = 16
        self.lr = 0.001
        self.epochs = 5
        self.num_classes = 2
        self.device = "cpu"

    def setup_search(
        self,
        data_root,
        resize_list,
        disable_augment_list,
        adv_aug_list,
        dropout_list,
        weight_decay_list,
        earlystop_list,
        class_weight_list,
        batch_size,
        lr,
        epochs,
        num_classes,
        device="cpu"
    ):
        """
        Called by the UI to pass in HPC settings. We'll store them, so run() can do the search.
        """
        self.data_root = data_root
        self.resize_list = resize_list
        self.disable_augment_list = disable_augment_list
        self.adv_aug_list = adv_aug_list
        self.dropout_list = dropout_list
        self.weight_decay_list = weight_decay_list
        self.earlystop_list = earlystop_list
        self.class_weight_list = class_weight_list
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.num_classes = num_classes
        self.device = device

        # Prepare a simple container to store HPC results
        self.searcher = HPCResults()

    def run(self):
        """
        Execute a brute-force HPC search, calling a mini-train method for each combo.
        We store the best combo by val_acc in self.searcher.best_params and best_acc.
        """
        # Build the HPC combos in Pythonic style
        combos = list(itertools.product(
            self.resize_list,
            self.disable_augment_list,
            self.adv_aug_list,
            self.dropout_list,
            self.weight_decay_list,
            self.earlystop_list,
            self.class_weight_list
        ))
        total = len(combos)
        if total == 0:
            self.log_msg.emit("[AutoSearchWorker] No HPC combos specified!")
            self.finished.emit()
            return

        self.log_msg.emit(f"[AutoSearchWorker] Searching over {total} combos...")

        best_acc = -1.0
        best_params_str = ""
        for idx, combo in enumerate(combos):
            resize_val = combo[0]
            disable_aug = combo[1]
            adv_aug = combo[2]
            dropout = combo[3]
            w_decay = combo[4]
            earlystop = combo[5]
            cw = combo[6]

            # Create an Experiment with these HPC
            # For demonstration, we do a short training or direct skip
            # We'll do a small 'simulate_train' or a real partial train

            val_acc = self._simulate_train(
                resize_val=resize_val,
                disable_aug=disable_aug,
                adv_aug=adv_aug,
                dropout=dropout,
                weight_decay=w_decay,
                earlystop=earlystop,
                class_weight=cw
            )

            # Build param string
            combo_str = (
                f"resize={resize_val}, disable_aug={disable_aug}, adv_aug={adv_aug}, "
                f"dropout={dropout}, w_decay={w_decay}, earlystop={earlystop}, cw={cw}"
            )
            # store in HPCResults
            self.searcher.add_result(combo_str, val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                best_params_str = combo_str

            # progress
            pct = int(100*(idx+1)/total)
            self.progress.emit(pct)
            self.log_msg.emit(f"[{idx+1}/{total}] HPC combo => {combo_str}, val_acc={val_acc:.2f}")

        self.searcher.best_acc = best_acc
        self.searcher.best_params = best_params_str

        self.log_msg.emit(f"[AutoSearchWorker] Done! Best Acc={best_acc:.2f} => {best_params_str}")
        self.finished.emit()

    def _simulate_train(
        self,
        resize_val,
        disable_aug,
        adv_aug,
        dropout,
        weight_decay,
        earlystop,
        class_weight
    ):
        """
        A quick approach: either do a *short real training*, or in this example,
        we'll just generate a random number to simulate 'val_acc'.
        In your real code, you might build an `Experiment` and call run_single_training.
        """
        import random
        # If you want to do partial real training, you might do:
        # exp = Experiment(
        #     data_root=self.data_root,
        #     model_variant="binary" if self.num_classes==2 else "multi",
        #     input_size=resize_val,
        #     batch_size=self.batch_size,
        #     lr=self.lr,
        #     epochs=1,  # maybe fewer epochs to speed HPC
        #     disable_augment=disable_aug,
        #     adv_aug=adv_aug,
        #     dropout=dropout,
        #     weight_decay=weight_decay,
        #     earlystop_patience=earlystop,
        #     use_class_weight=class_weight,
        #     device=self.device
        # )
        # exp.load_data()
        # best_acc, _, _ = exp.run_single_training(epochs=1)
        # return best_acc

        # For demonstration, just return a random val_acc
        return random.uniform(60, 98)


class HPCResults:
    """
    Holds the best HPC param string & val_acc, and a list of all combos.
    """
    def __init__(self):
        self.best_params = ""
        self.best_acc = 0.0
        self.results = []  # list of (param_str, val_acc)

    def add_result(self, param_str, val_acc):
        self.results.append((param_str, val_acc))
import os
from PIL import Image

def count_images_in_classes(folder: str) -> dict:
    """
    Return a dict {class_name: number_of_images}
    for each subfolder in 'folder'.
    """
    if not os.path.isdir(folder):
        return {}
    result = {}
    for cls in os.listdir(folder):
        cpath = os.path.join(folder, cls)
        if os.path.isdir(cpath):
            n = sum(
                f.lower().endswith(('.png','.jpg','.jpeg'))
                for f in os.listdir(cpath)
            )
            result[cls] = n
    return result

def find_image_in_subfolders(root_dir: str, image_name: str) -> str:
    """
    Search each subfolder of root_dir for image_name.
    Return the full path if found, else None.
    """
    if not os.path.isdir(root_dir):
        return None

    for subdir in os.listdir(root_dir):
        cpath = os.path.join(root_dir, subdir)
        if os.path.isdir(cpath):
            candidate = os.path.join(cpath, image_name)
            if os.path.isfile(candidate):
                return candidate
    return None

def load_image_as_grayscale(path: str) -> Image.Image:
    """
    Open an image at path as grayscale PIL image. 
    Returns None if fails.
    """
    try:
        return Image.open(path).convert('L')
    except:
        return None
import os
from PIL import Image

def count_images_in_classes(folder: str) -> dict:
    """
    Return a dict {class_name: number_of_images} for each subfolder in 'folder'.
    """
    if not os.path.isdir(folder):
        return {}
    result = {}
    for cls in os.listdir(folder):
        cpath = os.path.join(folder, cls)
        if os.path.isdir(cpath):
            n = sum(
                f.lower().endswith(('.png', '.jpg', '.jpeg'))
                for f in os.listdir(cpath)
            )
            result[cls] = n
    return result


def find_image_in_subfolders(root_dir: str, image_name: str) -> str:
    """
    Search each subfolder of root_dir for image_name.
    Return the full path if found, else None.
    """
    if not os.path.isdir(root_dir):
        return None

    for subdir in os.listdir(root_dir):
        cpath = os.path.join(root_dir, subdir)
        if os.path.isdir(cpath):
            candidate = os.path.join(cpath, image_name)
            if os.path.isfile(candidate):
                return candidate
    return None


def load_image_as_grayscale(path: str) -> Image.Image:
    """
    Open an image at path as grayscale PIL image.
    Returns None if fails.
    """
    try:
        return Image.open(path).convert('L')
    except:
        return None
import os
import hashlib
from PIL import Image


def rotate_image_pil(pil_img, angle=0, resample=Image.BILINEAR):
    if angle == 0:
        return pil_img
    return pil_img.rotate(angle, resample=resample)


def compute_file_hash(file_path: str) -> str:
    """Compute MD5 hash of a file's content."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def remove_duplicates_in_folder(folder_path: str) -> list:
    """
    OLD function: remove duplicates only if they are in the SAME folder.
    (We keep it in case other code calls it.)
    """
    file_hashes = {}
    duplicates = []
    if not os.path.isdir(folder_path):
        return duplicates

    for fname in os.listdir(folder_path):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(folder_path, fname)
            file_hash = compute_file_hash(full_path)
            if file_hash in file_hashes:
                duplicates.append(full_path)
            else:
                file_hashes[file_hash] = full_path
    return duplicates


def remove_duplicates_across_dataset(root_dir: str) -> list:
    """
    OLD function: check duplicates for each immediate subfolder in root_dir.
    But it does NOT go deeper into sub-subfolders.
    """
    all_duplicates = []
    if not os.path.isdir(root_dir):
        return all_duplicates

    for subdir in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, subdir)
        if os.path.isdir(class_dir):
            dups = remove_duplicates_in_folder(class_dir)
            all_duplicates.extend(dups)

    return all_duplicates


def remove_duplicates_recursively(root_dir: str) -> list:
    """
    NEW function: Recursively find duplicates in ALL subfolders under `root_dir`.
    We build one dictionary keyed by MD5 hash so duplicates are recognized
    across sub-subfolders as well.
    Returns a list of file paths that are duplicates (excluding the first occurrence).
    """
    if not os.path.isdir(root_dir):
        return []

    # 1) Collect all .png/.jpg/.jpeg under root_dir (any depth).
    all_files = []
    for dirpath, dirs, files in os.walk(root_dir):
        for fname in files:
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(dirpath, fname)
                all_files.append(full_path)

    # 2) Compute hashes and detect duplicates
    hash_dict = {}
    duplicates = []
    for fpath in all_files:
        hval = compute_file_hash(fpath)
        if hval in hash_dict:
            duplicates.append(fpath)
        else:
            hash_dict[hval] = fpath

    return duplicates
import os
import csv
import torch
import torch.nn.functional as F
from PIL import Image

from src.models import get_model
from src.transforms_config import create_transform

class InferenceModel:
    """
    Enhanced helper class for loading a trained model checkpoint
    and performing single-image or batch (folder) prediction.
    For 'double' variant: first we classify (NoTumor vs Tumor).
       If Tumor, we do the 3-class head to differentiate
       (Pituitary / Meningioma / Glioma).
    """

    def __init__(
        self,
        checkpoint_path,
        model_variant="double",
        input_size=128,
        normalization="zscore",
        device="cpu",
        dropout=0.0
    ):
        self.device = torch.device(device)

        # Build the chosen model
        self.model, _ = get_model(
            model_variant=model_variant,
            input_size=input_size,
            dropout=dropout
        )
        self.model.to(self.device)
        self.model.eval()

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model_variant = model_variant

        # Build transform (same logic as training)
        self.transform = create_transform(
            image_size=(input_size, input_size),
            augment=False
        )

    def predict(self, pil_img):
        """
        Single-image prediction.
        For 'double': we do binary => if "tumor",
                      then do threeclass => classify Pituitary / Meningioma / Glioma.

        Returns a dict containing:
          {
            "prediction": "NoTumor" or "Pituitary"/"Meningioma"/"Glioma",
            "binary_probs": [float, float],        # raw probabilities for no/tumor
            "binary_perc":  [str, str],            # e.g. ["85.2%", "14.8%"]
            "multi_probs":  [floats or None],      # 3-class probabilities if tumor
            "multi_perc":   [str or None],         # 3-class percentages if tumor
          }

        If not double, returns a simpler structure with:
          {
            "class_index": <int>,
            "probabilities": [float...]
          }
        """
        x = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.model_variant == "double":
                # 1) Binary (NoTumor vs Tumor)
                out_bin = self.model(x, mode='binary')
                bin_probs = F.softmax(out_bin, dim=1)[0]  # shape [2]
                bin_probs_float = [float(p) for p in bin_probs]
                bin_probs_percent = [f"{p*100:.1f}%" for p in bin_probs_float]

                # Check if tumor (binary index=1)
                if bin_probs[1] > bin_probs[0]:
                    # 2) Three-class => Pituitary / Meningioma / Glioma
                    out_3c = self.model(x, mode='threeclass')
                    multi_probs = F.softmax(out_3c, dim=1)[0]  # shape [3]
                    multi_probs_float = [float(p) for p in multi_probs]
                    multi_probs_percent = [f"{p*100:.1f}%" for p in multi_probs_float]

                    subtype_idx = multi_probs.argmax().item()
                    # Tumor labels
                    tumor_labels = ["Pituitary", "Meningioma", "Glioma"]
                    subtype_label = tumor_labels[subtype_idx]

                    return {
                        "prediction": subtype_label,
                        "binary_probs": bin_probs_float,
                        "binary_perc": bin_probs_percent,
                        "multi_probs": multi_probs_float,
                        "multi_perc": multi_probs_percent
                    }
                else:
                    # NoTumor
                    return {
                        "prediction": "NoTumor",
                        "binary_probs": bin_probs_float,
                        "binary_perc": bin_probs_percent,
                        "multi_probs": None,
                        "multi_perc": None
                    }
            else:
                # Single-head (binary, threeclass, or multi)
                out = self.model(x)
                probs = F.softmax(out, dim=1)[0]
                probs_float = [float(p) for p in probs]
                class_idx = probs.argmax().item()
                return {
                    "class_index": class_idx,
                    "probabilities": probs_float
                }

    def predict_batch(
        self,
        image_paths=None,
        folder_path=None,
        export_csv=None
    ):
        """
        Batch inference on multiple images.

        Each returned dict has the same keys as `predict()`.
        - If double => "prediction", "binary_probs", "multi_probs" etc.
        - Otherwise => "class_index", "probabilities".
        """
        # Gather the file list from either `image_paths` or `folder_path`.
        file_list = []
        if image_paths:
            file_list = list(image_paths)
        elif folder_path:
            if not os.path.isdir(folder_path):
                raise ValueError(f"Folder not found: {folder_path}")
            all_files = os.listdir(folder_path)
            # pick images only
            for f in all_files:
                ext = f.lower()
                if ext.endswith(".png") or ext.endswith(".jpg") or ext.endswith(".jpeg"):
                    file_list.append(os.path.join(folder_path, f))
        else:
            raise ValueError("Must provide image_paths or folder_path for batch inference.")

        results = []
        for img_path in file_list:
            if not os.path.isfile(img_path):
                # skip invalid
                continue
            try:
                pil_img = Image.open(img_path).convert('L')
            except Exception as e:
                results.append({
                    "file": img_path,
                    "error": f"Could not open: {e}"
                })
                continue

            # Single predict
            pred_dict = self.predict(pil_img)
            pred_dict["file"] = img_path
            results.append(pred_dict)

        # optionally export CSV
        if export_csv:
            self._export_results_csv(results, export_csv)

        return results

    def _predict_internal(self, pil_img):
        """
        Internal helper that just returns the full record:
        For 'double': {prediction, binary_probs, multi_probs}
        For 'binary': {class_index, probabilities}
        For 'multi':  {class_index, probabilities}
        """
        x = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.model_variant == "double":
                out_bin = self.model(x, mode='binary')
                bin_probs = F.softmax(out_bin, dim=1)[0]
                if bin_probs[1] > bin_probs[0]:
                    out_multi = self.model(x, mode='multi')
                    multi_probs = F.softmax(out_multi, dim=1)[0]
                    subtype_idx = multi_probs.argmax().item()
                    return {
                        "prediction": f"tumor_subtype_{subtype_idx}",
                        "binary_probs": bin_probs.tolist(),
                        "multi_probs": multi_probs.tolist()
                    }
                else:
                    return {
                        "prediction": "no_tumor",
                        "binary_probs": bin_probs.tolist(),
                        "multi_probs": None
                    }
            else:
                out = self.model(x)
                probs = F.softmax(out, dim=1)[0]
                pred_idx = probs.argmax().item()
                return {
                    "class_index": int(pred_idx),
                    "probabilities": probs.tolist()
                }

    def _export_results_csv(self, results, csv_path):
        """
        Exports batch inference results to a CSV file.
        For double, we store columns for prediction, binary & multi probabilities, etc.
        """
        fieldnames = [
            "file",
            "prediction",
            "binary_probs", "binary_perc",
            "multi_probs",  "multi_perc",
            "class_index",  "probabilities",
            "error"
        ]
        try:
            with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in results:
                    writer.writerow(row)
        except Exception as e:
            print(f"[WARN] Could not export CSV to {csv_path}: {e}")
# logic/inference_two_step.py

import os
import torch
import torch.nn.functional as F
from PIL import Image

from src.models import get_model
from src.transforms_config import create_transform


class BinaryInferenceModel:
    """
    Single-head HPC model that does binary classification:
      0 => NoTumor
      1 => Tumor
    """
    def __init__(self, checkpoint_path, input_size=128, device="cpu", dropout=0.0):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Binary checkpoint not found: {checkpoint_path}")
        self.device = torch.device(device)

        # Build a "binary" net => get_model("binary", input_size=..., dropout=...)
        self.model, _ = get_model("binary", input_size=input_size, dropout=dropout)
        self.model.to(self.device)
        self.model.eval()

        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)

        self.transform = create_transform(
            image_size=(input_size, input_size),
            augment=False
        )

    def predict(self, pil_img):
        x = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)  # shape [1,2]
            probs = F.softmax(logits, dim=1)[0]
            pred_idx = int(probs.argmax().item())
        return {
            "prediction": pred_idx,  # 0 => no tumor, 1 => tumor
            "prob": probs.cpu().tolist()
        }


class ThreeClassInferenceModel:
    """
    Single-head HPC model that does 3-class:
      0 => Pituitary
      1 => Meningioma
      2 => Glioma
    """
    def __init__(self, checkpoint_path, input_size=128, device="cpu", dropout=0.0):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Threeclass checkpoint not found: {checkpoint_path}")
        self.device = torch.device(device)

        self.model, _ = get_model("threeclass", input_size=input_size, dropout=dropout)
        self.model.to(self.device)
        self.model.eval()

        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)

        self.transform = create_transform(
            image_size=(input_size, input_size),
            augment=False
        )
        self.labels = ["Pituitary", "Meningioma", "Glioma"]

    def predict(self, pil_img):
        x = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)[0]
            idx = int(probs.argmax().item())
        return {
            "prediction_idx": idx,
            "prediction_label": self.labels[idx],
            "prob": probs.cpu().tolist()
        }


class TwoStepDoublePipeline:
    """
    Combines a binary HPC model + a threeclass HPC model
    to replicate a 'double' approach without training a separate net.

    Steps:
      1) Binary => no tumor => done
         If tumor => classify pituitary/meningioma/glioma
    """
    def __init__(
        self,
        binary_checkpoint,
        three_checkpoint,
        input_size=128,
        device="cpu",
        dropout=0.0
    ):
        self.binary_model = BinaryInferenceModel(
            checkpoint_path=binary_checkpoint,
            input_size=input_size,
            device=device,
            dropout=dropout
        )
        self.three_model = ThreeClassInferenceModel(
            checkpoint_path=three_checkpoint,
            input_size=input_size,
            device=device,
            dropout=dropout
        )

    def predict_image(self, pil_img):
        bin_res = self.binary_model.predict(pil_img)  # {prediction=0 or 1, prob=[pNo, pTumor]}
        if bin_res["prediction"] == 0:
            # No Tumor => done
            return {
                "final_prediction": "NoTumor",
                "binary_prob": bin_res["prob"],
                "threeclass_prob": None
            }
        else:
            # Tumor => check 3-class
            three_res = self.three_model.predict(pil_img)
            return {
                "final_prediction": three_res["prediction_label"],
                "binary_prob": bin_res["prob"],
                "threeclass_prob": three_res["prob"]
            }
import os
import time
import shutil
import numpy as np
from PIL import Image, ImageFilter
import cv2  # for CLAHE-based histogram eq

# IMPORT the new remove_duplicates_recursively function
from logic.duplicates import remove_duplicates_recursively

def _adaptive_histogram_equalize(pil_img):
    """
    Applies OpenCV's CLAHE to a PIL grayscale image.
    """
    arr = np.array(pil_img, dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq_arr = clahe.apply(arr)
    return Image.fromarray(eq_arr)

def _get_pil_interpolation(mode_str):
    """
    Convert a string like "BILINEAR"|"BICUBIC"|"LANCZOS"
    into the corresponding PIL interpolation constant.
    Defaults to Image.BILINEAR if unknown.
    """
    mode_str = mode_str.upper()
    if mode_str == "BICUBIC":
        return Image.BICUBIC
    elif mode_str == "LANCZOS":
        return Image.LANCZOS
    else:
        return Image.BILINEAR

def preprocess_entire_dataset(
    root_dir,
    dst_root,
    resize_val=128,
    delete_duplicates=False,
    apply_hist_eq=False,
    filter_mode="none",
    interpolation="BILINEAR",
    progress_fn=None
):
    """
    Copy images from 'root_dir' (with subfolders e.g. Training/Testing) into 'dst_root'.
    Then optionally:
      - remove duplicates (MD5 hash check) [RECURSIVE]
      - apply CLAHE-based histogram eq
      - apply blur/sharpen filters
      - resize with chosen interpolation

    progress_fn: optional callback(current_idx, total_count).

    Returns:
        dist_info (dict): distribution info { subset: {class: count} }
        error_log (list): any errors
        total_processed (int): how many images were processed
        dups_removed (int): how many duplicates removed
        elapsed_sec (float): time taken in seconds
    """
    t0 = time.time()
    error_log = []
    dist_info = {}
    total_processed = 0
    dups_removed = 0

    if not os.path.isdir(root_dir):
        raise ValueError(f"Source '{root_dir}' does not exist.")

    subsets = ["Training", "Testing"]
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root, ignore_errors=True)
    os.makedirs(dst_root, exist_ok=True)

    # Step 1: Copy structure
    for subset in subsets:
        src_subset = os.path.join(root_dir, subset)
        dst_subset = os.path.join(dst_root, subset)
        if os.path.isdir(src_subset):
            os.makedirs(dst_subset, exist_ok=True)

            classes = [
                d for d in os.listdir(src_subset)
                if os.path.isdir(os.path.join(src_subset, d))
            ]
            for cls_ in classes:
                os.makedirs(os.path.join(dst_subset, cls_), exist_ok=True)

            images_copied = 0
            for cls_ in classes:
                src_class_dir = os.path.join(src_subset, cls_)
                dst_class_dir = os.path.join(dst_subset, cls_)
                images = [
                    f for f in os.listdir(src_class_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
                for img_name in images:
                    src_path = os.path.join(src_class_dir, img_name)
                    dst_path = os.path.join(dst_class_dir, img_name)
                    try:
                        shutil.copy2(src_path, dst_path)
                        images_copied += 1
                    except Exception as e:
                        error_log.append(f"CopyError: {src_path} => {e}")

    # Step 2: Remove duplicates *recursively* if requested
    if delete_duplicates:
        duplicates = remove_duplicates_recursively(dst_root)
        dups_removed = len(duplicates)
        for dup_file in duplicates:
            try:
                os.remove(dup_file)
            except OSError:
                pass

    # Step 3: Process (hist eq, filter, resize) for all images
    all_image_paths = []
    for subset in subsets:
        subset_path = os.path.join(dst_root, subset)
        if os.path.isdir(subset_path):
            classes = [
                d for d in os.listdir(subset_path)
                if os.path.isdir(os.path.join(subset_path, d))
            ]
            for cls_ in classes:
                cpath = os.path.join(subset_path, cls_)
                files = [
                    f for f in os.listdir(cpath)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
                for fname in files:
                    fullpath = os.path.join(cpath, fname)
                    all_image_paths.append(fullpath)

    interp_mode = _get_pil_interpolation(interpolation)
    n_images = len(all_image_paths)

    def apply_filter(pil_im, mode):
        if mode == "blur":
            return pil_im.filter(ImageFilter.BLUR)
        elif mode == "sharpen":
            return pil_im.filter(ImageFilter.SHARPEN)
        return pil_im

    idx = 0
    for ipath in all_image_paths:
        idx += 1
        if progress_fn:
            progress_fn(idx, n_images)

        try:
            with Image.open(ipath).convert('L') as im:
                # hist eq if requested
                if apply_hist_eq:
                    im = _adaptive_histogram_equalize(im)
                # blur/sharpen if requested
                im = apply_filter(im, filter_mode.lower())
                # resize
                im = im.resize((resize_val, resize_val), interp_mode)
                # save
                im.save(ipath)
                total_processed += 1

        except Exception as e:
            error_log.append(f"ProcessingError: {ipath} => {e}")

    # Step 4: Summarize distribution
    for subset in subsets:
        subset_path = os.path.join(dst_root, subset)
        if os.path.isdir(subset_path):
            cls_dict = {}
            for cls_ in os.listdir(subset_path):
                cpath = os.path.join(subset_path, cls_)
                if os.path.isdir(cpath):
                    n = sum(
                        f.lower().endswith(('.png', '.jpg', '.jpeg'))
                        for f in os.listdir(cpath)
                    )
                    cls_dict[cls_] = n
            dist_info[subset] = cls_dict

    elapsed_sec = time.time() - t0
    return dist_info, error_log, total_processed, dups_removed, elapsed_sec
import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
import os
from PIL import Image
from torch.utils.data import Dataset

class BrainTumorDataset(Dataset):
    """
    A custom dataset that can handle multiple model variants:
      - binary: 2-class => 0=NoTumor, 1=Tumor (Pituitary/Meningioma/Glioma)
      - threeclass: 3-class => 0=Pituitary, 1=Meningioma, 2=Glioma (no NoTumor)
      - simple (or multi): 4-class => 0=NoTumor, 1=Pituitary, 2=Meningioma, 3=Glioma
      - double: still store (0=NoTumor,1=Pituitary,2=Meningioma,3=Glioma). The
        "double" model will read 0 => no tumor, else => yes tumor in the
        training logic.  (We do not do separate label logic here.)
    """

    def __init__(self, root_dir, subset="Training", transform=None, model_variant="simple"):
        super().__init__()
        self.root_dir = root_dir
        self.subset = subset
        self.transform = transform
        self.model_variant = model_variant.lower()

        self.data_path = os.path.join(self.root_dir, self.subset)
        if not os.path.isdir(self.data_path):
            raise FileNotFoundError(f"{self.data_path} does not exist. Please check your folder names.")

        self.samples = []
        self.classes = []  # We will fill or define differently per variant

        if self.model_variant == "simple" or self.model_variant == "multi":
            # 4-class => subfolders: "NoTumor", "Pituitary", "Meningioma", "Glioma"
            # numeric labels => 0=NoTumor, 1=Pituitary, 2=Meningioma, 3=Glioma
            label_map = {
                "notumor": 0,
                "pituitary": 1,
                "meningioma": 2,
                "glioma": 3
            }
            self.classes = ["NoTumor", "Pituitary", "Meningioma", "Glioma"]
            self._gather_subfolders(label_map)

        elif self.model_variant == "binary":
            # 2-class => NoTumor (0), Tumor (1). We combine pituitary/meningioma/glioma => 1
            self.classes = ["NoTumor", "Tumor"]
            self._gather_binary()

        elif self.model_variant == "threeclass":
                # 3-class => Pituitary=0, Meningioma=1, Glioma=2
                label_map = {
                    "pituitary": 0,
                    "meningioma": 1,
                    "glioma": 2
                }
                # So classes must match label_map order:
                self.classes = ["Pituitary", "Meningioma", "Glioma"]
                self._gather_subfolders(label_map)

        elif self.model_variant == "double":
            # For the double-headed net, we still store them as 0..3:
            #   0=NoTumor, 1=Pituitary, 2=Meningioma, 3=Glioma
            # The training code decides binary vs. threeclass heads later.
            label_map = {
                "notumor": 0,
                "pituitary": 1,
                "meningioma": 2,
                "glioma": 3
            }
            self.classes = ["NoTumor", "Pituitary", "Meningioma", "Glioma"]
            self._gather_subfolders(label_map)

        else:
            raise ValueError(f"Unknown model_variant: {self.model_variant}")

    def _gather_subfolders(self, label_map):
        """ Helper that scans self.data_path for subfolders in label_map. """
        for subf in os.listdir(self.data_path):
            sf_path = os.path.join(self.data_path, subf)
            if not os.path.isdir(sf_path):
                continue
            sf_lower = subf.lower()
            if sf_lower in label_map:
                label_idx = label_map[sf_lower]
                # gather .png/.jpg/.jpeg
                for fname in os.listdir(sf_path):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        full_path = os.path.join(sf_path, fname)
                        self.samples.append((full_path, label_idx))

    def _gather_binary(self):
        """
        In 'binary' variant: 0=NoTumor, else 1=Tumor.
        So if subfolder is exactly 'NoTumor', label=0, otherwise label=1.
        """
        for subf in os.listdir(self.data_path):
            sf_path = os.path.join(self.data_path, subf)
            if not os.path.isdir(sf_path):
                continue

            sf_lower = subf.lower()
            if sf_lower == "notumor":
                label_idx = 0
            else:
                label_idx = 1

            for fname in os.listdir(sf_path):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    full_path = os.path.join(sf_path, fname)
                    self.samples.append((full_path, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def is_empty(self):
        return (len(self.samples) == 0)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import numpy as np
from sklearn.model_selection import KFold

# For ROC/AUC:
try:
    from sklearn.metrics import roc_curve, roc_auc_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

from src.dataset import BrainTumorDataset
from src.transforms_config import create_transform
from src.models import get_model, SimpleCNN

class Experiment:
    """
    An experiment pipeline that can do:
      - single-run train+eval
      - or k-fold cross-validation (folds>1).
    Also accumulates confusion matrices & class metrics for final reporting.

    The key fix is a single `_train_one_epoch()` method so HPC code can do:
      train_loss, train_acc, _ = exp._train_one_epoch()
    """

    def __init__(
        self,
        data_root="cleaned_data",
        model_variant="multi",
        input_size=128,
        batch_size=8,
        lr=1e-3,
        epochs=1,
        disable_augment=False,
        adv_aug=False,
        strong_aug=False,
        dropout=0.0,
        weight_decay=0.0,
        earlystop_patience=0,
        use_class_weight=False,
        device="cpu",
        normalization="none",
        folds=1
    ):
        self.data_root = data_root
        self.model_variant = model_variant
        self.input_size = input_size
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.disable_augment = disable_augment
        self.adv_aug = adv_aug
        self.strong_aug = strong_aug
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.earlystop_patience = earlystop_patience
        self.use_class_weight = use_class_weight
        self.folds = folds
        self.device = torch.device(device)

        self.log_func = print  # default logger

        # Build transformations (no intensity normalization logic here by default):
        self.transform = create_transform(
            image_size=(input_size, input_size),
            augment=(not disable_augment),
            adv_aug=self.adv_aug,
            strong_aug=self.strong_aug
        )

        # Build model (placeholder for final #classes logic below)
        self.model, self.num_classes = get_model(
            model_variant=self.model_variant,
            input_size=self.input_size,
            dropout=self.dropout
        )
        self.model.to(self.device)

        self.optimizer = None
        # single-head criterion
        self.criterion = None
        # double variant => two separate losses
        self.criterion_bin = None
        self.criterion_multi = None

        # final results
        self.final_roc = {"fpr": None, "tpr": None, "auc": None}
        self.final_confmat = None
        self.final_class_metrics = {}

    def set_logger(self, log_callable):
        self.log_func = log_callable

    def load_data(self):
        """
        Loads Training + Testing sets for single-run (folds=1),
        or just the Training set for k-fold (folds>1).
        Then detect how many classes. If 'multi' or 'simple', we rebuild the model to match that #classes.
        """
        if self.folds > 1:
            self.log_func("Loading entire Training folder for k-fold CV.")
            self.train_dataset = BrainTumorDataset(
                root_dir=self.data_root,
                subset="Training",
                transform=self.transform,
                model_variant=self.model_variant
            )
            self.test_dataset = None
        else:
            # Normal single-run
            train_set = BrainTumorDataset(
                root_dir=self.data_root,
                subset="Training",
                transform=self.transform,
                model_variant=self.model_variant
            )
            val_set = BrainTumorDataset(
                root_dir=self.data_root,
                subset="Testing",
                transform=self.transform,
                model_variant=self.model_variant
            )
            self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
            self.val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
            self.train_dataset = train_set
            self.test_dataset = val_set

        # Detect number of classes from the variant
        if self.model_variant == "binary":
            real_num_classes = 2
        elif self.model_variant == "double":
            real_num_classes = None  # double has separate heads
        elif self.model_variant == "threeclass":
            real_num_classes = 3
        else:
            # "multi" or "simple" => read from dataset
            real_num_classes = len(self.train_dataset.classes)

        self.log_func(f"[load_data] Using model_variant='{self.model_variant}'. Detected {real_num_classes} class(es).")

        # If "multi"/"simple", rebuild a single CNN with that # classes
        if self.model_variant in ["multi", "simple"]:
            self.log_func(f"Rebuilding single-head model with {real_num_classes} classes.")
            self.model = SimpleCNN(
                num_classes=real_num_classes,
                input_size=self.input_size,
                dropout=self.dropout
            ).to(self.device)
            self.num_classes = real_num_classes

    def _setup_optimizer_criterion(self):
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        if self.model_variant == "double":
            # define 2 losses
            self.criterion_bin = nn.CrossEntropyLoss()
            self.criterion_multi = nn.CrossEntropyLoss()
            return

        # single-head
        if self.use_class_weight:
            # build class weights from train_dataset freq
            freq = [0]*self.num_classes
            for _, lbl in self.train_dataset.samples:
                freq[lbl] += 1
            total_count = float(sum(freq))
            weights = []
            for c in range(self.num_classes):
                if freq[c] > 0:
                    weights.append(total_count / freq[c])
                else:
                    weights.append(0.0)
            weights_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        else:
            self.criterion = nn.CrossEntropyLoss()

    # ----------------------------------------------------------------
    # HPC partial training calls `_train_one_epoch()`
    # ----------------------------------------------------------------
    def _train_one_epoch(self):
        """
        Single epoch training for HPC partial.
        If double => train both heads in the same pass (binary + multi).
        Otherwise => normal single-head cross-entropy.

        Returns (epoch_loss, epoch_acc, best_batch_acc).
        We'll measure epoch_acc from the *binary classification* if "double".
        """
        self.model.train()
        total_loss = 0.0
        total_correct_bin = 0
        total_samples = 0
        best_batch_acc = 0.0

        if not hasattr(self, 'train_loader'):
            # if HPC tries partial training but we haven't loaded sets, do a fallback
            return 0.0, 0.0, 0.0

        if self.model_variant == "double":
            # Train both heads in a single pass
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Binary => label=0 => no tumor, else => tumor
                bin_labels = (labels != 0).long()

                self.optimizer.zero_grad()

                # forward both heads
                out_bin = self.model(images, mode='binary')
                out_multi = self.model(images, mode='threeclass')

                # binary loss
                loss_bin = self.criterion_bin(out_bin, bin_labels)

                # multi => only for tumor images
                tumor_mask = (labels != 0)
                if torch.any(tumor_mask):
                    multi_labels = labels[tumor_mask] - 1  # shift [1->0,2->1,3->2]
                    multi_out = out_multi[tumor_mask]
                    loss_multi = self.criterion_multi(multi_out, multi_labels)
                else:
                    loss_multi = 0.0

                loss = loss_bin + loss_multi
                loss.backward()
                self.optimizer.step()

                bs = images.size(0)
                total_loss += float(loss) * bs
                total_samples += bs

                # measure binary accuracy
                _, bin_preds = torch.max(out_bin, dim=1)
                batch_correct = (bin_preds == bin_labels).sum().item()
                total_correct_bin += batch_correct

                batch_acc = 100.0 * batch_correct / bs
                if batch_acc > best_batch_acc:
                    best_batch_acc = batch_acc

            epoch_loss = total_loss / total_samples if total_samples else 0.0
            epoch_acc = 100.0 * total_correct_bin / total_samples if total_samples else 0.0
            return epoch_loss, epoch_acc, best_batch_acc

        else:
            # single-head
            total_correct = 0
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                bs = images.size(0)
                total_loss += float(loss) * bs
                total_samples += bs

                preds = logits.argmax(dim=1)
                batch_correct = (preds == labels).sum().item()
                total_correct += batch_correct

                batch_acc = 100.0 * batch_correct / bs
                if batch_acc > best_batch_acc:
                    best_batch_acc = batch_acc

            epoch_loss = total_loss / total_samples if total_samples else 0.0
            epoch_acc = 100.0 * total_correct / total_samples if total_samples else 0.0
            return epoch_loss, epoch_acc, best_batch_acc

    # ----------------------------------------------------------------
    # HPC partial code also calls `_evaluate()` after each epoch
    # ----------------------------------------------------------------
    def _evaluate(self):
        """
        Evaluate on `self.val_loader`. We'll do a 2x2 confusion matrix for "double"
        (the binary head). For single-head variants, we do a normal confusion matrix 
        sized to #classes.

        Returns a dict with "accuracy", "confusion_matrix", and optional "roc_*" keys.
        """
        if not hasattr(self, 'val_loader'):
            return {"accuracy":0.0, "confusion_matrix":None}

        self.model.eval()
        total_correct = 0
        total_samples = 0

        # confmat size
        if self.model_variant == "binary":
            confmat_dim = 2
        elif self.model_variant == "double":
            confmat_dim = 2  # we measure binary head's confmat
        elif self.model_variant == "threeclass":
            confmat_dim = 3
        else:
            # "multi"/"simple"
            confmat_dim = self.num_classes

        confmat = torch.zeros(confmat_dim, confmat_dim, dtype=torch.int32, device=self.device)
        all_labels = []
        all_scores = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                if self.model_variant == "double":
                    # evaluate binary head
                    bin_labels = (labels != 0).long()
                    out_bin = self.model(images, mode='binary')
                    probs_bin = nn.functional.softmax(out_bin, dim=1)
                    _, bin_preds = torch.max(out_bin, dim=1)

                    batch_correct = (bin_preds == bin_labels).sum().item()
                    total_correct += batch_correct
                    total_samples += labels.size(0)

                    for t,p in zip(bin_labels, bin_preds):
                        confmat[t,p] += 1

                    all_labels.extend(bin_labels.cpu().numpy().tolist())
                    all_scores.extend(probs_bin[:,1].cpu().numpy().tolist())

                else:
                    # single-head
                    logits = self.model(images)
                    probs = nn.functional.softmax(logits, dim=1)
                    preds = logits.argmax(dim=1)

                    batch_correct = (preds==labels).sum().item()
                    total_correct += batch_correct
                    total_samples += labels.size(0)

                    for t,p in zip(labels, preds):
                        confmat[t,p]+=1

                    if self.model_variant=="binary":
                        all_labels.extend(labels.cpu().numpy().tolist())
                        all_scores.extend(probs[:,1].cpu().numpy().tolist())

        accuracy = 100.0 * total_correct / total_samples if total_samples>0 else 0.0
        metrics = {
            "accuracy": accuracy,
            "confusion_matrix": confmat
        }
        if (self.model_variant in ["binary","double"]) and _HAS_SKLEARN:
            try:
                fpr, tpr, _ = roc_curve(all_labels, all_scores)
                auc_val = roc_auc_score(all_labels, all_scores)
                metrics["roc_fpr"] = fpr
                metrics["roc_tpr"] = tpr
                metrics["roc_auc"] = auc_val
            except ValueError:
                metrics["roc_auc"] = None
        return metrics

    def _extract_roc_data(self, val_metrics):
        if "roc_fpr" in val_metrics:
            return {
                "fpr": val_metrics["roc_fpr"],
                "tpr": val_metrics["roc_tpr"],
                "auc": val_metrics.get("roc_auc", None)
            }
        else:
            return {"fpr":None,"tpr":None,"auc":None}

    def _compute_class_metrics(self, val_metrics):
        if val_metrics["confusion_matrix"] is None:
            return {}
        cmat = val_metrics["confusion_matrix"].detach().cpu().numpy().astype(float)
        numc = cmat.shape[0]
        precision, recall, f1_vals, support = [], [], [], []

        for i in range(numc):
            row_sum = cmat[i,:].sum()
            col_sum = cmat[:,i].sum()
            tp = cmat[i,i]
            prec = tp/col_sum if col_sum>0 else 0.0
            rec = tp/row_sum if row_sum>0 else 0.0
            f1 = 2*prec*rec/(prec+rec) if (prec+rec)>1e-9 else 0.0
            precision.append(prec)
            recall.append(rec)
            f1_vals.append(f1)
            support.append(int(row_sum))

        total_tp = np.trace(cmat)
        total_sum = cmat.sum()
        micro_f1 = total_tp/total_sum if total_sum>0 else 0.0
        macro_f1 = float(np.mean(f1_vals))

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1_vals,
            "support": support,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1
        }

    # ----------------------------------------------------------------
    # K-Fold code (if needed)
    # ----------------------------------------------------------------
    def run_kfold(self, folds=5, epochs=5, progress_cb=None):
        """
        If folds>1, we only loaded 'Training' as self.train_dataset.
        We'll do a simpler approach, using `_train_one_epoch_fold()` for the fold logic,
        focusing on single-head or the binary part of 'double'. Adjust if needed.
        """
        from sklearn.model_selection import KFold
        n_samples = len(self.train_dataset)
        idx_arr = np.arange(n_samples)
        kf = KFold(n_splits=folds, shuffle=True, random_state=42)

        accum_confmat = None
        sum_metrics = None
        total_acc = 0.0

        for fold_i, (tr_idx, val_idx) in enumerate(kf.split(idx_arr)):
            self.log_func(f"\n--- Fold {fold_i+1}/{folds} ---")
            # re-init model each fold
            self.model, _ = get_model(
                model_variant=self.model_variant,
                input_size=self.input_size,
                dropout=self.dropout
            )
            self.model.to(self.device)
            self._setup_optimizer_criterion()

            from torch.utils.data import Subset
            tr_ds = Subset(self.train_dataset, tr_idx)
            val_ds = Subset(self.train_dataset, val_idx)
            tr_loader = DataLoader(tr_ds, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

            best_val = 0.0
            no_improve = 0
            for ep in range(epochs):
                tr_loss, tr_acc, _ = self._train_one_epoch_fold(tr_loader)
                val_metrics = self._evaluate_fold(val_loader)
                val_acc = val_metrics["accuracy"]

                msg = f"[Fold {fold_i+1}, Ep {ep+1}/{epochs}] TrainAcc={tr_acc:.1f}%, ValAcc={val_acc:.1f}%"
                self.log_func(msg)

                if val_acc>best_val:
                    best_val = val_acc
                    no_improve = 0
                    best_confmat = val_metrics["confusion_matrix"]
                    best_mets = self._compute_class_metrics(val_metrics)
                else:
                    no_improve += 1
                    if self.earlystop_patience>0 and no_improve>=self.earlystop_patience:
                        self.log_func(f"[EarlyStop fold {fold_i+1} at ep {ep+1}]")
                        break

            total_acc += best_val
            if accum_confmat is None:
                accum_confmat = best_confmat.clone()
            else:
                accum_confmat += best_confmat

            if sum_metrics is None:
                sum_metrics = {}
                for k, v in best_mets.items():
                    sum_metrics[k] = np.array(v, dtype=float)
            else:
                for k, v in best_mets.items():
                    sum_metrics[k] += np.array(v, dtype=float)

        avg_acc = total_acc/folds
        final_metrics = {}
        if sum_metrics:
            for k, arr_val in sum_metrics.items():
                final_metrics[k] = (arr_val/folds).tolist()

        self.final_confmat = accum_confmat
        # compute micro/macro from accum
        final_val_metrics = {"accuracy": avg_acc, "confusion_matrix": accum_confmat}
        extra = self._compute_class_metrics(final_val_metrics)
        for kk, vv in extra.items():
            final_metrics[kk] = vv

        return avg_acc, accum_confmat, final_metrics, None

    def _train_one_epoch_fold(self, loader):
        """
        Minimal k-fold training epoch on a custom loader. 
        We'll do the same logic as `_train_one_epoch`, but ignoring multi-head for "double"
        (or just training the binary head). Adjust if needed.
        Returns (epoch_loss, epoch_acc, best_batch_acc).
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        best_batch_acc = 0.0

        if self.model_variant == "double":
            # e.g. train just the binary head in folds
            for images, labels in loader:
                images = images.to(self.device)
                bin_labels = (labels!=0).long()

                self.optimizer.zero_grad()
                out_bin = self.model(images, mode='binary')
                loss_bin = self.criterion_bin(out_bin, bin_labels)
                loss_bin.backward()
                self.optimizer.step()

                bs = images.size(0)
                total_loss += float(loss_bin)*bs
                total_samples += bs

                preds = out_bin.argmax(dim=1)
                batch_correct = (preds==bin_labels).sum().item()
                total_correct += batch_correct

                batch_acc = 100.0*batch_correct/bs
                if batch_acc>best_batch_acc:
                    best_batch_acc = batch_acc

        else:
            # single-head
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                bs = images.size(0)
                total_loss += float(loss)*bs
                preds = outputs.argmax(dim=1)
                batch_correct = (preds==labels).sum().item()
                total_correct += batch_correct
                total_samples += bs

                batch_acc = 100.0*batch_correct/bs
                if batch_acc>best_batch_acc:
                    best_batch_acc = batch_acc

        epoch_loss = total_loss / total_samples if total_samples else 0.0
        epoch_acc = 100.0 * total_correct / total_samples if total_samples else 0.0
        return epoch_loss, epoch_acc, best_batch_acc

    def _evaluate_fold(self, loader):
        """
        Evaluate on the fold's loader. 
        We'll do a binary confmat for 'double' or 'binary',
        else a single-head confmat sized to #classes.
        Returns a dict with "accuracy", "confusion_matrix", plus optional ROC.
        """
        self.model.eval()
        total_correct = 0
        total_samples = 0

        if self.model_variant=="double" or self.model_variant=="binary":
            confmat_dim = 2
        else:
            confmat_dim = self.num_classes

        confmat = torch.zeros(confmat_dim, confmat_dim, dtype=torch.int32, device=self.device)
        all_labels, all_scores = [], []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                if self.model_variant=="double":
                    bin_labels = (labels!=0).long()
                    out_bin = self.model(images, mode='binary')
                    preds_bin = out_bin.argmax(dim=1)
                    probs_bin = nn.functional.softmax(out_bin, dim=1)

                    bs = images.size(0)
                    total_samples += bs
                    total_correct += (preds_bin==bin_labels).sum().item()

                    for t,p in zip(bin_labels, preds_bin):
                        confmat[t,p]+=1

                    all_labels.extend(bin_labels.cpu().numpy().tolist())
                    all_scores.extend(probs_bin[:,1].cpu().numpy().tolist())

                else:
                    logits = self.model(images)
                    preds = logits.argmax(dim=1)
                    probs = nn.functional.softmax(logits, dim=1)

                    bs = images.size(0)
                    total_samples += bs
                    total_correct += (preds==labels).sum().item()

                    for t,p in zip(labels, preds):
                        confmat[t,p]+=1

                    if self.model_variant=="binary":
                        all_labels.extend(labels.cpu().numpy().tolist())
                        all_scores.extend(probs[:,1].cpu().numpy().tolist())

        acc = 100.0 * total_correct / total_samples if total_samples else 0.0
        metrics = {
            "accuracy": acc,
            "confusion_matrix": confmat
        }
        if (self.model_variant in ["binary","double"]) and _HAS_SKLEARN:
            try:
                fpr, tpr, _ = roc_curve(all_labels, all_scores)
                aucv = roc_auc_score(all_labels, all_scores)
                metrics["roc_fpr"] = fpr
                metrics["roc_tpr"] = tpr
                metrics["roc_auc"] = aucv
            except ValueError:
                metrics["roc_auc"] = None
        return metrics
import torch
import torch.nn as nn

################################################################################
# Single-branch CNN for any # of classes
################################################################################

class SimpleCNN(nn.Module):
    """
    A baseline CNN with optional dropout.
    - num_classes: number of output classes
    - input_size: input image dimension (square)
    - dropout: dropout probability
    """
    def __init__(self, num_classes=4, input_size=128, dropout=0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )
        flattened_dim = 16 * (input_size // 4) * (input_size // 4)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

################################################################################
# Two‐headed CNN => "double" (binary + 3-class)
################################################################################

class DoubleHeadCNN(nn.Module):
    """
    A CNN that uses a single shared feature extractor but has two output "heads":
      (1) Binary: NoTumor vs Tumor
      (2) Threeclass: Pituitary / Meningioma / Glioma

    We choose the output by calling forward(..., mode='binary') or mode='threeclass'.
    """
    def __init__(self, input_size=128, dropout=0.0):
        super().__init__()
        self.shared_features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )
        flattened_dim = 16 * (input_size // 4) * (input_size // 4)

        # Head 1: Binary => [NoTumor, Tumor]
        self.head_binary = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)  # => 2 outputs
        )

        # Head 2: threeclass => [Pituitary, Meningioma, Glioma]
        self.head_threeclass = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3)
        )

    def forward(self, x, mode='binary'):
        """
        mode='binary' => returns [batch_size, 2] (NoTumor vs Tumor)
        mode='threeclass' => returns [batch_size, 3] (pituitary / meningioma / glioma)
        """
        feats = self.shared_features(x)
        if mode == 'binary':
            return self.head_binary(feats)
        elif mode == 'threeclass':
            return self.head_threeclass(feats)
        else:
            raise ValueError(f"Invalid mode='{mode}' in DoubleHeadCNN.")

################################################################################
# get_model with new names: "simple" (4-class), "threeclass", "binary", "double"
################################################################################

def get_model(model_variant, input_size=128, dropout=0.0):
    """
    Create a model instance:
      - "simple" => single CNN, 4-class
      - "threeclass" => single CNN, 3-class
      - "binary" => single CNN, 2-class
      - "double" => two-headed net (binary + multi)
    """
    if model_variant == "simple":
        return SimpleCNN(num_classes=4, input_size=input_size, dropout=dropout), 4

    elif model_variant == "threeclass":
        return SimpleCNN(num_classes=3, input_size=input_size, dropout=dropout), 3

    elif model_variant == "binary":
        return SimpleCNN(num_classes=2, input_size=input_size, dropout=dropout), 2

    elif model_variant == "double":
        return DoubleHeadCNN(input_size=input_size, dropout=dropout), None

    else:
        raise ValueError(f"Unknown model variant: {model_variant}")
"""
File: src/transforms_config.py

Provides a function `create_transform` that constructs a PyTorch
transformation pipeline for grayscale MRI images.
"""

import torchvision.transforms as T

def create_transform(
    image_size=(128, 128),
    augment=False,
    adv_aug=False,
    strong_aug=False,
    rotation_deg=0,
    brightness_range=(1.0, 1.0)
):
    """
    Build a transform pipeline (no explicit intensity normalization here).

    Args:
        image_size (tuple): (width, height) to resize/crop.
        augment (bool): If True, apply mild data augmentation (rotation, brightness).
        adv_aug (bool): If True, apply advanced augmentation like RandomResizedCrop.
        strong_aug (bool): If True, apply extra transforms like RandomPerspective.
        rotation_deg (float): Max degrees for RandomRotation if augment=True.
        brightness_range (tuple): Brightness/contrast jitter range if augment=True.

    Returns:
        torchvision.transforms.Compose: A composed transform to apply to images.
    """
    transform_list = []

    # If we're not using advanced augmentation, do a direct Resize
    if not adv_aug:
        transform_list.append(T.Resize(image_size))

    # Mild augmentations
    if augment:
        if rotation_deg > 0:
            transform_list.append(T.RandomRotation(degrees=rotation_deg))
        if brightness_range != (1.0, 1.0):
            transform_list.append(
                T.ColorJitter(
                    brightness=brightness_range,
                    contrast=brightness_range
                )
            )

    # If adv_aug is True, we do a RandomResizedCrop
    if adv_aug:
        transform_list.append(T.RandomResizedCrop(size=image_size, scale=(0.8, 1.0)))

    # If strong_aug is True, add perspective warp
    if strong_aug:
        transform_list.append(T.RandomPerspective(distortion_scale=0.2, p=0.5))

    # Finally convert to tensor
    transform_list.append(T.ToTensor())

    return T.Compose(transform_list)
