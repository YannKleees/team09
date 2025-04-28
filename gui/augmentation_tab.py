# File: gui/augmentation_tab.py

import os
import random
import math
from PIL import Image
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QFormLayout, QHBoxLayout,
    QPushButton, QCheckBox, QSlider, QFileDialog, QMessageBox, QSpinBox,
    QTextEdit, QProgressBar, QTabWidget, QGridLayout, QSpacerItem, QSizePolicy,
    QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from gui.widgets import MplCanvas
from logic.augmentation import (
    preview_augmentation,
    bulk_augment_dataset
)

import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F

# Additional helper to count distribution
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
    for subset in ["Training", "Testing"]:
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
    Bulk augmentation in background. We always do balance_classes=True internally.
    """
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(int, list)  # (total_new_images, error_log)

    def __init__(
        self,
        copies=2,
        rotate_deg=15,
        do_rotate=True,
        do_bc=True,
        do_gamma=True,
        # we remove balance_classes param => always True
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
        self.copies = copies
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
            # always balanced
            total_new_images, error_log = bulk_augment_dataset(
                src_root="cleaned_data",
                dst_root="augmented_data",
                copies_per_image=self.copies,
                rotate_deg=self.rotate_deg,
                do_rotate=self.do_rotate,
                do_brightness_contrast=self.do_bc,
                do_gamma=self.do_gamma,
                balance_classes=True,  # forced
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


class AugmentationTab(QWidget):
    """
    Reorganized to use subtabs:
      - Tab 1: Preview (multi-preview)
      - Tab 2: Bulk Augmentation
      - ALWAYS balanced => no checkbox
      - Add a 'Distribution Dashboard' group to compare cleaned vs. augmented
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        main_layout = QVBoxLayout(self)

        intro_label = QLabel("<h3>Data Augmentation (Balanced)</h3>")
        intro_label.setWordWrap(True)
        main_layout.addWidget(intro_label)

        # Sub-tab widget
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
        ...
        # (identical to your advanced version)
        # Just copy/paste the entire code from your final version above
        # minus the "balance_chk" references, etc.
        # We'll show the full final code snippet below for clarity.
        # ...
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

        orig_lbl = QLabel("Original")
        grid_layout.addWidget(orig_lbl, row, 0, 1, 1, alignment=Qt.AlignHCenter)
        orig_canvas = MplCanvas(grid_widget, width=3, height=3)
        self._show_pil_on_canvas(orig_canvas, self.original_img)
        grid_layout.addWidget(orig_canvas, row+1, 0)

        for i in range(n_previews):
            col = i+1
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
        bulk_widget = QWidget()
        layout = QVBoxLayout(bulk_widget)

        # 1) The main config box
        bulk_box = QGroupBox("Bulk Augmentation Configuration")
        bform = QFormLayout()

        self.copy_spin = QSpinBox()
        self.copy_spin.setRange(1, 20)
        self.copy_spin.setValue(2)

        # Rotation
        self.bulk_rotate_chk = QCheckBox("Rotate?")
        self.bulk_rotate_chk.setChecked(True)
        self.bulk_rotate_slider = QSlider(Qt.Horizontal)
        self.bulk_rotate_slider.setRange(0, 60)
        self.bulk_rotate_slider.setValue(15)

        # B/C
        self.bulk_bc_chk = QCheckBox("Brightness/Contrast?")
        self.bulk_bc_chk.setChecked(True)
        self.bulk_b_min = QDoubleSpinBoxRange(0.1, 2.0, 0.85, "Bmin")
        self.bulk_b_max = QDoubleSpinBoxRange(0.1, 2.0, 1.15, "Bmax")
        self.bulk_c_min = QDoubleSpinBoxRange(0.1, 2.0, 0.85, "Cmin")
        self.bulk_c_max = QDoubleSpinBoxRange(0.1, 2.0, 1.15, "Cmax")

        # Gamma
        self.bulk_gamma_chk = QCheckBox("Gamma?")
        self.bulk_gamma_chk.setChecked(True)
        self.bulk_g_min = QDoubleSpinBoxRange(0.1, 2.0, 0.85, "Gmin")
        self.bulk_g_max = QDoubleSpinBoxRange(0.1, 2.0, 1.15, "Gmax")

        # Additional toggles
        self.bulk_flip_chk = QCheckBox("Flip?")
        self.bulk_perspective_chk = QCheckBox("Perspective?")
        self.bulk_crop_chk = QCheckBox("Crop?")

        # No more "Balance Classes?" => always balanced

        self.bulk_btn = QPushButton("Start Bulk Augmentation")
        self.bulk_btn.clicked.connect(self.on_bulk_augment)

        self.aug_progress = QProgressBar()
        self.aug_progress.setValue(0)
        self.aug_log = QTextEdit()
        self.aug_log.setReadOnly(True)

        bform.addRow("Copies per Image:", self.copy_spin)

        # rotate
        rot_layout = QHBoxLayout()
        rot_layout.addWidget(self.bulk_rotate_chk)
        rot_layout.addWidget(self.bulk_rotate_slider)
        bform.addRow("Rotation:", rot_layout)

        # B/C
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

        # Gamma
        g_line = QHBoxLayout()
        g_line.addWidget(self.bulk_gamma_chk)
        bform.addRow("Gamma?", g_line)

        g2_line = QHBoxLayout()
        g2_line.addWidget(self.bulk_g_min)
        g2_line.addWidget(self.bulk_g_max)
        bform.addRow("Gamma Range:", g2_line)

        # advanced toggles
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

        # 2) Distribution Dashboard
        dist_box = QGroupBox("Distribution Dashboard")
        dist_layout = QVBoxLayout(dist_box)
        info_label = QLabel(
            "Shows how class counts changed from cleaned_data to augmented_data "
            "(always balanced)."
        )
        info_label.setWordWrap(True)
        dist_layout.addWidget(info_label)

        self.dist_btn = QPushButton("Show Distribution Comparison")
        self.dist_btn.clicked.connect(self.on_show_distribution)
        dist_layout.addWidget(self.dist_btn)

        # We'll place a small Matplotlib canvas for side-by-side bar chart
        self.dist_canvas = MplCanvas(self, width=6, height=4)
        dist_toolbar = NavigationToolbar(self.dist_canvas, self)
        dist_layout.addWidget(self.dist_canvas)
        dist_layout.addWidget(dist_toolbar)

        layout.addWidget(dist_box)

        return bulk_widget

    def on_bulk_augment(self):
        self.aug_log.clear()
        self.aug_progress.setValue(0)

        copies = self.copy_spin.value()
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
            copies=copies,
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
        self.aug_log.append(f"Total new images: {total_new_images}")
        if error_log:
            self.aug_log.append("<b>Errors:</b>")
            for e in error_log:
                self.aug_log.append(e)

    def on_show_distribution(self):
        """
        Compare class distribution in 'cleaned_data' vs. 'augmented_data',
        then plot side by side in a bar chart.
        """
        import matplotlib.pyplot as plt
        from collections import OrderedDict

        cleaned_dist = count_distribution("cleaned_data")
        aug_dist = count_distribution("augmented_data")

        # Build sorted list of keys
        # We might have keys like "Training/ClassA", "Training/ClassB"...
        all_keys = set(cleaned_dist.keys()).union(set(aug_dist.keys()))
        all_keys = sorted(all_keys)  # alphabetical => e.g. "Testing/ClassA" etc.

        cleaned_vals = []
        aug_vals = []
        for k in all_keys:
            cleaned_vals.append(cleaned_dist.get(k, 0))
            aug_vals.append(aug_dist.get(k, 0))

        # Plot them side by side
        self.dist_canvas.figure.clear()
        ax = self.dist_canvas.figure.subplots()

        x_positions = np.arange(len(all_keys))
        width = 0.35
        ax.bar(x_positions - width/2, cleaned_vals, width, label="Cleaned")
        ax.bar(x_positions + width/2, aug_vals, width, label="Augmented")

        ax.set_ylabel("Image Count")
        ax.set_title("Class Distribution: Cleaned vs. Augmented")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(all_keys, rotation=45, ha='right')
        ax.legend()

        self.dist_canvas.figure.tight_layout()
        self.dist_canvas.draw()

# Helper widget for double spin boxes
from PyQt5.QtWidgets import QDoubleSpinBox
class QDoubleSpinBoxRange(QDoubleSpinBox):
    def __init__(self, minv, maxv, default, label_name="", parent=None):
        super().__init__(parent)
        self.setRange(minv, maxv)
        self.setSingleStep(0.05)
        self.setValue(default)
        self.setDecimals(2)
        self.setToolTip(label_name)
