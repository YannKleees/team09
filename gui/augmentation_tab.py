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
    Count images per class in the given directory.

    :param root_dir: The root directory to scan.
    :return: Dict mapping subset/class to image count.
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
    QThread for performing bulk augmentation in the background.

    :param src_root: Path to the source folder.
    :param dst_root: Path to the destination folder.
    :param final_target: Target number of images per class.
    :param rotate_deg: Rotation angle in degrees.
    :param do_rotate: Whether to apply rotation.
    :param do_bc: Whether to apply brightness/contrast.
    :param do_gamma: Whether to apply gamma correction.
    :param do_flip: Whether to flip images.
    :param do_perspective: Whether to apply perspective warp.
    :param do_crop: Whether to apply random cropping.
    :param b_min: Minimum brightness value.
    :param b_max: Maximum brightness value.
    :param c_min: Minimum contrast value.
    :param c_max: Maximum contrast value.
    :param g_min: Minimum gamma value.
    :param g_max: Maximum gamma value.
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
    A QDoubleSpinBox with a preset range, step, and tooltip.

    :param minv: Minimum value.
    :param maxv: Maximum value.
    :param default: Default value.
    :param label_name: Tooltip text.
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
    Main panel for data augmentation with two subtabs:
    - Preview Tab
    - Bulk Augmentation Tab
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
