# File: gui/augmentation_tab.py

import random
from PIL import Image
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QFormLayout, QHBoxLayout,
    QPushButton, QCheckBox, QSlider, QFileDialog, QMessageBox, QSpinBox,
    QTextEdit, QProgressBar, QTabWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from gui.widgets import MplCanvas
from logic.augmentation import preview_augmentation, bulk_augment_dataset


class AugmentationWorker(QThread):
    """
    Bulk augmentation in background.
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
        balance_classes=False,
        parent=None
    ):
        super().__init__(parent)
        self.copies = copies
        self.rotate_deg = rotate_deg
        self.do_rotate = do_rotate
        self.do_bc = do_bc
        self.do_gamma = do_gamma
        self.balance_classes = balance_classes

    def run(self):
        self.progress.emit(0, "Augmenting...")
        try:
            total_new_images, error_log = bulk_augment_dataset(
                src_root="cleaned_data",
                dst_root="augmented_data",
                copies_per_image=self.copies,
                rotate_deg=self.rotate_deg,
                do_rotate=self.do_rotate,
                do_brightness_contrast=self.do_bc,
                do_gamma=self.do_gamma,
                balance_classes=self.balance_classes  # [ADDED]
            )
            self.progress.emit(100, "All done!")
            self.finished.emit(total_new_images, error_log)
        except Exception as e:
            msg = f"Augmentation error: {e}"
            self.progress.emit(0, msg)
            self.finished.emit(0, [msg])


class AugmentationTab(QWidget):
    """
    Reorganized to use subtabs:
      - Tab 1: Preview
      - Tab 2: Bulk Augmentation
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        main_layout = QVBoxLayout(self)

        intro_label = QLabel("<h3>Data Augmentation</h3>")
        intro_label.setWordWrap(True)
        main_layout.addWidget(intro_label)

        # Create sub-tab widget
        self.sub_tabs = QTabWidget()
        self.sub_tabs.setTabPosition(QTabWidget.North)

        # Create the two subtabs
        preview_tab = self._create_preview_tab()
        bulk_tab = self._create_bulk_tab()

        self.sub_tabs.addTab(preview_tab, "Preview")
        self.sub_tabs.addTab(bulk_tab, "Bulk Augmentation")

        main_layout.addWidget(self.sub_tabs)
        self.setLayout(main_layout)

        # Worker reference
        self.worker = None

        # We'll store the loaded image for preview
        self.original_img = None

    def _create_preview_tab(self):
        """
        Builds the "Preview" sub-tab UI.
        """
        preview_widget = QWidget()
        layout = QVBoxLayout(preview_widget)

        # -- Group box for the preview controls --
        preview_box = QGroupBox("Preview Configuration")
        form = QFormLayout()

        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.on_load_image)

        self.rotate_chk = QCheckBox("Rotate")
        self.rotate_chk.setChecked(True)
        self.rotate_slider = QSlider(Qt.Horizontal)
        self.rotate_slider.setRange(1, 30)
        self.rotate_slider.setValue(15)

        self.bright_chk = QCheckBox("Brightness/Contrast")
        self.bright_chk.setChecked(True)

        self.gamma_chk = QCheckBox("Gamma")
        self.gamma_chk.setChecked(True)

        self.norm_chk = QCheckBox("MinMax Preview")

        self.prev_btn = QPushButton("Apply")
        self.prev_btn.clicked.connect(self.on_preview)

        # Layout arrangement
        form.addRow(self.load_btn)

        rot_row = QHBoxLayout()
        rot_row.addWidget(self.rotate_chk)
        rot_row.addWidget(self.rotate_slider)
        form.addRow("Rotation Range:", rot_row)

        bc_row = QHBoxLayout()
        bc_row.addWidget(self.bright_chk)
        bc_row.addWidget(self.gamma_chk)
        form.addRow("Other:", bc_row)

        form.addRow(self.norm_chk, QLabel("(preview only)"))
        form.addRow(self.prev_btn)

        preview_box.setLayout(form)
        layout.addWidget(preview_box)

        # -- Matplotlib canvas for side-by-side display --
        self.canvas = MplCanvas(preview_widget, width=6, height=4)
        layout.addWidget(self.canvas)

        return preview_widget

    def _create_bulk_tab(self):
        """
        Builds the "Bulk Augmentation" sub-tab UI.
        """
        bulk_widget = QWidget()
        layout = QVBoxLayout(bulk_widget)

        # Group box for bulk
        bulk_box = QGroupBox("Bulk Augmentation Configuration")
        bulk_layout = QFormLayout()

        self.copy_spin = QSpinBox()
        self.copy_spin.setRange(1, 20)
        self.copy_spin.setValue(2)

        self.balance_chk = QCheckBox("Balance Classes?")  # [ADDED]
        self.bulk_btn = QPushButton("Start Bulk Augmentation")
        self.bulk_btn.clicked.connect(self.on_bulk_augment)

        self.aug_progress = QProgressBar()
        self.aug_log = QTextEdit()
        self.aug_log.setReadOnly(True)

        bulk_layout.addRow("Copies per Image:", self.copy_spin)
        bulk_layout.addRow(self.balance_chk)
        bulk_layout.addRow(self.bulk_btn)
        bulk_layout.addRow(self.aug_progress)
        bulk_layout.addRow(self.aug_log)

        bulk_box.setLayout(bulk_layout)
        layout.addWidget(bulk_box)

        return bulk_widget

    # --------------------------
    # Preview-related callbacks
    # --------------------------
    def on_load_image(self):
        dlg = QFileDialog(self)
        dlg.setNameFilter("Images (*.png *.jpg *.jpeg)")
        if dlg.exec_():
            selected = dlg.selectedFiles()[0]
            try:
                self.original_img = Image.open(selected).convert('L')
                self.show_images(self.original_img, None)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not load image:\n{e}")

    def on_preview(self):
        if not self.original_img:
            QMessageBox.warning(self, "No Image", "Load an image first.")
            return
        rotate_deg = self.rotate_slider.value() if self.rotate_chk.isChecked() else 0
        aug_tensor = preview_augmentation(
            pil_image=self.original_img,
            rotate_deg=rotate_deg,
            do_rotate=self.rotate_chk.isChecked(),
            do_brightness_contrast=self.bright_chk.isChecked(),
            do_gamma=self.gamma_chk.isChecked(),
            do_minmax=self.norm_chk.isChecked()
        )
        self.show_images(self.original_img, aug_tensor)

    def show_images(self, orig_pil, aug_tens):
        self.canvas.figure.clear()
        ax1, ax2 = self.canvas.figure.subplots(1, 2)

        ax1.imshow(orig_pil, cmap='gray')
        ax1.set_title("Original")
        ax1.axis('off')

        if aug_tens is not None:
            arr = aug_tens.squeeze().numpy()
            ax2.imshow(arr, cmap='gray')
            ax2.set_title("Augmented")
        else:
            ax2.set_title("No Preview")
        ax2.axis('off')
        self.canvas.draw()

    # --------------------------
    # Bulk-related callbacks
    # --------------------------
    def on_bulk_augment(self):
        copies = self.copy_spin.value()
        rotate_deg = self.rotate_slider.value() if self.rotate_chk.isChecked() else 0
        do_rotate = self.rotate_chk.isChecked()
        do_bc = self.bright_chk.isChecked()
        do_gamma = self.gamma_chk.isChecked()
        balance_classes = self.balance_chk.isChecked()

        self.aug_log.clear()
        self.worker = AugmentationWorker(
            copies=copies,
            rotate_deg=rotate_deg,
            do_rotate=do_rotate,
            do_bc=do_bc,
            do_gamma=do_gamma,
            balance_classes=balance_classes
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
