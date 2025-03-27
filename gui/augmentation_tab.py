import random
from PIL import Image

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QFormLayout, QHBoxLayout,
    QPushButton, QCheckBox, QSlider, QFileDialog, QMessageBox, QSpinBox, QTextEdit, QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from gui.widgets import MplCanvas
from logic.augmentation import preview_augmentation, bulk_augment_dataset

class AugmentationWorker(QThread):
    """
    QThread that performs bulk augmentation in the background.
    """
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(int, list)  # total_new_images, error_log

    def __init__(self, copies=2, rotate_deg=15, do_rotate=True, do_bc=True, do_gamma=True, parent=None):
        super().__init__(parent)
        self.copies = copies
        self.rotate_deg = rotate_deg
        self.do_rotate = do_rotate
        self.do_bc = do_bc
        self.do_gamma = do_gamma

    def run(self):
        self.progress.emit(0, "Starting Bulk Augmentation...")
        try:
            total_new_images, error_log = bulk_augment_dataset(
                src_root="cleaned_data",
                dst_root="augmented_data",
                copies_per_image=self.copies,
                rotate_deg=self.rotate_deg,
                do_rotate=self.do_rotate,
                do_brightness_contrast=self.do_bc,
                do_gamma=self.do_gamma
            )
            self.progress.emit(100, "All done!")
            self.finished.emit(total_new_images, error_log)
        except Exception as e:
            msg = f"Augmentation error: {str(e)}"
            self.progress.emit(0, msg)
            self.finished.emit(0, [msg])

class AugmentationTab(QWidget):
    """
    Enables:
      1) Previewing random augmentations on a single image
      2) Bulk augmentation from cleaned_data -> augmented_data
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()

        # Section: Preview
        preview_label = QLabel(
            "<h3>Data Augmentation Preview</h3>"
            "Load an image and preview rotation, brightness/contrast, gamma, etc."
        )
        preview_label.setWordWrap(True)
        layout.addWidget(preview_label)

        aug_box = QGroupBox("Augmentation Preview")
        form = QFormLayout()

        self.load_btn = QPushButton("Load Image")
        self.load_btn.setToolTip("Load a local image (PNG/JPG) to test transformations.")
        self.load_btn.clicked.connect(self.on_load_image)

        self.rotate_chk = QCheckBox("Random Rotation")
        self.rotate_chk.setChecked(True)
        self.rotate_chk.setToolTip("Randomly rotate the image within a chosen degree range.")

        self.rotate_slider = QSlider(Qt.Horizontal)
        self.rotate_slider.setRange(1, 30)
        self.rotate_slider.setValue(15)
        self.rotate_slider.setToolTip("Maximum rotation angle.")

        self.bright_chk = QCheckBox("Brightness/Contrast")
        self.bright_chk.setChecked(True)
        self.bright_chk.setToolTip("Randomly tweak brightness & contrast.")

        self.gamma_chk = QCheckBox("Gamma Correction")
        self.gamma_chk.setChecked(True)
        self.gamma_chk.setToolTip("Randomly adjust gamma levels.")

        self.norm_chk = QCheckBox("MinMax Normalization (Preview)")
        self.norm_chk.setToolTip("Apply min-max normalization after transformations for demonstration.")

        self.prev_btn = QPushButton("Preview")
        self.prev_btn.setToolTip("Apply the selected random transformations to the loaded image.")
        self.prev_btn.clicked.connect(self.on_preview)

        form.addRow(self.load_btn)
        rrow = QHBoxLayout()
        rrow.addWidget(self.rotate_chk)
        rrow.addWidget(self.rotate_slider)
        form.addRow("Rotation Range:", rrow)

        brow = QHBoxLayout()
        brow.addWidget(self.bright_chk)
        brow.addWidget(self.gamma_chk)
        form.addRow("Other Transforms:", brow)

        form.addRow(self.norm_chk, QLabel("Applies after transforms"))
        form.addRow(self.prev_btn)

        aug_box.setLayout(form)
        layout.addWidget(aug_box)

        self.canvas = MplCanvas(self, width=6, height=4)
        layout.addWidget(self.canvas)

        self.original_img = None

        # Section: Bulk Augmentation
        bulk_box = QGroupBox("Bulk Augmentation")
        bulk_layout = QFormLayout()

        self.copy_spin = QSpinBox()
        self.copy_spin.setRange(1, 20)
        self.copy_spin.setValue(2)
        self.copy_spin.setToolTip("How many augmented copies to create per image in 'cleaned_data'.")

        self.bulk_btn = QPushButton("Apply Bulk Augmentation")
        self.bulk_btn.setToolTip("Generate multiple augmented copies for every image in 'cleaned_data'.")
        self.bulk_btn.clicked.connect(self.on_bulk_augment)

        self.aug_progress = QProgressBar()
        self.aug_progress.setToolTip("Shows progress of the bulk augmentation process.")
        self.aug_log = QTextEdit()
        self.aug_log.setToolTip("Logs summary and errors from the bulk augmentation.")
        self.aug_log.setReadOnly(True)

        bulk_layout.addRow("Copies per Image:", self.copy_spin)
        bulk_layout.addRow(self.bulk_btn)
        bulk_layout.addRow(self.aug_progress)
        bulk_layout.addRow(self.aug_log)
        bulk_box.setLayout(bulk_layout)
        layout.addWidget(bulk_box)

        self.setLayout(layout)
        self.worker = None

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
            QMessageBox.warning(self, "No Image", "Please load an image first.")
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
        """
        Update the MplCanvas to show original + augmented side by side.
        """
        self.canvas.figure.clear()
        ax1, ax2 = self.canvas.figure.subplots(1, 2)

        # Original
        ax1.imshow(orig_pil, cmap='gray')
        ax1.set_title("Original")
        ax1.axis('off')

        # Augmented
        if aug_tens is not None:
            arr = aug_tens.squeeze().numpy()
            ax2.imshow(arr, cmap='gray')
            ax2.set_title("Augmented")
        else:
            ax2.set_title("No Preview Yet")
        ax2.axis('off')

        self.canvas.draw()

    def on_bulk_augment(self):
        copies = self.copy_spin.value()
        rotate_deg = self.rotate_slider.value() if self.rotate_chk.isChecked() else 0
        do_rotate = self.rotate_chk.isChecked()
        do_bc = self.bright_chk.isChecked()
        do_gamma = self.gamma_chk.isChecked()

        self.aug_log.clear()

        self.worker = AugmentationWorker(
            copies=copies,
            rotate_deg=rotate_deg,
            do_rotate=do_rotate,
            do_bc=do_bc,
            do_gamma=do_gamma
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
        self.aug_log.append(f"Total new augmented images: {total_new_images}")
        if error_log:
            self.aug_log.append("<b>Errors:</b>")
            for e in error_log:
                self.aug_log.append(e)
