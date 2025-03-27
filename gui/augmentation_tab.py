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
            # We'll do partial increments; there's not an easy way to measure sub-progress
            # so we'll just do a final 100% when complete
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
    1) Preview mild transforms on a single loaded image
    2) Bulk apply random augmentation to the entire cleaned_data -> augmented_data
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()

        # Section 1: Preview
        doc_label = QLabel(
            "<b>Data Augmentation Preview</b><br>"
            "Try mild transformations on a chosen image.<br>"
            "Rotation, brightness/contrast, gamma, minmax normalization."
        )
        doc_label.setWordWrap(True)
        layout.addWidget(doc_label)

        aug_box = QGroupBox("Augmentation Preview")
        form = QFormLayout()

        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.on_load_image)

        self.rotate_chk = QCheckBox("Random Rotation")
        self.rotate_chk.setChecked(True)
        self.rotate_slider = QSlider(Qt.Horizontal)
        self.rotate_slider.setRange(1, 30)
        self.rotate_slider.setValue(15)

        self.bright_chk = QCheckBox("Brightness/Contrast")
        self.bright_chk.setChecked(True)

        self.gamma_chk = QCheckBox("Gamma Correction")
        self.gamma_chk.setChecked(True)

        self.norm_chk = QCheckBox("MinMax Normalization (Preview)")

        self.prev_btn = QPushButton("Preview")
        self.prev_btn.clicked.connect(self.on_preview)

        form.addRow(self.load_btn)
        row1 = QHBoxLayout()
        row1.addWidget(self.rotate_chk)
        row1.addWidget(self.rotate_slider)
        form.addRow("Rotation Range:", row1)

        row2 = QHBoxLayout()
        row2.addWidget(self.bright_chk)
        row2.addWidget(self.gamma_chk)
        form.addRow("Other Transforms:", row2)

        form.addRow(self.norm_chk, QLabel("Applies after transforms"))
        form.addRow(self.prev_btn)

        aug_box.setLayout(form)
        layout.addWidget(aug_box)

        self.canvas = MplCanvas(self, width=6, height=4)
        layout.addWidget(self.canvas)

        self.original_img = None

        # Section 2: Bulk Augmentation
        bulk_box = QGroupBox("Bulk Augmentation to 'augmented_data'")
        bulk_layout = QFormLayout()

        self.copy_spin = QSpinBox()
        self.copy_spin.setRange(1, 20)
        self.copy_spin.setValue(2)

        self.bulk_btn = QPushButton("Apply Bulk Augmentation")
        self.bulk_btn.clicked.connect(self.on_bulk_augment)

        self.aug_progress = QProgressBar()
        self.aug_log = QTextEdit()
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
        self.canvas.figure.clear()
        ax1, ax2 = self.canvas.figure.subplots(1,2)

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
        """
        Apply random augmentation in bulk to 'cleaned_data', saving into 'augmented_data'.
        We'll re-use the same rotation/bc/gamma flags from the preview, 
        plus a user-chosen 'copies_per_image'.
        """
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

        QMessageBox.information(self, "Done", "Bulk Augmentation complete!")
