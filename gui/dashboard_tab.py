# gui/dashboard_tab.py

import os
import random
import numpy as np
from PIL import Image

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QHBoxLayout,
    QPushButton, QComboBox, QTextEdit, QMessageBox
)
from PyQt5.QtCore import Qt
from gui.widgets import MplCanvas

class DashboardTab(QWidget):
    """
    A "Dashboard" that summarizes data structure, image metadata,
    intensity distribution, etc.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QVBoxLayout(self)

        info_label = QLabel(
            "<b>Data Overview & Dashboard</b><br>"
            "Quickly see file counts, class distribution, basic image metadata, "
            "and intensity histograms from a sample of images."
        )
        info_label.setWordWrap(True)
        main_layout.addWidget(info_label)

        # A small combo box to pick which folder to analyze
        self.folder_combo = QComboBox()
        self.folder_combo.addItems(["data/archive", "cleaned_data", "augmented_data"])
        self.btn_analyze = QPushButton("Analyze Selected Folder")
        self.btn_analyze.clicked.connect(self.on_analyze)

        layout_row = QHBoxLayout()
        layout_row.addWidget(self.folder_combo)
        layout_row.addWidget(self.btn_analyze)
        main_layout.addLayout(layout_row)

        # Log area
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        main_layout.addWidget(self.log_box)

        # Matplotlib canvas for an intensity histogram
        self.canvas = MplCanvas(self, width=5, height=4)
        main_layout.addWidget(self.canvas)

        self.setLayout(main_layout)

    def on_analyze(self):
        """
        1) Takes the chosen folder (e.g., "cleaned_data")
        2) Gathers class distribution (# images per class)
        3) Loads a random subset of images, calculates average intensities
           and produces an aggregated histogram on the canvas.
        4) Logs the results.
        """
        root_folder = self.folder_combo.currentText().strip()
        if not os.path.isdir(root_folder):
            QMessageBox.warning(self, "Invalid Folder", f"{root_folder} does not exist.")
            return

        self.log_box.clear()
        # Step 1: Classes, distribution
        dist_info = {}
        total_images = 0
        subfolders = ["Training", "Testing"] if "archive" in root_folder or "cleaned" in root_folder or "augmented" in root_folder else [""]

        for subset in subfolders:
            if subset:
                subset_path = os.path.join(root_folder, subset)
                if not os.path.isdir(subset_path):
                    continue
                subset_count = 0
                class_list = [
                    d for d in os.listdir(subset_path)
                    if os.path.isdir(os.path.join(subset_path, d))
                ]
                dist_info[subset] = {}
                for c in class_list:
                    cpath = os.path.join(subset_path, c)
                    n = sum(
                        f.lower().endswith(('.png','.jpg','.jpeg'))
                        for f in os.listdir(cpath)
                    )
                    dist_info[subset][c] = n
                    subset_count += n
                total_images += subset_count
            else:
                # single-level structure, if not using subfolders
                # (e.g. if user picks a random folder)
                pass

        # Log the distribution
        for subset, cdict in dist_info.items():
            self.log_box.append(f"<b>{subset} distribution</b>:")
            ssum = 0
            for cls_, cnt in cdict.items():
                self.log_box.append(f"   {cls_}: {cnt}")
                ssum += cnt
            self.log_box.append(f"   Subtotal: {ssum}\n")
        self.log_box.append(f"Grand total images: {total_images}\n")

        # Step 2: Gather a random sample of images for metadata & histogram
        # We'll sample up to 30 images to keep it quick
        sample_imgs = self._gather_random_images(root_folder, subfolders, max_samples=30)
        if not sample_imgs:
            self.log_box.append("No images found to sample.")
            self.canvas.figure.clear()
            self.canvas.draw()
            return

        intensities = []
        widths = []
        heights = []

        # We'll collect random pixel intensities from each image
        for path in sample_imgs:
            try:
                pil_img = Image.open(path).convert('L')
                arr = np.array(pil_img)
                intensities.append(arr.flatten())  # gather all intensities
                widths.append(pil_img.width)
                heights.append(pil_img.height)
            except Exception as e:
                self.log_box.append(f"[Error reading {path}]: {e}")

        if intensities:
            intensities = np.concatenate(intensities)
            avg_intensity = np.mean(intensities)
            std_intensity = np.std(intensities)
            self.log_box.append(f"Sampled {len(sample_imgs)} images.")
            self.log_box.append(f"Average intensity: {avg_intensity:.2f}, std: {std_intensity:.2f}")
            self.log_box.append(f"Average width x height: {np.mean(widths):.1f} x {np.mean(heights):.1f}")

            # Step 3: Plot histogram
            self.canvas.figure.clear()
            ax = self.canvas.figure.add_subplot(111)
            ax.hist(intensities, bins=50, color='gray')
            ax.set_title("Intensity Distribution (sampled images)")
            ax.set_xlabel("Pixel Intensity [0..255]")
            ax.set_ylabel("Frequency")
            self.canvas.figure.tight_layout()
            self.canvas.draw()
        else:
            self.log_box.append("No intensities could be read.")
            self.canvas.figure.clear()
            self.canvas.draw()

    def _gather_random_images(self, root_folder, subfolders, max_samples=30):
        """
        Return a random subset of up to max_samples image paths.
        """
        all_imgs = []
        for subset in subfolders:
            subpath = os.path.join(root_folder, subset)
            if not os.path.isdir(subpath):
                continue
            class_list = [
                d for d in os.listdir(subpath)
                if os.path.isdir(os.path.join(subpath, d))
            ]
            for c in class_list:
                cpath = os.path.join(subpath, c)
                images = [
                    os.path.join(cpath, f) for f in os.listdir(cpath)
                    if f.lower().endswith(('.png','.jpg','.jpeg'))
                ]
                all_imgs.extend(images)

        random.shuffle(all_imgs)
        return all_imgs[:max_samples]
