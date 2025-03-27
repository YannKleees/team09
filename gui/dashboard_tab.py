import os
import random
import numpy as np
from PIL import Image

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QHBoxLayout,
    QPushButton, QComboBox, QTextEdit, QMessageBox, QSpinBox, QFormLayout
)
from PyQt5.QtCore import Qt
from gui.widgets import MplCanvas

class DashboardTab(QWidget):
    """
    Summaries & analytics for the dataset:
      - Show class distribution
      - Analyze random images for intensity stats
      - Show histogram of intensities
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QVBoxLayout(self)

        # Title / Info
        title_label = QLabel("<h2>Data Overview & Dashboard</h2>")
        title_label.setWordWrap(True)
        main_layout.addWidget(title_label)

        # Group: Folder Selection
        select_group = QGroupBox("Folder Selection & Analysis")
        select_layout = QFormLayout()

        self.folder_combo = QComboBox()
        self.folder_combo.setToolTip("Select which dataset folder to analyze.")
        self.folder_combo.addItems(["data/archive", "cleaned_data", "augmented_data"])

        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(1, 200)
        self.sample_spin.setValue(30)
        self.sample_spin.setToolTip("How many images to sample for intensity stats.")

        self.btn_analyze = QPushButton("Analyze Selected Folder")
        self.btn_analyze.setToolTip("Compute distribution & intensity stats, then plot a histogram.")
        self.btn_analyze.clicked.connect(self.on_analyze)

        select_layout.addRow("Select Folder:", self.folder_combo)
        select_layout.addRow("Images to Sample:", self.sample_spin)
        select_layout.addRow(self.btn_analyze)
        select_group.setLayout(select_layout)

        main_layout.addWidget(select_group)

        # Log area
        self.log_box = QTextEdit()
        self.log_box.setToolTip("Displays distribution stats and intensity details.")
        self.log_box.setReadOnly(True)
        main_layout.addWidget(self.log_box)

        # Canvas for plotting the histogram
        self.canvas = MplCanvas(self, width=5, height=4)
        main_layout.addWidget(self.canvas)

        self.setLayout(main_layout)

    def on_analyze(self):
        """
        1) Takes the chosen folder (e.g., "cleaned_data")
        2) Gathers class distribution (# images per class)
        3) Loads a random subset of images (user-defined count) for intensity stats
        4) Plots a histogram of intensities
        5) Displays results in a log box
        """
        root_folder = self.folder_combo.currentText().strip()
        if not os.path.isdir(root_folder):
            QMessageBox.warning(self, "Invalid Folder", f"Path '{root_folder}' does not exist.")
            return

        self.log_box.clear()

        subfolders = ["Training", "Testing"] if any(x in root_folder for x in ["archive", "cleaned", "augmented"]) else [""]

        # Step 1: Compute distribution
        distribution_info, total_images = self._compute_distribution(root_folder, subfolders)

        # Step 2: Display distribution stats in log box
        if distribution_info:
            self.log_box.append("<b>--- Class Distribution ---</b>\n")
            for subset, class_dict in distribution_info.items():
                self.log_box.append(f"<b>{subset}</b>:")
                sub_sum = 0
                for cls_, cnt in class_dict.items():
                    self.log_box.append(f"   {cls_} => {cnt}")
                    sub_sum += cnt
                self.log_box.append(f"   Subtotal: {sub_sum}\n")
            self.log_box.append(f"<b>Total images across all classes:</b> {total_images}\n")

        # Step 3: Random sample of images
        num_to_sample = self.sample_spin.value()
        sample_imgs = self._gather_random_images(root_folder, subfolders, max_samples=num_to_sample)
        if not sample_imgs:
            self.log_box.append("No images found to sample.")
            self.canvas.figure.clear()
            self.canvas.draw()
            return

        intensities, widths, heights, error_count = self._collect_image_stats(sample_imgs)

        if error_count > 0:
            self.log_box.append(f"<b>There were {error_count} images that failed to load properly.</b>\n")

        # Step 4: If we do have intensities, compute more advanced stats
        if len(intensities) > 0:
            self._display_image_stats(intensities, widths, heights)
            self._plot_histogram(intensities)
        else:
            self.log_box.append("<b>No valid intensities found; cannot plot histogram.</b>")
            self.canvas.figure.clear()
            self.canvas.draw()

    def _compute_distribution(self, root_folder, subfolders):
        """
        Return a nested dict of {subset -> {class_name -> count}} and total_images
        """
        distribution_info = {}
        total_images = 0

        for subset in subfolders:
            subset_path = os.path.join(root_folder, subset)
            if not os.path.isdir(subset_path):
                continue

            class_list = [d for d in os.listdir(subset_path) if os.path.isdir(os.path.join(subset_path, d))]
            distribution_info[subset] = {}
            sub_sum = 0
            for c in class_list:
                cpath = os.path.join(subset_path, c)
                num_images = sum(f.lower().endswith(('.png','.jpg','.jpeg')) for f in os.listdir(cpath))
                distribution_info[subset][c] = num_images
                sub_sum += num_images

            total_images += sub_sum

        return distribution_info, total_images

    def _gather_random_images(self, root_folder, subfolders, max_samples=30):
        """
        Return up to max_samples random images from the given dataset structure.
        """
        all_imgs = []
        for subset in subfolders:
            subset_path = os.path.join(root_folder, subset)
            if not os.path.isdir(subset_path):
                continue
            class_list = [d for d in os.listdir(subset_path) if os.path.isdir(os.path.join(subset_path, d))]
            for c in class_list:
                cpath = os.path.join(subset_path, c)
                images = [os.path.join(cpath, f) for f in os.listdir(cpath)
                          if f.lower().endswith(('.png','.jpg','.jpeg'))]
                all_imgs.extend(images)

        random.shuffle(all_imgs)
        return all_imgs[:max_samples]

    def _collect_image_stats(self, image_paths):
        """
        Return a tuple:
           intensities (list),
           widths (list),
           heights (list),
           error_count (int)
        """
        intensities = []
        widths = []
        heights = []
        error_count = 0

        for path in image_paths:
            try:
                pil_img = Image.open(path).convert('L')
                arr = np.array(pil_img, dtype=np.float32)
                intensities.extend(arr.flatten())  # accumulate intensities
                widths.append(pil_img.width)
                heights.append(pil_img.height)
            except Exception as e:
                error_count += 1

        return intensities, widths, heights, error_count

    def _display_image_stats(self, intensities, widths, heights):
        """
        Compute and display advanced stats: min, max, median, mean, std
        for intensities and dimensions.
        """
        intensities = np.array(intensities)
        widths = np.array(widths, dtype=np.float32)
        heights = np.array(heights, dtype=np.float32)

        # Intensity stats
        i_min, i_max = intensities.min(), intensities.max()
        i_median = np.median(intensities)
        i_mean = intensities.mean()
        i_std = intensities.std()

        # Dimension stats
        w_min, w_max = widths.min(), widths.max()
        w_median = np.median(widths)
        w_mean = widths.mean()
        w_std = widths.std()

        h_min, h_max = heights.min(), heights.max()
        h_median = np.median(heights)
        h_mean = heights.mean()
        h_std = heights.std()

        self.log_box.append("<b>--- Sampled Image Stats ---</b>\n")
        self.log_box.append("<b>Pixel Intensities</b>:")
        self.log_box.append(f"   Min: {i_min:.1f}, Max: {i_max:.1f}")
        self.log_box.append(f"   Median: {i_median:.2f}, Mean: {i_mean:.2f}, Std: {i_std:.2f}\n")

        self.log_box.append("<b>Widths</b>:")
        self.log_box.append(f"   Min: {w_min:.1f}, Max: {w_max:.1f}")
        self.log_box.append(f"   Median: {w_median:.1f}, Mean: {w_mean:.1f}, Std: {w_std:.1f}\n")

        self.log_box.append("<b>Heights</b>:")
        self.log_box.append(f"   Min: {h_min:.1f}, Max: {h_max:.1f}")
        self.log_box.append(f"   Median: {h_median:.1f}, Mean: {h_mean:.1f}, Std: {h_std:.1f}\n")

    def _plot_histogram(self, intensities):
        """
        Plot a histogram of pixel intensities onto the MplCanvas.
        """
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)
        ax.hist(intensities, bins=50, color='#0078d7', alpha=0.7)
        ax.set_title("Distribution of Sampled Pixel Intensities")
        ax.set_xlabel("Pixel Intensity [0..255]")
        ax.set_ylabel("Count")
        self.canvas.figure.tight_layout()
        self.canvas.draw()
