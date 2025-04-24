import os
import random
import numpy as np
from PIL import Image

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QFormLayout,
    QPushButton, QSpinBox, QTextEdit, QMessageBox, QScrollArea,
    QSizePolicy, QHBoxLayout
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from gui.widgets import MplCanvas
from src.transforms_config import create_transform


class DashboardTab(QWidget):
    """
    Basic Data overview & dashboard:
      - Summarize distribution of classes
      - Sample intensities and plot a histogram
      - Compare normalization options
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

            /* Larger, more stylish buttons */
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
                background-color: #005CB5; /* Slightly darker on hover */
            }
            QPushButton:pressed {
                background-color: #004C94; /* Even darker when pressed */
            }
            QPushButton:disabled {
                background-color: #9FBADF; /* Desaturated look when disabled */
            }

            QSpinBox {
                font-size: 10pt;
                padding: 4px;
            }

            QTextEdit {
                background-color: #FFFFFF;
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                font-size: 10pt;
            }

            QScrollArea {
                border: none;
            }
        """)

        # Give this widget an object name so the stylesheet can target it
        self.setObjectName("DashboardTab")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Scrollable container
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)

        # Content widget and layout
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(15)

        # Title and description
        title_label = QLabel("<h2 style='color: #0078D7; margin-bottom: 0;'>Data Overview & Dashboard</h2>")
        title_label.setWordWrap(True)
        content_layout.addWidget(title_label)

        desc_label = QLabel(
            "Analyze class distribution, sample intensities, and view a histogram. "
            "You can also compare different normalization methods. "
            "Select a folder and click 'Analyze Folder' to begin."
        )
        desc_label.setWordWrap(True)
        content_layout.addWidget(desc_label)

        # Folder Selection & Analysis group
        select_group = QGroupBox("Folder Selection & Analysis")
        form_layout = QFormLayout()
        form_layout.setSpacing(10)

        self.folder_edit = QLabel("data/archive")
        self.folder_edit.setToolTip("Path containing subfolders like 'Training' & 'Testing'.")
        self.folder_edit.setStyleSheet("font-weight: 600; color: #444444;")

        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(1, 200)
        self.sample_spin.setValue(30)
        self.sample_spin.setToolTip("How many images to sample for intensity stats.")

        # Make a horizontal layout for the buttons so they can sit side by side
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        self.btn_analyze = QPushButton("Analyze Folder")
        self.btn_analyze.setToolTip("Compute distribution & intensity stats; then plot a histogram.")
        # Expand horizontally
        self.btn_analyze.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_analyze.clicked.connect(self.on_analyze)

        self.btn_compare_norm = QPushButton("Compare Normalizations")
        self.btn_compare_norm.setToolTip("Open a dialog to compare min-max vs. z-score normalization.")
        # Also expand
        self.btn_compare_norm.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_compare_norm.clicked.connect(self.on_compare_normalization)

        button_layout.addWidget(self.btn_analyze)
        button_layout.addWidget(self.btn_compare_norm)

        form_layout.addRow("Dataset Folder:", self.folder_edit)
        form_layout.addRow("Images to Sample:", self.sample_spin)
        # Buttons in a single row
        form_layout.addRow(button_layout)

        select_group.setLayout(form_layout)
        content_layout.addWidget(select_group)

        # Log box
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setToolTip("Displays distribution stats and intensity details.")
        self.log_box.setMinimumHeight(150)
        content_layout.addWidget(self.log_box)

        # Plot area
        plot_group = QGroupBox("Histogram of Sampled Intensities")
        plot_layout = QVBoxLayout()

        self.canvas = MplCanvas(self, width=5, height=4)
        self.canvas_toolbar = NavigationToolbar(self.canvas, self)

        plot_layout.addWidget(self.canvas)
        plot_layout.addWidget(self.canvas_toolbar)
        plot_group.setLayout(plot_layout)
        content_layout.addWidget(plot_group)

        # Put content into the scroll area
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)

    def on_analyze(self):
        root_folder = self.folder_edit.text().strip()
        if not os.path.isdir(root_folder):
            QMessageBox.warning(self, "Invalid Folder", f"Path '{root_folder}' does not exist.")
            return

        self.log_box.clear()

        subfolders = (
            ["Training", "Testing"]
            if any(x in root_folder for x in ["archive", "cleaned", "augmented"])
            else [""]
        )

        # 1) Compute distribution
        distribution_info, total_images = self._compute_distribution(root_folder, subfolders)

        # 2) Log distribution
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

        # 3) Gather random sample
        num_to_sample = self.sample_spin.value()
        sample_imgs = self._gather_random_images(root_folder, subfolders, max_samples=num_to_sample)
        if not sample_imgs:
            self.log_box.append("No images found to sample in this folder.")
            self.canvas.figure.clear()
            self.canvas.draw()
            return

        intensities = []
        error_count = 0
        for path in sample_imgs:
            try:
                pil_img = Image.open(path).convert('L')
                arr = np.array(pil_img, dtype=np.float32)
                intensities.extend(arr.flatten())
            except Exception:
                error_count += 1

        if error_count > 0:
            self.log_box.append(f"<b>Failed to read {error_count} images.</b>\n")

        if len(intensities) == 0:
            self.log_box.append("<b>No valid intensities found; cannot plot histogram.</b>")
            self.canvas.figure.clear()
            self.canvas.draw()
            return

        # 4) Stats
        intensities_np = np.array(intensities, dtype=np.float32)
        i_min, i_max = intensities_np.min(), intensities_np.max()
        i_mean = intensities_np.mean()
        i_med = np.median(intensities_np)
        i_std = intensities_np.std()

        self.log_box.append("<b>--- Sampled Image Stats ---</b>\n")
        self.log_box.append(f"Min: {i_min:.1f}, Max: {i_max:.1f}")
        self.log_box.append(f"Mean: {i_mean:.2f}, Median: {i_med:.2f}, Std: {i_std:.2f}\n")

        # 5) Plot histogram
        self.canvas.figure.clear()

        # Match figure background to the tab’s background
        self.canvas.figure.patch.set_facecolor('#F7F7F7')

        ax = self.canvas.figure.add_subplot(111)
        ax.hist(intensities_np, bins=50, color='#0078D7', alpha=0.7)

        ax.set_title("Distribution of Sampled Pixel Intensities", color='#333333')
        ax.set_xlabel("Pixel Intensity", color='#333333')
        ax.set_ylabel("Count", color='#333333')

        # Start axes at zero (optional)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        ax.set_facecolor('#FFFFFF')
        ax.tick_params(axis='x', colors='#333333')
        ax.tick_params(axis='y', colors='#333333')

        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def on_compare_normalization(self):
        from gui.normalization_comparison_dialog import NormalizationComparisonDialog

        root_folder = self.folder_edit.text().strip()
        if not os.path.isdir(root_folder):
            QMessageBox.warning(self, "Invalid Folder", f"Path '{root_folder}' does not exist.")
            return

        dialog = NormalizationComparisonDialog(root_folder, parent=self)
        dialog.exec_()

    def _compute_distribution(self, root_folder, subfolders):
        distribution_info = {}
        total_images = 0
        for subset in subfolders:
            subset_path = os.path.join(root_folder, subset)
            if not os.path.isdir(subset_path):
                continue
            class_list = [
                d for d in os.listdir(subset_path)
                if os.path.isdir(os.path.join(subset_path, d))
            ]
            distribution_info[subset] = {}
            sub_sum = 0
            for c in class_list:
                cpath = os.path.join(subset_path, c)
                num_images = sum(
                    f.lower().endswith(('.png', '.jpg', '.jpeg'))
                    for f in os.listdir(cpath)
                )
                distribution_info[subset][c] = num_images
                sub_sum += num_images
            total_images += sub_sum
        return distribution_info, total_images

    def _gather_random_images(self, root_folder, subfolders, max_samples=30):
        all_imgs = []
        for subset in subfolders:
            subset_path = os.path.join(root_folder, subset)
            if not os.path.isdir(subset_path):
                continue
            class_list = [
                d for d in os.listdir(subset_path)
                if os.path.isdir(os.path.join(subset_path, d))
            ]
            for c in class_list:
                cpath = os.path.join(subset_path, c)
                images = [
                    os.path.join(cpath, f)
                    for f in os.listdir(cpath)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
                all_imgs.extend(images)

        random.shuffle(all_imgs)
        return all_imgs[:max_samples]
