# File: gui/advanced_eda_tab.py

import os
import random
import numpy as np
from PIL import Image

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QPushButton,
    QTextEdit, QMessageBox, QHBoxLayout, QSlider, QScrollArea,
    QComboBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from gui.widgets import MplCanvas
#
# IMPORTANT: Import your advanced EDA logic from logic/advanced_eda
#
from logic.advanced_eda import (
    gather_random_images,
    collect_resolutions,
    detect_poor_quality,
    histogram_equalize,
    detect_extended_outliers,
    gather_subset_resolutions
)


class AdvancedEdaTab(QWidget):
    """
    A separate tab dedicated to 'Advanced EDA':
      - resolution checks, poor-quality scans,
      - histogram eq example,
      - a gamma demonstration,
      - plus multi-subset comparison & extended outlier detection.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # 1) Create the main layout for this entire tab
        main_layout = QVBoxLayout(self)

        # 2) We'll place everything inside a scrollable widget
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)

        # 3) Create a child widget that holds all content in a vertical layout
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(10, 10, 10, 10)

        # Title / Info
        title_label = QLabel("<h2>Advanced EDA</h2>")
        title_label.setWordWrap(True)
        content_layout.addWidget(title_label)

        info_label = QLabel(
            "This tab demonstrates resolution checks, outlier detection, "
            "histogram equalization, a gamma transformation demo, "
            "and some multi-subset comparisons."
        )
        info_label.setWordWrap(True)
        content_layout.addWidget(info_label)

        # Button to run analysis
        self.eda_btn = QPushButton("Run Basic Analysis")
        self.eda_btn.clicked.connect(self.on_run_advanced_eda)
        content_layout.addWidget(self.eda_btn)

        # -----------------------------------------------------------
        # Group 1: Resolution Stats & Plots
        # -----------------------------------------------------------
        self.res_group = QGroupBox("Resolution Statistics")
        res_layout = QVBoxLayout()
        self.res_text = QTextEdit()
        self.res_text.setReadOnly(True)
        res_layout.addWidget(self.res_text)

        # A canvas + navigation toolbar
        self.res_canvas = MplCanvas(self, width=5, height=4)
        res_toolbar = NavigationToolbar(self.res_canvas, self)
        res_layout.addWidget(self.res_canvas)
        res_layout.addWidget(res_toolbar)

        self.res_group.setLayout(res_layout)
        content_layout.addWidget(self.res_group)

        # -----------------------------------------------------------
        # Group 2: Histogram Equalization Demo
        # -----------------------------------------------------------
        self.eq_group = QGroupBox("Histogram Equalization Demo")
        eq_layout = QVBoxLayout()

        # Another canvas for the original vs. equalized image
        self.eq_canvas = MplCanvas(self, width=5, height=4)
        eq_toolbar = NavigationToolbar(self.eq_canvas, self)
        eq_layout.addWidget(self.eq_canvas)
        eq_layout.addWidget(eq_toolbar)
        self.eq_group.setLayout(eq_layout)
        content_layout.addWidget(self.eq_group)

        # -----------------------------------------------------------
        # Group 3: Gamma Demo
        # -----------------------------------------------------------
        self.gamma_group = QGroupBox("Gamma Demo")
        gamma_layout = QVBoxLayout()

        gamma_info = QLabel(
            "Adjust the slider to apply gamma correction to a sample image. "
            "Lower values (<1.0) lighten shadows; higher values (>1.0) darken shadows."
        )
        gamma_info.setWordWrap(True)
        gamma_layout.addWidget(gamma_info)

        # A slider for gamma in [0.5..2.0], with 100 steps
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(50, 200)  # 0.5..2.0 (scaled by 100)
        self.gamma_slider.setValue(100)      # 1.0 by default
        self.gamma_slider.valueChanged.connect(self.on_gamma_slider_changed)
        gamma_layout.addWidget(self.gamma_slider)

        # A canvas + toolbar
        self.gamma_canvas = MplCanvas(self, width=5, height=4)
        gamma_toolbar = NavigationToolbar(self.gamma_canvas, self)
        gamma_layout.addWidget(self.gamma_canvas)
        gamma_layout.addWidget(gamma_toolbar)

        self.gamma_group.setLayout(gamma_layout)
        content_layout.addWidget(self.gamma_group)

        # -----------------------------------------------------------
        # Group 4: Extended Outlier Detection
        # -----------------------------------------------------------
        self.outlier_group = QGroupBox("Extended Outlier Detection")
        outlier_layout = QVBoxLayout()

        self.ext_outlier_btn = QPushButton("Run Extended Outlier Detection")
        self.ext_outlier_btn.setToolTip("Check brightness thresholds, dimension, and variance.")
        self.ext_outlier_btn.clicked.connect(self.on_extended_outlier_detection)
        outlier_layout.addWidget(self.ext_outlier_btn)

        self.outlier_text = QTextEdit()
        self.outlier_text.setReadOnly(True)
        outlier_layout.addWidget(self.outlier_text)

        self.outlier_group.setLayout(outlier_layout)
        content_layout.addWidget(self.outlier_group)

        # -----------------------------------------------------------
        # Group 5: Multi-Subset Comparison
        # -----------------------------------------------------------
        self.subset_group = QGroupBox("Multi-Subset Comparison")
        subset_layout = QVBoxLayout()

        info_sub = QLabel(
            "Compare resolution distributions across subsets. "
            "Pick 'Training', 'Testing', or 'Both'."
        )
        info_sub.setWordWrap(True)
        subset_layout.addWidget(info_sub)

        # A combo box for picking which subsets to compare
        self.subset_combo = QComboBox()
        self.subset_combo.addItems(["Training only", "Testing only", "Both"])
        subset_layout.addWidget(self.subset_combo)

        self.compare_btn = QPushButton("Compare Subsets")
        self.compare_btn.clicked.connect(self.on_compare_subsets)
        subset_layout.addWidget(self.compare_btn)

        # Canvas for multi-subset
        self.subset_canvas = MplCanvas(self, width=5, height=4)
        self.subset_toolbar = NavigationToolbar(self.subset_canvas, self)
        subset_layout.addWidget(self.subset_canvas)
        subset_layout.addWidget(self.subset_toolbar)

        self.subset_group.setLayout(subset_layout)
        content_layout.addWidget(self.subset_group)

        # -----------------------------------------------------------
        # Text area for logs about poor-quality
        # -----------------------------------------------------------
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        content_layout.addWidget(self.log_area)

        # A simple note about the assumed folder
        note_label = QLabel(
            "<i>Note: This example currently looks in <b>data/archive</b> "
            "for images. Adjust the code if needed, or let the user pick a path.</i>"
        )
        note_label.setWordWrap(True)
        content_layout.addWidget(note_label)

        # Finally, set the content widget into the scroll area
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)

        self.setLayout(main_layout)

        # We'll hold onto a reference to the sample image for gamma demos
        self.gamma_sample_img = None

    def on_run_advanced_eda(self):
        """
        1) Scans up to 200 images in data/archive (Training, Testing) to check:
           - resolution min/max/mean
           - poor-quality detection
           - then plot resolution stats
           - show histogram equalization example
           - pick an image for gamma demos
        """
        root_folder = "data/archive"  # Hardcoded example
        self.log_area.clear()
        self.res_text.clear()

        if not os.path.isdir(root_folder):
            QMessageBox.warning(self, "Invalid Folder", f"Path '{root_folder}' does not exist.")
            return

        # Gather images
        subfolders = ["Training", "Testing"]
        all_imgs = gather_random_images(root_folder, subfolders, max_samples=200)
        if not all_imgs:
            self.log_area.append("No images found for Advanced EDA in data/archive.")
            return

        # 1) Summarize resolution
        widths, heights, error_count = collect_resolutions(all_imgs)
        if error_count > 0:
            self.log_area.append(f"Failed to read {error_count} images.\n")

        if widths and heights:
            w_min, w_max = min(widths), max(widths)
            h_min, h_max = min(heights), max(heights)
            w_mean, h_mean = np.mean(widths), np.mean(heights)
            self.res_text.append("<b>Resolution Summary (Sample up to 200 imgs):</b>")
            self.res_text.append(
                f"Width:  min={w_min}, max={w_max}, mean={w_mean:.1f}\n"
                f"Height: min={h_min}, max={h_max}, mean={h_mean:.1f}\n"
            )
        else:
            self.res_text.append("No valid images for resolution stats.\n")

        # 2) Identify "poor-quality" scans
        poor_quality = detect_poor_quality(all_imgs)
        if poor_quality:
            self.log_area.append(
                f"Detected {len(poor_quality)} possible poor-quality scans (low-res or low-variance).\n"
            )
        else:
            self.log_area.append("No poor-quality scans found.\n")

        # 3) Plot resolution stats (histograms, etc.) on self.res_canvas
        self._plot_resolution_distributions(widths, heights)

        # 4) Show a random image side-by-side with histogram equalization
        self._show_histogram_eq_demo(all_imgs)

        # 5) Also pick an image for gamma demonstration
        if all_imgs:
            self.gamma_sample_img = random.choice(all_imgs)
            self._update_gamma_demo()  # Show initial gamma=1.0

    def on_gamma_slider_changed(self, value):
        """
        Called whenever user moves the gamma slider. 'value' is 50..200 => 0.5..2.0
        """
        if self.gamma_sample_img:
            self._update_gamma_demo()

    def _update_gamma_demo(self):
        """
        Read gamma_sample_img from disk and apply gamma. Then show in gamma_canvas.
        """
        self.gamma_canvas.figure.clear()
        if not self.gamma_sample_img:
            return

        gamma_val = self.gamma_slider.value() / 100.0
        try:
            with Image.open(self.gamma_sample_img).convert('L') as im:
                fig = self.gamma_canvas.figure
                ax = fig.subplots(1, 1)
                ax.set_title(f"Gamma = {gamma_val:.2f}")

                # Build LUT for gamma correction
                lut = [pow(i / 255.0, gamma_val) * 255.0 for i in range(256)]
                out = im.point(lut)

                ax.imshow(out, cmap='gray')
                ax.axis('off')

                fig.tight_layout()
                self.gamma_canvas.draw()

        except Exception as e:
            self.log_area.append(f"Gamma Demo error for {self.gamma_sample_img}: {e}\n")

    def _plot_resolution_distributions(self, widths, heights):
        """Plot histograms/scatter of widths and heights on self.res_canvas."""
        self.res_canvas.figure.clear()
        fig = self.res_canvas.figure

        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        ax1.hist(widths, bins=20, color="blue", alpha=0.7)
        ax1.set_title("Widths")
        ax1.set_xlabel("pixels")
        ax1.set_ylabel("count")

        ax2.hist(heights, bins=20, color="green", alpha=0.7)
        ax2.set_title("Heights")
        ax2.set_xlabel("pixels")

        ax3.scatter(widths, heights, alpha=0.3)
        ax3.set_title("Width vs. Height")
        ax3.set_xlabel("width")
        ax3.set_ylabel("height")

        fig.tight_layout()
        self.res_canvas.draw()

    def _show_histogram_eq_demo(self, image_paths):
        """Pick a random image, display original vs. histogram-equalized."""
        self.eq_canvas.figure.clear()
        if not image_paths:
            return

        sample_path = random.choice(image_paths)
        try:
            with Image.open(sample_path).convert('L') as im:
                fig = self.eq_canvas.figure
                ax1, ax2 = fig.subplots(1, 2)

                ax1.imshow(im, cmap='gray')
                ax1.set_title("Original")
                ax1.axis('off')

                eq_img = histogram_equalize(im)
                ax2.imshow(eq_img, cmap='gray')
                ax2.set_title("Histogram Equalized")
                ax2.axis('off')

                fig.tight_layout()
                self.eq_canvas.draw()

                self.log_area.append(f"Histogram EQ demo image:\n{sample_path}\n")

        except Exception as e:
            self.log_area.append(f"Failed to apply histogram equalization:\n{e}\n")

    # -----------------------------------------------------------
    # Extended outlier detection
    # -----------------------------------------------------------
    def on_extended_outlier_detection(self):
        """
        Demo of detecting "extended" outliers based on brightness,
        dimension threshold, etc.
        """
        root_folder = "data/archive"  # Hardcoded for demonstration
        if not os.path.isdir(root_folder):
            QMessageBox.warning(self, "Invalid Folder", f"Path '{root_folder}' does not exist.")
            return

        self.outlier_text.clear()
        subfolders = ["Training", "Testing"]
        all_imgs = gather_random_images(root_folder, subfolders, max_samples=200)
        if not all_imgs:
            self.outlier_text.append("No images found in data/archive for outlier detection.")
            return

        # run extended detection
        extended_out = detect_extended_outliers(
            all_imgs,
            brightness_low=15.0,
            brightness_high=240.0,
            dimension_threshold=50,
            var_threshold=5.0
        )
        if extended_out:
            self.outlier_text.append(
                f"Found {len(extended_out)} extended outliers among ~200 samples.\n"
            )
            for path in extended_out[:10]:  # only show first 10
                self.outlier_text.append(path)
        else:
            self.outlier_text.append("No extended outliers found among sampled images.")

    # -----------------------------------------------------------
    # Multi-subset comparison
    # -----------------------------------------------------------
    def on_compare_subsets(self):
        """
        Gather resolution data for 'Training' and/or 'Testing' to plot side by side.
        """
        root_folder = "data/archive"
        if not os.path.isdir(root_folder):
            QMessageBox.warning(self, "Invalid Folder", f"Path '{root_folder}' not found.")
            return

        choice = self.subset_combo.currentText()
        if choice == "Training only":
            subsets = ["Training"]
        elif choice == "Testing only":
            subsets = ["Testing"]
        else:
            subsets = ["Training", "Testing"]

        subset_w, subset_h = gather_subset_resolutions(root_folder, subsets)
        self._plot_subset_comparison(subset_w, subset_h)

    def _plot_subset_comparison(self, subset_w, subset_h):
        """
        subset_w, subset_h are dicts:
          e.g. { "Training": [...], "Testing": [...] }
        We'll do a side-by-side histogram for each subset's widths & heights.
        """
        self.subset_canvas.figure.clear()
        fig = self.subset_canvas.figure

        # Suppose we do 2 rows: top => widths, bottom => heights
        # If there's only 1 subset, we show just 1 histogram each row
        # If 2 subsets, we show 2 hist overlaps or subplots
        subsets = sorted(subset_w.keys())

        if len(subsets) == 1:
            # single subset => just 2 subplots
            ax1 = fig.add_subplot(2, 1, 1)
            wdata = subset_w[subsets[0]]
            ax1.hist(wdata, bins=30, alpha=0.7, color='blue')
            ax1.set_title(f"Widths - {subsets[0]}")

            ax2 = fig.add_subplot(2, 1, 2)
            hdata = subset_h[subsets[0]]
            ax2.hist(hdata, bins=30, alpha=0.7, color='green')
            ax2.set_title(f"Heights - {subsets[0]}")

        else:
            # 2 subsets => either overlap them or do side-by-side subplots
            # For simplicity, let's do an overlapped histogram in each row
            ax1 = fig.add_subplot(2, 1, 1)
            for sub in subsets:
                wdata = subset_w[sub]
                ax1.hist(wdata, bins=30, alpha=0.5, label=sub)
            ax1.set_title("Widths (overlapped)")
            ax1.legend()

            ax2 = fig.add_subplot(2, 1, 2)
            for sub in subsets:
                hdata = subset_h[sub]
                ax2.hist(hdata, bins=30, alpha=0.5, label=sub)
            ax2.set_title("Heights (overlapped)")
            ax2.legend()

        fig.tight_layout()
        self.subset_canvas.draw()
