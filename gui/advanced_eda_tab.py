import os
import random
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QPushButton,
    QTextEdit, QMessageBox, QHBoxLayout, QSlider,
    QScrollArea, QFormLayout
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from gui.widgets import MplCanvas


class AdvancedEdaTab(QWidget):
    """
    A separate tab dedicated to 'Advanced EDA':
      1) Auto-detect image resolution & find min/max/mean.
      2) Identify "poor-quality" scans (toy example).
      3) Plot resolution distribution (width/height) in a dedicated canvas.
      4) Show an example of histogram equalization on a random sample in a second canvas.
      5) NEW: Gamma transform demo with a slider.

    Now placed inside a QScrollArea so it's scrollable, and each Matplotlib
    canvas has a navigation toolbar for zoom & pan.
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
            "histogram equalization, and now a gamma demonstration."
        )
        info_label.setWordWrap(True)
        content_layout.addWidget(info_label)

        # Button to run analysis
        self.eda_btn = QPushButton("Run Advanced EDA")
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
            "Adjust the slider to apply gamma correction to the same sample image. "
            "Lower values (<1.0) lighten shadows; higher values (>1.0) darken shadows."
        )
        gamma_info.setWordWrap(True)
        gamma_layout.addWidget(gamma_info)

        # A slider for gamma in [0.5..2.0], with 100 steps
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(50, 200)  # 0.5 .. 2.0 (scaled by 100)
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
        # Text area for logs about poor-quality
        # -----------------------------------------------------------
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        content_layout.addWidget(self.log_area)

        # A simple note about the assumed folder
        note_label = QLabel(
            "<i>Note: This example currently looks in <b>data/archive</b> "
            "for images. Adjust the code if needed, or integrate with an editable path.</i>"
        )
        note_label.setWordWrap(True)
        content_layout.addWidget(note_label)

        # Finally, set the content widget into the scroll area,
        # and put the scroll area in our main_layout
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
           - then show histogram equalization example
           - pick an image for gamma demos
        """
        root_folder = "data/archive"  # Hardcoded example; adapt if needed
        self.log_area.clear()
        self.res_text.clear()

        if not os.path.isdir(root_folder):
            QMessageBox.warning(self, "Invalid Folder", f"Path '{root_folder}' does not exist.")
            return

        # Gather images
        subfolders = ["Training", "Testing"]
        all_imgs = self._gather_random_images(root_folder, subfolders, max_samples=200)
        if not all_imgs:
            self.log_area.append("No images found for Advanced EDA in data/archive.")
            return

        # 1) Summarize resolution
        widths, heights, error_count = self._collect_resolutions(all_imgs)

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
        poor_quality = self._detect_poor_quality(all_imgs)
        if poor_quality:
            self.log_area.append(
                f"Detected {len(poor_quality)} possible poor-quality scans (low-res or low-variance).\n"
            )
        else:
            self.log_area.append("No poor-quality scans found (based on simple heuristics).\n")

        # 3) Plot resolution stats (histograms, etc.) on self.res_canvas
        self._plot_resolution_distributions(widths, heights)

        # 4) Show a random image side-by-side with histogram equalization
        self._show_histogram_eq_demo(all_imgs)

        # 5) Also pick an image for gamma demonstration
        if all_imgs:
            self.gamma_sample_img = random.choice(all_imgs)
            self._update_gamma_demo()  # Show initial gamma=1.0

    # --------------------------------------------------------------------------
    # GAMMA DEMO
    # --------------------------------------------------------------------------
    def on_gamma_slider_changed(self, value):
        """
        Called whenever user moves the gamma slider. 'value' is from 50..200 => 0.5..2.0
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

        # Convert slider value (50..200) => gamma in [0.5 .. 2.0]
        gamma_val = self.gamma_slider.value() / 100.0

        try:
            with Image.open(self.gamma_sample_img).convert('L') as im:
                fig = self.gamma_canvas.figure
                ax = fig.subplots(1, 1)
                ax.set_title(f"Gamma = {gamma_val:.2f}")

                # For gamma correction via .point():
                lut = []
                for i in range(256):
                    lut.append(pow(i/255.0, gamma_val) * 255.0)
                out = im.point(lut)

                ax.imshow(out, cmap='gray')
                ax.axis('off')

                fig.tight_layout()
                self.gamma_canvas.draw()

        except Exception as e:
            self.log_area.append(f"Gamma Demo error for {self.gamma_sample_img}: {e}\n")

    # --------------------------------------------------------------------------
    # Helper methods
    # --------------------------------------------------------------------------
    def _gather_random_images(self, root_folder, subfolders, max_samples=200):
        """Collect up to 'max_samples' images (png/jpg) from each subfolder/class."""
        imgs = []
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
                found = [
                    os.path.join(cpath, f)
                    for f in os.listdir(cpath)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
                imgs.extend(found)
        random.shuffle(imgs)
        return imgs[:max_samples]

    def _collect_resolutions(self, image_paths):
        """Return lists of widths, heights, plus a count of read errors."""
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

    def _detect_poor_quality(self, image_paths):
        """Return a list of poor-quality image paths (toy check)."""
        poor_quality = []
        for path in image_paths:
            try:
                with Image.open(path) as im:
                    w, h = im.size
                    # Very small dimension
                    if w < 50 or h < 50:
                        poor_quality.append(path)
                        continue
                    # Very low variance => potential outlier
                    arr = np.array(im.convert('L'), dtype=np.float32)
                    if arr.size > 0 and arr.std() < 5.0:
                        poor_quality.append(path)
            except:
                pass
        return poor_quality

    def _plot_resolution_distributions(self, widths, heights):
        """Plot histograms/scatter of widths and heights on self.res_canvas."""
        self.res_canvas.figure.clear()
        fig = self.res_canvas.figure

        # e.g. 1 row x 3 columns: width hist, height hist, width-vs-height scatter
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        # Width histogram
        ax1.hist(widths, bins=20, color="blue", alpha=0.7)
        ax1.set_title("Widths")
        ax1.set_xlabel("pixels")
        ax1.set_ylabel("count")

        # Height histogram
        ax2.hist(heights, bins=20, color="green", alpha=0.7)
        ax2.set_title("Heights")
        ax2.set_xlabel("pixels")

        # Scatter: width vs. height
        ax3.scatter(widths, heights, alpha=0.3)
        ax3.set_title("Width vs. Height")
        ax3.set_xlabel("width")
        ax3.set_ylabel("height")

        fig.tight_layout()
        self.res_canvas.draw()

    def _show_histogram_eq_demo(self, image_paths):
        """Pick a random image, display original vs. histogram-equalized in eq_canvas."""
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

                eq_img = ImageOps.equalize(im)
                ax2.imshow(eq_img, cmap='gray')
                ax2.set_title("Histogram Equalized")
                ax2.axis('off')

                fig.tight_layout()
                self.eq_canvas.draw()

                # Log which image we used
                self.log_area.append(
                    f"Histogram EQ demo image:\n{sample_path}\n"
                )
        except Exception as e:
            self.log_area.append(f"Failed to apply histogram equalization on sample:\n{e}\n")
