# gui/info_tab.py

import os
import random
import numpy as np
from PIL import Image

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QPushButton,
    QTextEdit, QMessageBox, QHBoxLayout, QSlider, QScrollArea,
    QComboBox, QLineEdit, QSpinBox, QFileDialog, QFormLayout
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from gui.widgets import MplCanvas

from logic.advanced_eda import (
    gather_random_images,
    collect_resolutions,
    histogram_equalize,
    gather_subset_resolutions
)

class InfoTab(QWidget):
    """
    An “ultimate guide” tab that:
      1) Explains each application tab (Preprocessing, etc.).
      2) Demonstrates how they work with small interactive demos:
         - resolution stats
         - histogram equalization
         - gamma correction
         - subset comparisons
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        main_layout = QVBoxLayout(self)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(12)

        title_label = QLabel("<h2>Ultimate Guide & Demonstrations</h2>")
        title_label.setWordWrap(True)
        content_layout.addWidget(title_label)

        intro_label = QLabel(
            "This tab provides an all-in-one overview of the other tabs, "
            "plus interactive EDA demos to illustrate how your data changes "
            "during Preprocessing and more."
        )
        intro_label.setWordWrap(True)
        content_layout.addWidget(intro_label)

        overview_group = QGroupBox("Overview of Other Tabs")
        overview_layout = QVBoxLayout(overview_group)

        overview_layout.addWidget(self._make_label(
            "<b>Dashboard Tab:</b> Offers general data overviews—like class distributions or basic stats."
        ))
        overview_layout.addWidget(self._make_label(
            "<b>Info Tab (current):</b> This instructive tab with live EDA demos."
        ))
        overview_layout.addWidget(self._make_label(
            "<b>Preprocessing Tab:</b> Cleans raw data, removes duplicates, applies blur/sharpen, histogram eq, etc."
        ))
        overview_layout.addWidget(self._make_label(
            "<b>Augmentation Tab:</b> Generates synthetic images (rotate, flip, brightness, etc.), optionally balancing classes."
        ))
        overview_layout.addWidget(self._make_label(
            "<b>Comparison Tab:</b> Let’s you see the same file in raw, cleaned, or augmented sets side by side."
        ))
        # Removed the HPC line from the overview
        overview_layout.addWidget(self._make_label(
            "<b>Inference Tab:</b> Loads a final trained model to classify a single image."
        ))

        content_layout.addWidget(overview_group)

        controls_box = QGroupBox("Analysis Controls & Settings")
        controls_form = QFormLayout(controls_box)

        self.root_edit = QLineEdit("data/archive")
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._on_browse_folder)
        h_line = QHBoxLayout()
        h_line.addWidget(self.root_edit)
        h_line.addWidget(btn_browse)
        controls_form.addRow("Root Folder:", h_line)

        self.max_images_spin = QSpinBox()
        self.max_images_spin.setRange(1, 1000)
        self.max_images_spin.setValue(200)
        controls_form.addRow("Max Images to Sample:", self.max_images_spin)


        self.res_bin_slider = QSlider(Qt.Horizontal)
        self.res_bin_slider.setRange(5, 50)
        self.res_bin_slider.setValue(20)
        controls_form.addRow("Resolution Histogram Bins:", self.res_bin_slider)

        self.eq_img_edit = QLineEdit()
        self.eq_img_edit.setPlaceholderText("Optional image for histogram eq; or leave blank for random.")
        eq_btn = QPushButton("Pick...")
        eq_btn.clicked.connect(self._on_pick_eq_image)
        eq_h = QHBoxLayout()
        eq_h.addWidget(self.eq_img_edit)
        eq_h.addWidget(eq_btn)
        controls_form.addRow("Histogram EQ Image:", eq_h)

        self.run_demos_btn = QPushButton("Run Demos")
        self.run_demos_btn.clicked.connect(self.on_run_demos)
        controls_form.addRow(self.run_demos_btn)

        content_layout.addWidget(controls_box)

        self.res_group = QGroupBox("Resolution Demo")
        res_layout = QVBoxLayout(self.res_group)

        self.res_text = QTextEdit()
        self.res_text.setReadOnly(True)
        res_layout.addWidget(self.res_text)

        self.res_canvas = MplCanvas(self, width=7, height=5)  # bigger chart
        self.res_toolbar = NavigationToolbar(self.res_canvas, self)
        res_layout.addWidget(self.res_canvas)
        res_layout.addWidget(self.res_toolbar)

        content_layout.addWidget(self.res_group)

        self.eq_group = QGroupBox("Histogram Equalization Demo")
        eq_layout = QVBoxLayout(self.eq_group)

        eq_label = QLabel("Shows a PIL grayscale image (left) and the CLAHE-equalized version (right).")
        eq_label.setWordWrap(True)
        eq_layout.addWidget(eq_label)

        self.eq_canvas = MplCanvas(self, width=7, height=5)
        self.eq_toolbar = NavigationToolbar(self.eq_canvas, self)
        eq_layout.addWidget(self.eq_canvas)
        eq_layout.addWidget(self.eq_toolbar)

        content_layout.addWidget(self.eq_group)

        self.gamma_group = QGroupBox("Gamma Correction Demo")
        gamma_layout = QVBoxLayout(self.gamma_group)

        gamma_label = QLabel(
            "Move the slider to lighten/darken the image. (<1 => lighten, >1 => darken). "
            "Similar transformations appear in the Augmentation Tab."
        )
        gamma_label.setWordWrap(True)
        gamma_layout.addWidget(gamma_label)

        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(50, 200)
        self.gamma_slider.setValue(100)
        self.gamma_slider.valueChanged.connect(self.on_gamma_slider_changed)
        gamma_layout.addWidget(self.gamma_slider)

        self.gamma_canvas = MplCanvas(self, width=7, height=5)
        self.gamma_toolbar = NavigationToolbar(self.gamma_canvas, self)
        gamma_layout.addWidget(self.gamma_canvas)
        gamma_layout.addWidget(self.gamma_toolbar)

        content_layout.addWidget(self.gamma_group)

        self.subset_group = QGroupBox("Subset Comparison Demo")
        subset_layout = QVBoxLayout(self.subset_group)

        subset_label = QLabel(
            "Compare resolution distributions across subsets (Training, Testing, Both)."
        )
        subset_label.setWordWrap(True)
        subset_layout.addWidget(subset_label)

        self.subset_combo = QComboBox()
        self.subset_combo.addItems(["Training only", "Testing only", "Both"])
        subset_layout.addWidget(self.subset_combo)

        self.subset_btn = QPushButton("Compare Subsets")
        self.subset_btn.clicked.connect(self.on_compare_subsets)
        subset_layout.addWidget(self.subset_btn)

        self.subset_canvas = MplCanvas(self, width=7, height=5)
        self.subset_toolbar = NavigationToolbar(self.subset_canvas, self)
        subset_layout.addWidget(self.subset_canvas)
        subset_layout.addWidget(self.subset_toolbar)

        content_layout.addWidget(self.subset_group)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        content_layout.addWidget(self.log_area)

        # Footer
        footer_label = QLabel(
            "<i>All demos above are simplified examples. For the real pipeline, see each dedicated tab.</i>"
        )
        footer_label.setWordWrap(True)
        content_layout.addWidget(footer_label)

        scroll_area.setWidget(content_widget)
        self.setLayout(main_layout)

        # Internals
        self.gamma_sample_img = None
        self.selected_eq_image = None

    def _make_label(self, text):
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        return lbl

    def _on_browse_folder(self):
        dlg = QFileDialog(self, "Select Root Folder")
        dlg.setFileMode(QFileDialog.Directory)
        if dlg.exec_():
            self.root_edit.setText(dlg.selectedFiles()[0])

    def _on_pick_eq_image(self):
        dlg = QFileDialog(self, "Pick an image for histogram eq", self.root_edit.text() or ".")
        dlg.setNameFilter("Images (*.png *.jpg *.jpeg)")
        if dlg.exec_():
            chosen = dlg.selectedFiles()[0]
            self.eq_img_edit.setText(chosen)
            self.selected_eq_image = chosen

    def on_run_demos(self):
        root_folder = self.root_edit.text().strip()
        if not root_folder or not os.path.isdir(root_folder):
            QMessageBox.warning(self, "Invalid Folder", "Please select a valid folder containing images.")
            return

        self.log_area.clear()
        self.res_text.clear()

        subfolders = ["Training", "Testing"]
        max_imgs = self.max_images_spin.value()
        all_imgs = gather_random_images(root_folder, subfolders, max_samples=max_imgs)
        if not all_imgs:
            self.log_area.append("No images found in selected folder.")
            return

        widths, heights, err_count = collect_resolutions(all_imgs)
        if err_count > 0:
            self.log_area.append(f"Error reading {err_count} images.\n")
        if widths and heights:
            wmin, wmax = min(widths), max(widths)
            hmin, hmax = min(heights), max(heights)
            wmean, hmean = np.mean(widths), np.mean(heights)
            self.res_text.append(f"<b>Resolution Summary (up to {max_imgs} images)</b>\n")
            self.res_text.append(
                f"Width => min:{wmin}, max:{wmax}, mean:{wmean:.1f}\n"
                f"Height => min:{hmin}, max:{hmax}, mean:{hmean:.1f}\n"
            )
        else:
            self.res_text.append("No valid images for resolution stats.\n")

        bins = self.res_bin_slider.value()
        self._plot_res_distributions(widths, heights, bins)

        eq_path = self.eq_img_edit.text().strip()
        if eq_path and os.path.isfile(eq_path):
            self.selected_eq_image = eq_path
        else:
            self.selected_eq_image = random.choice(all_imgs)
        self._show_hist_eq_demo(self.selected_eq_image)

        self.gamma_sample_img = random.choice(all_imgs)
        self._update_gamma_demo()

        self.log_area.append("\nDemos complete.\n")

    def _plot_res_distributions(self, widths, heights, bins):
        self.res_canvas.figure.clear()
        fig = self.res_canvas.figure

        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        ax1.hist(widths, bins=bins, color='blue', alpha=0.6)
        ax1.set_title("Widths")
        ax1.set_xlabel("Pixels")
        ax1.set_ylabel("Count")

        ax2.hist(heights, bins=bins, color='green', alpha=0.6)
        ax2.set_title("Heights")
        ax2.set_xlabel("Pixels")

        ax3.scatter(widths, heights, alpha=0.3)
        ax3.set_title("Width vs. Height")
        ax3.set_xlabel("Width")
        ax3.set_ylabel("Height")

        fig.tight_layout()
        self.res_canvas.draw()

    def _show_hist_eq_demo(self, path):
        self.eq_canvas.figure.clear()
        if not path:
            return
        try:
            with Image.open(path).convert('L') as im:
                ax1, ax2 = self.eq_canvas.figure.subplots(1, 2)

                ax1.imshow(im, cmap='gray')
                ax1.set_title("Original")
                ax1.axis('off')

                eq_img = histogram_equalize(im)
                ax2.imshow(eq_img, cmap='gray')
                ax2.set_title("Equalized")
                ax2.axis('off')

                self.eq_canvas.figure.tight_layout()
                self.eq_canvas.draw()

                self.log_area.append(f"Histogram EQ on: {os.path.basename(path)}\n")

        except Exception as e:
            self.log_area.append(f"Histogram EQ error: {e}\n")

    # -----------------------------------------
    # Gamma Correction
    # -----------------------------------------
    def on_gamma_slider_changed(self, val):
        if self.gamma_sample_img:
            self._update_gamma_demo()

    def _update_gamma_demo(self):
        self.gamma_canvas.figure.clear()
        gamma_val = self.gamma_slider.value() / 100.0

        try:
            with Image.open(self.gamma_sample_img).convert('L') as im:
                lut = [pow(i / 255.0, gamma_val)*255 for i in range(256)]
                out_img = im.point(lut)

                ax = self.gamma_canvas.figure.subplots(1, 1)
                ax.imshow(out_img, cmap='gray')
                ax.set_title(f"Gamma = {gamma_val:.2f}")
                ax.axis('off')

                self.gamma_canvas.figure.tight_layout()
                self.gamma_canvas.draw()

        except Exception as e:
            self.log_area.append(f"Gamma error: {e}\n")

    # -----------------------------------------
    # Subset Comparison
    # -----------------------------------------
    def on_compare_subsets(self):
        root_folder = self.root_edit.text().strip()
        if not root_folder or not os.path.isdir(root_folder):
            QMessageBox.warning(self, "Invalid Folder", f"Folder '{root_folder}' not found.")
            return

        choice = self.subset_combo.currentText()
        if choice == "Training only":
            sub_list = ["Training"]
        elif choice == "Testing only":
            sub_list = ["Testing"]
        else:
            sub_list = ["Training", "Testing"]

        w_dict, h_dict = gather_subset_resolutions(root_folder, sub_list)
        self._plot_subset_comparison(w_dict, h_dict)

    def _plot_subset_comparison(self, subset_w, subset_h):
        self.subset_canvas.figure.clear()
        fig = self.subset_canvas.figure

        subs = sorted(subset_w.keys())
        if not subs:
            self.log_area.append("No subsets found to compare.\n")
            return

        if len(subs) == 1:
            # single subset => top: widths, bottom: heights
            ax1 = fig.add_subplot(2, 1, 1)
            wdata = subset_w[subs[0]]
            ax1.hist(wdata, bins=30, alpha=0.6, color='blue')
            ax1.set_title(f"{subs[0]} - Widths")

            ax2 = fig.add_subplot(2, 1, 2)
            hdata = subset_h[subs[0]]
            ax2.hist(hdata, bins=30, alpha=0.6, color='green')
            ax2.set_title(f"{subs[0]} - Heights")
        else:
            # overlapped
            ax1 = fig.add_subplot(2, 1, 1)
            for s in subs:
                ax1.hist(subset_w[s], bins=30, alpha=0.4, label=s)
            ax1.set_title("Widths (Overlapped)")
            ax1.legend()

            ax2 = fig.add_subplot(2, 1, 2)
            for s in subs:
                ax2.hist(subset_h[s], bins=30, alpha=0.4, label=s)
            ax2.set_title("Heights (Overlapped)")
            ax2.legend()

        fig.tight_layout()
        self.subset_canvas.draw()
