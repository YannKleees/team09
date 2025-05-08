# File: gui/info_tab.py

import os
import random
import numpy as np
from PIL import Image

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QPushButton,
    QTextEdit, QMessageBox, QHBoxLayout, QSlider, QScrollArea,
    QComboBox, QLineEdit, QCheckBox, QSpinBox, QFileDialog, QFormLayout,
    QSizePolicy
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# If you’re using your own MplCanvas:
from gui.widgets import MplCanvas

# Logic modules
from logic.advanced_eda import (
    gather_random_images,
    collect_resolutions,
    detect_poor_quality,
    histogram_equalize,
    gather_subset_resolutions
)
from logic.augmentation import preview_augmentation
from logic.comparison import load_image_as_grayscale


class InfoTab(QWidget):
    """
    An “ultimate guide” tab that:
      1) Explains each application tab (Preprocessing, Augmentation, HPC, etc.).
      2) Demonstrates how they work with small interactive demos:
         - resolution stats & poor-quality detection
         - histogram equalization
         - gamma correction
         - subset comparisons
         - augmentation previews
         - a mini HPC search simulation
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Master layout: Use a scroll area to comfortably handle the many sections
        main_layout = QVBoxLayout(self)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)

        # Container for all content
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(12)

        # ------------------- Title & Intro -------------------
        title_label = QLabel("<h2>Ultimate Guide & Demonstrations</h2>")
        title_label.setWordWrap(True)
        content_layout.addWidget(title_label)

        intro_label = QLabel(
            "This tab provides an all-in-one overview of every other tab, "
            "plus interactive EDA demos to illustrate how your data changes "
            "during Preprocessing, Augmentation, HPC, and more."
        )
        intro_label.setWordWrap(True)
        content_layout.addWidget(intro_label)

        # ------------------- Overview of Tabs -------------------
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
        overview_layout.addWidget(self._make_label(
            "<b>Auto/HPC Search Tab:</b> Tries different hyperparameters in quick training loops; logs the best combos."
        ))
        overview_layout.addWidget(self._make_label(
            "<b>Inference Tab:</b> Loads a final trained model to classify a single image (binary, multi, or double)."
        ))

        content_layout.addWidget(overview_group)

        # ------------------- Analysis Controls -------------------
        controls_box = QGroupBox("Analysis Controls & Settings")
        controls_form = QFormLayout(controls_box)

        # Root folder
        self.root_edit = QLineEdit("data/archive")
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._on_browse_folder)
        h_line = QHBoxLayout()
        h_line.addWidget(self.root_edit)
        h_line.addWidget(btn_browse)
        controls_form.addRow("Root Folder:", h_line)

        # Max images
        self.max_images_spin = QSpinBox()
        self.max_images_spin.setRange(1, 1000)
        self.max_images_spin.setValue(200)
        controls_form.addRow("Max Images to Sample:", self.max_images_spin)

        # Poor-quality
        self.pq_chk = QCheckBox("Enable poor-quality detection?")
        self.pq_chk.setChecked(True)
        controls_form.addRow(self.pq_chk)

        # Resolution bin slider
        self.res_bin_slider = QSlider(Qt.Horizontal)
        self.res_bin_slider.setRange(5, 50)
        self.res_bin_slider.setValue(20)
        controls_form.addRow("Resolution Histogram Bins:", self.res_bin_slider)

        # hist eq image
        self.eq_img_edit = QLineEdit()
        self.eq_img_edit.setPlaceholderText("Optional image for histogram eq; or leave blank for random.")
        eq_btn = QPushButton("Pick...")
        eq_btn.clicked.connect(self._on_pick_eq_image)
        eq_h = QHBoxLayout()
        eq_h.addWidget(self.eq_img_edit)
        eq_h.addWidget(eq_btn)
        controls_form.addRow("Histogram EQ Image:", eq_h)

        # Button to run all demos
        self.run_demos_btn = QPushButton("Run Demos")
        self.run_demos_btn.clicked.connect(self.on_run_demos)
        controls_form.addRow(self.run_demos_btn)

        content_layout.addWidget(controls_box)

        # ------------------- Resolution & Poor Quality Demo -------------------
        self.res_group = QGroupBox("Resolution & Poor-Quality Demo")
        res_layout = QVBoxLayout(self.res_group)

        self.res_text = QTextEdit()
        self.res_text.setReadOnly(True)
        res_layout.addWidget(self.res_text)

        self.res_canvas = MplCanvas(self, width=7, height=5)  # bigger chart
        self.res_toolbar = NavigationToolbar(self.res_canvas, self)
        res_layout.addWidget(self.res_canvas)
        res_layout.addWidget(self.res_toolbar)

        content_layout.addWidget(self.res_group)

        # ------------------- Histogram EQ Demo -------------------
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

        # ------------------- Gamma Demo -------------------
        self.gamma_group = QGroupBox("Gamma Correction Demo")
        gamma_layout = QVBoxLayout(self.gamma_group)

        gamma_label = QLabel(
            "Move the slider to lighten/darken the image. (<1 => lighten, >1 => darken). "
            "Similar transformations appear in the Augmentation Tab."
        )
        gamma_label.setWordWrap(True)
        gamma_layout.addWidget(gamma_label)

        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(50, 200)  # 0.5..2.0 scaled by 100
        self.gamma_slider.setValue(100)
        self.gamma_slider.valueChanged.connect(self.on_gamma_slider_changed)
        gamma_layout.addWidget(self.gamma_slider)

        self.gamma_canvas = MplCanvas(self, width=7, height=5)
        self.gamma_toolbar = NavigationToolbar(self.gamma_canvas, self)
        gamma_layout.addWidget(self.gamma_canvas)
        gamma_layout.addWidget(self.gamma_toolbar)

        content_layout.addWidget(self.gamma_group)

        # ------------------- Subset Comparison Demo -------------------
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

        # ------------------- Augmentation Preview Demo -------------------
        self.aug_group = QGroupBox("Augmentation Preview Demo")
        aug_layout = QVBoxLayout(self.aug_group)

        aug_label = QLabel(
            "Pick an image, see multiple random transformations side by side. "
            "Mirrors the 'preview' functionality in the Augmentation Tab."
        )
        aug_label.setWordWrap(True)
        aug_layout.addWidget(aug_label)

        self.aug_img_edit = QLineEdit()
        self.aug_img_edit.setPlaceholderText("Select image for augmentation preview.")
        aug_btn = QPushButton("Browse Image...")
        aug_btn.clicked.connect(self._on_browse_augment_img)

        aug_hbox = QHBoxLayout()
        aug_hbox.addWidget(self.aug_img_edit)
        aug_hbox.addWidget(aug_btn)
        aug_layout.addLayout(aug_hbox)

        self.aug_preview_btn = QPushButton("Generate Augmented Previews")
        self.aug_preview_btn.clicked.connect(self.on_generate_aug_preview)
        aug_layout.addWidget(self.aug_preview_btn)

        self.aug_canvas = MplCanvas(self, width=7, height=5)
        self.aug_toolbar = NavigationToolbar(self.aug_canvas, self)
        aug_layout.addWidget(self.aug_canvas)
        aug_layout.addWidget(self.aug_toolbar)

        content_layout.addWidget(self.aug_group)

        # ------------------- HPC Mini-Demo -------------------
        self.hpc_group = QGroupBox("HPC (Hyperparameter Tuning) Mini-Demo")
        hpc_layout = QVBoxLayout(self.hpc_group)

        hpc_desc = QLabel(
            "Simulated HPC search with random results for different hyperparams (e.g., LR). "
            "In the real HPC tab, partial training is done for each combo."
        )
        hpc_desc.setWordWrap(True)
        hpc_layout.addWidget(hpc_desc)

        self.hpc_run_btn = QPushButton("Simulate HPC Search")
        self.hpc_run_btn.clicked.connect(self.on_hpc_demo)
        hpc_layout.addWidget(self.hpc_run_btn)

        self.hpc_canvas = MplCanvas(self, width=7, height=5)
        self.hpc_toolbar = NavigationToolbar(self.hpc_canvas, self)
        hpc_layout.addWidget(self.hpc_canvas)
        hpc_layout.addWidget(self.hpc_toolbar)

        content_layout.addWidget(self.hpc_group)

        # ------------------- Log / Output -------------------
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
        self.aug_img_path = None

    # -----------------------------------------
    # Helper: a word-wrapped QLabel
    # -----------------------------------------
    def _make_label(self, text):
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        return lbl

    # -----------------------------------------
    # Folder & EQ image pick
    # -----------------------------------------
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

    # -----------------------------------------
    # “Run Demos” => resolution stats, eq, gamma
    # -----------------------------------------
    def on_run_demos(self):
        root_folder = self.root_edit.text().strip()
        if not root_folder or not os.path.isdir(root_folder):
            QMessageBox.warning(self, "Invalid Folder", "Please select a valid folder containing images.")
            return

        # Clear logs
        self.log_area.clear()
        self.res_text.clear()

        # Gather random images
        subfolders = ["Training", "Testing"]
        max_imgs = self.max_images_spin.value()
        all_imgs = gather_random_images(root_folder, subfolders, max_samples=max_imgs)
        if not all_imgs:
            self.log_area.append("No images found in selected folder.")
            return

        # 1) resolution stats
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

        # 2) optional poor-quality
        if self.pq_chk.isChecked():
            pq = detect_poor_quality(all_imgs)
            self.log_area.append(f"{len(pq)} poor-quality scans found (if any).")
        else:
            self.log_area.append("Poor-quality detection disabled.\n")

        # 3) plot resolution distribution
        bins = self.res_bin_slider.value()
        self._plot_res_distributions(widths, heights, bins)

        # 4) hist eq => pick user-chosen or random
        eq_path = self.eq_img_edit.text().strip()
        if eq_path and os.path.isfile(eq_path):
            self.selected_eq_image = eq_path
        else:
            self.selected_eq_image = random.choice(all_imgs)
        self._show_hist_eq_demo(self.selected_eq_image)

        # 5) gamma => pick random
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

    # -----------------------------------------
    # Augmentation Preview
    # -----------------------------------------
    def _on_browse_augment_img(self):
        dlg = QFileDialog(self, "Select Image for Augmentation Preview")
        dlg.setNameFilter("Images (*.png *.jpg *.jpeg)")
        if dlg.exec_():
            path = dlg.selectedFiles()[0]
            self.aug_img_edit.setText(path)
            self.aug_img_path = path

    def on_generate_aug_preview(self):
        if not self.aug_img_path or not os.path.isfile(self.aug_img_path):
            QMessageBox.warning(self, "No Image", "Please select a valid image for augmentation preview.")
            return

        pil_img = load_image_as_grayscale(self.aug_img_path)
        if pil_img is None:
            self.log_area.append("Could not load the chosen image.")
            return

        self.aug_canvas.figure.clear()
        fig = self.aug_canvas.figure
        n_cols = 4
        axarr = fig.subplots(1, n_cols, figsize=(14, 4))  # even bigger horizontally

        axarr[0].imshow(pil_img, cmap='gray')
        axarr[0].set_title("Original")
        axarr[0].axis('off')

        for i in range(1, n_cols):
            aug_tensor = preview_augmentation(
                pil_image=pil_img,
                rotate_deg=15,
                do_rotate=True,
                do_brightness_contrast=True,
                do_gamma=True,
                do_minmax=False,
                do_flip=True,
                do_perspective=False,
                do_crop=False
            )
            arr = aug_tensor.squeeze().numpy()
            axarr[i].imshow(arr, cmap='gray')
            axarr[i].set_title(f"Aug {i}")
            axarr[i].axis('off')

        fig.tight_layout()
        self.aug_canvas.draw()

        self.log_area.append("Random Augmentation previews generated.\n")

    # -----------------------------------------
    # HPC Demo
    # -----------------------------------------
    def on_hpc_demo(self):
        import random
        combos = [0.0001, 0.0005, 0.001, 0.003, 0.01]
        val_accs = [random.uniform(70, 95) for _ in combos]

        self.hpc_canvas.figure.clear()
        ax = self.hpc_canvas.figure.add_subplot(111)

        xvals = range(len(combos))
        ax.bar(xvals, val_accs, color='purple', alpha=0.6)
        ax.set_title("Simulated HPC: LR vs. Val Accuracy")
        ax.set_xlabel("Combo Index")
        ax.set_ylabel("Val Accuracy (%)")
        ax.set_xticks(xvals)
        ax.set_xticklabels([str(c) for c in combos], rotation=45)

        self.hpc_canvas.figure.tight_layout()
        self.hpc_canvas.draw()

        best_val = max(val_accs)
        idx_best = val_accs.index(best_val)
        best_lr = combos[idx_best]
        self.log_area.append(f"HPC Demo => Best LR: {best_lr} with ValAcc ~ {best_val:.2f}%\n")
