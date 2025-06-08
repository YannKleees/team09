# gui/compare_tab.py

import os
import glob
import numpy as np
from PIL import Image

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QHBoxLayout,
    QPushButton, QComboBox, QTextEdit, QMessageBox, QScrollArea
)
from PyQt5.QtCore import Qt

from gui.widgets import MplCanvas
from logic.comparison import load_image_as_grayscale


class CompareTab(QWidget):
    """
    Lets you browse images in a chosen dataset folder, then tries to find the
    same filename in other sets (raw, cleaned, augmented). Now shows:
      - The Original, Cleaned, plus *all* augmented variants side by side
      - Below each image, a histogram with mean/std/min/max/median
      - Color-coded borders for each type
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLayout(QVBoxLayout())

        info_label = QLabel(
            "<h3>Compare: Raw vs. Cleaned vs. Augmented (All Variants)</h3>"
            "Browse to see how one filename appears across raw, cleaned, and multiple augmentations."
        )
        info_label.setWordWrap(True)
        self.layout().addWidget(info_label)

        # -----------------------------------
        # Browsing controls
        # -----------------------------------
        browse_group = QGroupBox("Image Navigation")
        browse_layout = QHBoxLayout()

        self.dataset_combo = QComboBox()
        self.dataset_combo.setToolTip("Choose which dataset folder to scan.")
        self.dataset_combo.addItems([
            "data/archive/Training",
            "cleaned_data/Training",
            "augmented_data/Training"
        ])

        self.load_btn = QPushButton("Load Image List")
        self.load_btn.setToolTip("Find all images in the selected folder (recursively).")
        self.load_btn.clicked.connect(self.on_load_images)

        self.prev_btn = QPushButton("Previous")
        self.prev_btn.setToolTip("Go to the previous image in the list.")
        self.prev_btn.clicked.connect(self.on_prev_image)

        self.next_btn = QPushButton("Next")
        self.next_btn.setToolTip("Go to the next image in the list.")
        self.next_btn.clicked.connect(self.on_next_image)

        browse_layout.addWidget(self.dataset_combo)
        browse_layout.addWidget(self.load_btn)
        browse_layout.addWidget(self.prev_btn)
        browse_layout.addWidget(self.next_btn)
        browse_group.setLayout(browse_layout)
        self.layout().addWidget(browse_group)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.layout().addWidget(self.scroll_area)

        self.log_box = QTextEdit()
        self.log_box.setToolTip("Shows which files were found plus info about augmented images.")
        self.log_box.setReadOnly(True)
        self.layout().addWidget(self.log_box)

        self.all_images = []
        self.curr_index = -1

    def on_load_images(self):
        chosen_path = self.dataset_combo.currentText().strip()
        if not os.path.isdir(chosen_path):
            QMessageBox.warning(self, "Error", f"{chosen_path} is not a valid folder.")
            return

        pattern_list = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            pattern_list.extend(glob.glob(os.path.join(chosen_path, "**", ext), recursive=True))

        self.all_images = sorted(pattern_list)
        self.curr_index = 0 if self.all_images else -1

        self.log_box.clear()
        if not self.all_images:
            self.log_box.append("No images found in this folder.")
        else:
            self.log_box.append(f"Loaded {len(self.all_images)} images from {chosen_path}.")
            self.show_comparison()

    def on_prev_image(self):
        if not self.all_images:
            return
        self.curr_index -= 1
        if self.curr_index < 0:
            self.curr_index = len(self.all_images) - 1
        self.show_comparison()

    def on_next_image(self):
        if not self.all_images:
            return
        self.curr_index += 1
        if self.curr_index >= len(self.all_images):
            self.curr_index = 0
        self.show_comparison()

    def show_comparison(self):
        if not self.all_images or self.curr_index < 0:
            return

        curr_image_path = self.all_images[self.curr_index]
        base_name = os.path.basename(curr_image_path)

        root_orig = "data/archive/Training"
        root_clean = "cleaned_data/Training"
        root_aug = "augmented_data/Training"

        path_orig = self._locate_image(root_orig, base_name)
        path_clean = self._locate_image(root_clean, base_name)
        aug_paths = self._locate_all_aug_images(root_aug, base_name)

        images_and_labels = []
        if path_orig:
            images_and_labels.append((path_orig, "Original"))
        if path_clean:
            images_and_labels.append((path_clean, "Cleaned"))
        for i, ap in enumerate(aug_paths):
            images_and_labels.append((ap, f"Aug{i+1}"))

        from gui.widgets import MplCanvas
        new_canvas = MplCanvas(self, width=5, height=4)
        fig = new_canvas.figure

        if not images_and_labels:
            fig.clear()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No matching images found in raw/cleaned/aug sets.", ha='center', va='center')
            ax.axis('off')
            new_canvas.draw()

            self.scroll_area.setWidget(new_canvas)
            self.log_box.clear()
            self.log_box.append(f"No matching images for '{base_name}' in raw/cleaned/aug.")
            return

        n_images = len(images_and_labels)

        fig.clear()
        fig.set_size_inches(4*n_images, 6)
        axes = fig.subplots(nrows=2, ncols=n_images)

        if n_images == 1:
            axes = np.array([axes]).reshape(2,1)

        def color_spines(ax, color="black", lw=2):
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(lw)

        for col_idx, (img_path, label_str) in enumerate(images_and_labels):
            ax_img = axes[0, col_idx]
            ax_hist = axes[1, col_idx]

            arr = load_image_as_grayscale(img_path)
            if arr is None:
                ax_img.text(0.5, 0.5, "Error reading", ha='center', va='center', color='red')
                ax_img.axis('off')
                ax_hist.text(0.5, 0.5, "No data", ha='center', va='center', color='red')
                ax_hist.axis('off')
                continue

            ax_img.imshow(arr, cmap='gray')
            ax_img.axis('off')
            ax_img.set_title(label_str, fontsize=10)

            if label_str.lower().startswith("original"):
                color_spines(ax_img, "green")
            elif label_str.lower().startswith("clean"):
                color_spines(ax_img, "blue")
            else:
                color_spines(ax_img, "darkorange")

            flat = np.array(arr).ravel().astype(np.float32)
            ax_hist.hist(flat, bins=50, color='blue', alpha=0.6)
            ax_hist.set_xlabel("Intensity", fontsize=8)
            ax_hist.set_ylabel("Count", fontsize=8)

            mean_val = float(flat.mean())
            std_val = float(flat.std())
            mn_val = float(flat.min())
            mx_val = float(flat.max())
            med_val = float(np.median(flat))

            ax_hist.set_title(
                f"Mean={mean_val:.1f}, Std={std_val:.1f}\n"
                f"Min={mn_val:.0f}, Max={mx_val:.0f}, Med={med_val:.1f}",
                fontsize=9
            )

        fig.tight_layout()
        new_canvas.draw()
        self.scroll_area.setWidget(new_canvas)

        self.log_box.clear()
        self.log_box.append(f"<b>Comparing '{base_name}' across sets:</b>")
        self.log_box.append(f"Index {self.curr_index+1}/{len(self.all_images)} in {self.dataset_combo.currentText()}")
        self.log_box.append(f"Original path:   {path_orig if path_orig else 'Not found'}")
        self.log_box.append(f"Cleaned path:    {path_clean if path_clean else 'Not found'}")
        if aug_paths:
            self.log_box.append(f"Found {len(aug_paths)} augmented variant(s).")
            for i, ap in enumerate(aug_paths):
                self.log_box.append(f"   Aug{i+1}: {ap}")
        else:
            self.log_box.append("No augmented variants found for this file.")

    def _locate_image(self, root_dir, base_name):
        """
        Return the full path for base_name if found in root_dir (recursively).
        """
        if not os.path.isdir(root_dir):
            return None
        for subdir, dirs, files in os.walk(root_dir):
            if base_name in files:
                return os.path.join(subdir, base_name)
        return None

    def _locate_all_aug_images(self, root_dir, base_name):
        """
        Return a list of all augmented images (e.g. baseName_aug0.jpg),
        searching inside `root_dir`.
        We'll guess the 'stem' is base_name minus extension,
        then anything that starts with that + "_aug" + same ext.
        """
        if not os.path.isdir(root_dir):
            return []
        stem, ext = os.path.splitext(base_name)
        matches = []
        for subdir, dirs, files in os.walk(root_dir):
            for f in files:
                # check if it starts with <stem>_aug and ends with same extension
                if f.lower().startswith(stem.lower() + "_aug") and f.lower().endswith(ext.lower()):
                    matches.append(os.path.join(subdir, f))
        matches.sort()
        return matches
