import os
import glob
from PIL import Image

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QHBoxLayout,
    QPushButton, QComboBox, QTextEdit, QMessageBox
)
from PyQt5.QtCore import Qt

from gui.widgets import MplCanvas
from logic.comparison import load_image_as_grayscale

class CompareTab(QWidget):
    """
    Lets you browse images in a chosen dataset folder (Original, Cleaned, or Augmented),
    then tries to find the same filename in the other sets to show them side by side.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLayout(QVBoxLayout())

        info_label = QLabel(
            "<h3>Compare Raw vs. Cleaned vs. Augmented</h3>"
            "Browse the same image filename across all three sets (if it exists)."
        )
        info_label.setWordWrap(True)
        self.layout().addWidget(info_label)

        # Group box for folder selection + browsing
        self.group_browse = QGroupBox("Browse Images by Class Folder")
        g_layout = QHBoxLayout()

        self.dataset_combo = QComboBox()
        self.dataset_combo.setToolTip("Choose which dataset folder to scan.")
        self.dataset_combo.addItems(["data/archive/Training", "cleaned_data/Training", "augmented_data/Training"])

        self.load_btn = QPushButton("Load Image List")
        self.load_btn.setToolTip("Find all images (png/jpg/jpeg) in the selected folder.")
        self.load_btn.clicked.connect(self.on_load_images)

        self.prev_btn = QPushButton("Previous")
        self.prev_btn.setToolTip("Go to the previous image in the list.")
        self.prev_btn.clicked.connect(self.on_prev_image)

        self.next_btn = QPushButton("Next")
        self.next_btn.setToolTip("Go to the next image in the list.")
        self.next_btn.clicked.connect(self.on_next_image)

        for w in [self.dataset_combo, self.load_btn, self.prev_btn, self.next_btn]:
            g_layout.addWidget(w)
        self.group_browse.setLayout(g_layout)
        self.layout().addWidget(self.group_browse)

        # Canvas for side-by-side images
        self.canvas = MplCanvas(self, width=10, height=4)
        self.layout().addWidget(self.canvas)

        # Log
        self.log_box = QTextEdit()
        self.log_box.setToolTip("Shows which files were found in each dataset.")
        self.log_box.setReadOnly(True)
        self.layout().addWidget(self.log_box)

        # Internals
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
        path_aug = self._locate_image(root_aug, base_name)

        self.canvas.figure.clear()
        axarr = self.canvas.figure.subplots(1, 3)

        self._plot_image(axarr[0], path_orig, "Original")
        self._plot_image(axarr[1], path_clean, "Cleaned")
        self._plot_image(axarr[2], path_aug, "Augmented")

        self.canvas.figure.tight_layout()
        self.canvas.draw()

        self.log_box.clear()
        self.log_box.append(f"Comparing '{base_name}' across sets:")
        self.log_box.append(f"Index {self.curr_index+1}/{len(self.all_images)} in {self.dataset_combo.currentText()}")
        self.log_box.append(f"Original path:   {path_orig if path_orig else 'Not found'}")
        self.log_box.append(f"Cleaned path:    {path_clean if path_clean else 'Not found'}")
        self.log_box.append(f"Augmented path:  {path_aug if path_aug else 'Not found'}")

    def _locate_image(self, root_dir, base_name):
        for subdir, dirs, files in os.walk(root_dir):
            if base_name in files:
                return os.path.join(subdir, base_name)
        return None

    def _plot_image(self, ax, path, title):
        if path and os.path.isfile(path):
            pil_img = load_image_as_grayscale(path)
            if pil_img:
                ax.imshow(pil_img, cmap='gray')
            else:
                ax.text(0.5, 0.5, "Error reading", ha='center', va='center')
        else:
            ax.text(0.5, 0.5, "Not found", ha='center', va='center')
        ax.set_title(title)
        ax.axis('off')
