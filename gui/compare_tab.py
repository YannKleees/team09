import os
from PIL import Image

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QHBoxLayout,
    QPushButton, QTextEdit, QLineEdit, QMessageBox, QCheckBox
)
from PyQt5.QtCore import Qt

from gui.widgets import MplCanvas
from logic.comparison import (
    count_images_in_classes,
    find_image_in_subfolders,
    load_image_as_grayscale
)

class CompareTab(QWidget):
    """
    Compare 3 sets:
      - Original (data/archive)
      - Cleaned (cleaned_data)
      - Augmented (augmented_data)
    Show distribution and side-by-side-by-side comparison for a chosen image.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()

        info_label = QLabel(
            "<b>Compare Raw vs. Cleaned vs. Augmented</b><br>"
            "Check distribution stats & compare the same image by name."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Dist Stats
        dist_box = QGroupBox("Distribution Stats")
        dist_lay = QHBoxLayout()
        self.btn_compare_dists = QPushButton("Compute Dist Stats (All Three)")
        self.btn_compare_dists.clicked.connect(self.on_compare_dists)

        self.dist_log = QTextEdit()
        self.dist_log.setReadOnly(True)

        dist_lay.addWidget(self.btn_compare_dists)
        dist_lay.addWidget(self.dist_log)
        dist_box.setLayout(dist_lay)
        layout.addWidget(dist_box)

        # Image comparison
        pick_box = QGroupBox("Compare a Specific Image in All Three")
        pick_lay = QHBoxLayout()
        self.image_name_edit = QLineEdit("example.jpg")
        self.btn_show_comparison = QPushButton("Show Comparison (3-way)")
        self.btn_show_comparison.clicked.connect(self.on_show_comparison)
        pick_lay.addWidget(self.image_name_edit)
        pick_lay.addWidget(self.btn_show_comparison)
        pick_box.setLayout(pick_lay)
        layout.addWidget(pick_box)

        self.canvas = MplCanvas(self, width=10, height=4)
        layout.addWidget(self.canvas)

        self.final_check = QCheckBox("Everything looks correct & finalized!")
        layout.addWidget(self.final_check)

        self.setLayout(layout)

    def on_compare_dists(self):
        """
        Compare # images in:
          1) data/archive/Training (original)
          2) cleaned_data/Training
          3) augmented_data/Training
        """
        self.dist_log.clear()

        def show_distribution(root, label):
            if not os.path.isdir(root):
                self.dist_log.append(f"{label} not found or empty.\n")
                return
            dist_dic = count_images_in_classes(root)
            if not dist_dic:
                self.dist_log.append(f"{label} has no images.\n")
                return
            self.dist_log.append(f"<b>{label}</b>:\n")
            total_ = 0
            for c, num in dist_dic.items():
                self.dist_log.append(f"{c}: {num}")
                total_ += num
            self.dist_log.append(f"Total: {total_}\n")

        show_distribution("data/archive/Training", "Original (data/archive/Training)")
        self.dist_log.append("-----")
        show_distribution("cleaned_data/Training", "Cleaned (cleaned_data/Training)")
        self.dist_log.append("-----")
        show_distribution("augmented_data/Training", "Augmented (augmented_data/Training)")

    def on_show_comparison(self):
        """
        3-way comparison:
        original => data/archive/Training
        cleaned  => cleaned_data/Training
        augmented=> augmented_data/Training
        We do a row of 3 images side by side if found.
        """
        img_name = self.image_name_edit.text().strip()
        if not img_name:
            QMessageBox.warning(self, "No Name", "Please specify an image name.")
            return

        # Prepare paths
        orig_root = "data/archive/Training"
        clean_root= "cleaned_data/Training"
        aug_root  = "augmented_data/Training"

        path_orig = find_image_in_subfolders(orig_root, img_name)
        path_clean= find_image_in_subfolders(clean_root, img_name)
        path_aug  = find_image_in_subfolders(aug_root, img_name)

        self.canvas.figure.clear()
        axarr = self.canvas.figure.subplots(1,3, figsize=(12,4))

        # Original
        if path_orig:
            pil_img = load_image_as_grayscale(path_orig)
            if pil_img:
                axarr[0].imshow(pil_img, cmap='gray')
                axarr[0].set_title("Original")
            else:
                axarr[0].text(0.5,0.5,"Error loading original", ha='center', va='center')
        else:
            axarr[0].text(0.5,0.5,"Not found in original", ha='center', va='center')
        axarr[0].axis('off')

        # Cleaned
        if path_clean:
            pil_img2 = load_image_as_grayscale(path_clean)
            if pil_img2:
                axarr[1].imshow(pil_img2, cmap='gray')
                axarr[1].set_title("Cleaned")
            else:
                axarr[1].text(0.5,0.5,"Error loading cleaned", ha='center', va='center')
        else:
            axarr[1].text(0.5,0.5,"Not found in cleaned", ha='center', va='center')
        axarr[1].axis('off')

        # Augmented
        if path_aug:
            pil_img3 = load_image_as_grayscale(path_aug)
            if pil_img3:
                axarr[2].imshow(pil_img3, cmap='gray')
                axarr[2].set_title("Augmented")
            else:
                axarr[2].text(0.5,0.5,"Error loading augmented", ha='center', va='center')
        else:
            axarr[2].text(0.5,0.5,"Not found in augmented", ha='center', va='center')
        axarr[2].axis('off')

        self.canvas.figure.tight_layout()
        self.canvas.draw()
