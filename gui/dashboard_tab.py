import os
import random
import numpy as np
from PIL import Image
import cv2  # For advanced histogram eq if needed

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QFormLayout,
    QPushButton, QSpinBox, QTextEdit, QMessageBox, QScrollArea,
    QSizePolicy, QHBoxLayout, QCheckBox, QComboBox, QDialog,
    QGridLayout
)
from PyQt5.QtCore import Qt, QPoint
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar
)
from matplotlib.backend_bases import MouseButton, PickEvent
from matplotlib.patches import Rectangle

from gui.widgets import MplCanvas


########################################################################
## HELPER FUNCTIONS
########################################################################
def gather_subfolders(root_folder):
    """
    Return a list of subfolder names under `root_folder` that themselves
    contain images or class folders. Typically: ["Training","Testing","Validation"].
    """
    if not os.path.isdir(root_folder):
        return []
    subfolders = []
    for d in os.listdir(root_folder):
        dpath = os.path.join(root_folder, d)
        if os.path.isdir(dpath):
            # check if it has images or sub-subfolders
            subfolders.append(d)
    return sorted(subfolders)


def gather_images_in_folder(folder_path, max_samples=None):
    """
    Return a list of all .png/.jpg/.jpeg images inside `folder_path` (recursively).
    If max_samples is set, limit the total count.
    """
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            ext = f.lower()
            if ext.endswith('.png') or ext.endswith('.jpg') or ext.endswith('.jpeg'):
                all_files.append(os.path.join(root, f))

    if max_samples and len(all_files) > max_samples:
        random.shuffle(all_files)
        return all_files[:max_samples]
    return all_files


def load_grayscale_float32(image_path):
    """
    Load a single image as grayscale float32. Returns a flattened array.
    On error, returns None.
    """
    try:
        from PIL import Image
        img = Image.open(image_path).convert('L')
        arr = np.array(img, dtype=np.float32)
        return arr.flatten()  # flatten for intensity stats
    except:
        return None


def gather_class_distribution(root_folder, subfolders):
    """
    For each subset (e.g. "Training", "Testing"), gather how many images
    are in each class subfolder (e.g. 'NoTumor', 'Pituitary', etc.).
    
    Returns a dict of shape:
      {
        "Training": {
           "NoTumor": 123,
           "Pituitary": 111,
           ...
        },
        "Testing": { ... }
      }
    """
    dist_info = {}
    for subset in subfolders:
        dist_info[subset] = {}
        subset_path = os.path.join(root_folder, subset)
        if not os.path.isdir(subset_path):
            continue

        # For each class folder in this subset:
        for cls_ in os.listdir(subset_path):
            cpath = os.path.join(subset_path, cls_)
            if os.path.isdir(cpath):
                # Count how many images
                cnt = 0
                for f in os.listdir(cpath):
                    lowf = f.lower()
                    if lowf.endswith('.png') or lowf.endswith('.jpg') or lowf.endswith('.jpeg'):
                        cnt += 1
                dist_info[subset][cls_] = cnt

    return dist_info


########################################################################
## DashboardTab (Upgraded)
########################################################################
class DashboardTab(QWidget):
    """
    Upgraded "DashboardTab" with:
      1. Multi-Subset Comparisons
      2. Advanced Stats (quartiles, box plots)
      3. Interactive histogram exploration
      4. Original sampling + distribution
      5. Compare Normalizations
      6. (NEW) Display class distribution counts in each folder
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
                background-color: #005CB5;
            }
            QPushButton:disabled {
                background-color: #9FBADF;
            }

            QSpinBox, QLineEdit, QComboBox, QTextEdit {
                font-size: 10pt;
                padding: 4px;
            }
            QCheckBox {
                font-size: 10pt;
            }

            QTextEdit {
                background-color: #FFFFFF;
                border: 1px solid #CCCCCC;
                border-radius: 4px;
            }

            QScrollArea {
                border: none;
            }
        """)

        self.setObjectName("DashboardTab")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Scroll area for all content
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)

        # Container widget + layout
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(15)

        # Title and short description
        title_label = QLabel("<h2 style='color: #0078D7; margin-bottom: 0;'>Data Overview & Dashboard</h2>")
        title_label.setWordWrap(True)
        content_layout.addWidget(title_label)

        desc_label = QLabel(
            "Analyze class distribution, sample intensities, compare normalizations,\n"
            "and advanced features: multi-subset comparisons, box plots, interactive histogram exploration.\n"
            "Also see how many images exist in each folder/class."
        )
        desc_label.setWordWrap(True)
        content_layout.addWidget(desc_label)

        # --- Folder Selection & Analysis group ---
        select_group = QGroupBox("Folder Selection & Analysis")
        form_layout = QFormLayout()
        form_layout.setSpacing(10)

        self.folder_edit_label = QLabel("data/archive")  # just a label, or you can use QLineEdit
        self.folder_edit_label.setToolTip("Path containing subfolders like 'Training', 'Testing', etc.")
        self.folder_edit_label.setStyleSheet("font-weight: 600; color: #444444;")

        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(1, 200)
        self.sample_spin.setValue(30)
        self.sample_spin.setToolTip("How many images to sample for intensity stats (each subset).")

        # Button row
        self.btn_analyze = QPushButton("Analyze Folder")
        self.btn_analyze.setToolTip("Compute distribution & intensity stats for subsets, then plot.")
        self.btn_analyze.clicked.connect(self.on_analyze)

        h_buttons = QHBoxLayout()
        h_buttons.addWidget(self.btn_analyze)

        form_layout.addRow("Dataset Folder:", self.folder_edit_label)
        form_layout.addRow("Images to Sample:", self.sample_spin)
        form_layout.addRow(h_buttons)

        select_group.setLayout(form_layout)
        content_layout.addWidget(select_group)

        # Log box
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(150)
        content_layout.addWidget(self.log_box)

        # --- (NEW) Class Distribution group ---
        self.distribution_group = QGroupBox("Folder Info & Class Distribution")
        dist_layout = QVBoxLayout(self.distribution_group)
        dist_layout.setSpacing(10)

        dist_desc = QLabel(
            "Displays how many images exist in each subset folder\n"
            "and each class subfolder."
        )
        dist_desc.setWordWrap(True)
        dist_layout.addWidget(dist_desc)

        self.distribution_text = QTextEdit()
        self.distribution_text.setReadOnly(True)
        self.distribution_text.setMinimumHeight(220)  # make it taller
        dist_layout.addWidget(self.distribution_text)

        content_layout.addWidget(self.distribution_group)

        # Plot area (Histogram of intensities)
        self.plot_group = QGroupBox("Histogram of Sampled Intensities")
        plot_layout = QVBoxLayout()

        self.canvas = MplCanvas(self, width=5, height=4)
        self.canvas_toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.canvas)
        plot_layout.addWidget(self.canvas_toolbar)

        self.plot_group.setLayout(plot_layout)
        content_layout.addWidget(self.plot_group)

        # --- Advanced Stats group: box plots, quartiles, multi-subset selection ---
        self.adv_group = QGroupBox("Advanced Stats & Multi-Subset Comparison")
        adv_layout = QVBoxLayout(self.adv_group)
        adv_layout.setSpacing(10)

        # We'll dynamically find subfolders in the root. The user can pick which to compare.
        self.subset_checkboxes = []
        self.subset_box_layout = QVBoxLayout()
        adv_layout.addWidget(QLabel("Select subsets to compare (Training, Testing, Validation, etc.):"))
        adv_layout.addLayout(self.subset_box_layout)

        self.adv_run_btn = QPushButton("Compare Selected Subsets")
        self.adv_run_btn.clicked.connect(self.on_compare_subsets)
        adv_layout.addWidget(self.adv_run_btn)

        # Boxplot canvas
        self.box_canvas = MplCanvas(self, width=5, height=3)
        self.box_toolbar = NavigationToolbar(self.box_canvas, self)
        adv_layout.addWidget(self.box_canvas)
        adv_layout.addWidget(self.box_toolbar)

        # We'll have a text box for advanced stats
        self.adv_stats_box = QTextEdit()
        self.adv_stats_box.setReadOnly(True)
        adv_layout.addWidget(self.adv_stats_box)

        content_layout.addWidget(self.adv_group)

        # Footer note
        footer_label = QLabel(
            "<i>Note: This demonstration attempts to read images from the selected folder or the default <b>data/archive</b> if not changed.</i>"
        )
        footer_label.setWordWrap(True)
        content_layout.addWidget(footer_label)

        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)

        # Internal variables to store intensities for each subset
        self.subset_to_intensities = {}
        # For interactive bin clicking:
        self._current_hist_data = {}

    # --------------------------------------------------------------------------
    # Compare Normalization (placeholder - not heavily used in this demo)
    # --------------------------------------------------------------------------
    def on_compare_normalization(self):
        root_folder = self.folder_edit_label.text().strip()
        if not os.path.isdir(root_folder):
            QMessageBox.warning(self, "Invalid Folder", f"Path '{root_folder}' does not exist.")
            return
        # Placeholder for a normalization comparison dialog
        QMessageBox.information(self, "Normalization Comparison", "Dialog not implemented in this example.")

    # --------------------------------------------------------------------------
    # Main "Analyze Folder" => Finds subsets, samples intensities, plots them
    # + (NEW) Show class distribution in distribution_text
    # --------------------------------------------------------------------------
    def on_analyze(self):
        """
        1) Detect which subfolders exist (e.g. "Training","Testing","Validation") 
        2) For each subfolder, gather intensities from up to 'sample_spin' images
        3) Show them in a single combined histogram + show stats in the log
        4) Update the checkboxes in the "Advanced Stats" area for subset selection
        5) (NEW) Show how many images are in each folder/class in distribution_text
        """
        root_folder = self.folder_edit_label.text().strip()
        if not os.path.isdir(root_folder):
            QMessageBox.warning(self, "Invalid Folder", "Selected dataset folder not found.")
            return

        # Clear old data
        self.log_box.clear()
        self.subset_to_intensities.clear()
        self.distribution_text.clear()

        # 1) Find subfolders
        subfolders = gather_subfolders(root_folder)
        if not subfolders:
            self.log_box.append("No subfolders found (e.g. no 'Training','Testing'?). Aborting.")
            return

        # (NEW) Show class distribution
        dist_info = gather_class_distribution(root_folder, subfolders)
        self._show_class_distribution(dist_info)

        # 2) For each subfolder, gather random images -> intensities
        sample_count = self.sample_spin.value()
        for subset in subfolders:
            subset_path = os.path.join(root_folder, subset)
            all_imgs = gather_images_in_folder(subset_path, max_samples=sample_count)

            intensities = []
            error_count = 0
            for path in all_imgs:
                arr_f = load_grayscale_float32(path)
                if arr_f is None:
                    error_count += 1
                    continue
                intensities.extend(arr_f)

            self.subset_to_intensities[subset] = np.array(intensities, dtype=np.float32)
            self.log_box.append(f"<b>{subset}:</b> sampled {len(intensities)} intensities ({error_count} errors)")

        # 3) Combined histogram plot
        self.plot_combined_histogram()

        # 4) Update subset checkboxes
        #    Clear old checkboxes
        for cb in self.subset_checkboxes:
            self.subset_box_layout.removeWidget(cb)
            cb.deleteLater()
        self.subset_checkboxes.clear()

        for subset in subfolders:
            cb = QCheckBox(subset)
            cb.setChecked(True)
            self.subset_box_layout.addWidget(cb)
            self.subset_checkboxes.append(cb)

        self.log_box.append("\nAnalysis complete. Check 'Advanced Stats & Multi-Subset Comparison' for more.\n")

    def _show_class_distribution(self, dist_info):
        """
        Display the class/folder distribution in the distribution_text box.
        dist_info is a dict: { "Training": {"NoTumor": 100, ...}, "Testing": {...}, ...}
        """
        self.distribution_text.clear()
        self.distribution_text.append("<b>Class Distribution in Each Subfolder:</b>\n")
        for subset, class_map in dist_info.items():
            self.distribution_text.append(f"<u>{subset}</u>:")
            if not class_map:
                self.distribution_text.append("  (No classes or images found)")
            else:
                for cls_name, count in class_map.items():
                    self.distribution_text.append(f"  {cls_name}: {count}")
            self.distribution_text.append("")  # spacing

    def plot_combined_histogram(self):
        """
        Plot an overlapping histogram with a different color for each subset.
        We'll also enable pick events so user can click bars (interactive).
        """
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)

        colors = ["blue", "green", "red", "orange", "purple", "teal"]
        color_idx = 0
        self._current_hist_data.clear()  # store bin info for interactive clicks

        for subset, data in self.subset_to_intensities.items():
            if len(data) == 0:
                continue
            c = colors[color_idx % len(colors)]
            color_idx += 1
            # Plot histogram with alpha
            counts, bins, patches = ax.hist(
                data,
                bins=50,
                alpha=0.4,
                label=subset,
                color=c,
                picker=True  # enable picking
            )
            # We'll store the bin edges & counts so we can handle pick events
            for p in patches:
                p.set_picker(True)

        ax.set_title("Combined Histogram of Sampled Intensities")
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Count")
        ax.legend()

        # Connect pick event
        self.canvas.mpl_connect("pick_event", self.on_hist_pick)

        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def on_hist_pick(self, event: PickEvent):
        """
        Called when user clicks on a bar (patch) in the histogram.
        We'll attempt to identify the bin range, then show images in that range if feasible.
        """
        artist = event.artist
        if not isinstance(artist, Rectangle):
            return

        # Extract bin range from rectangle
        x_left = artist.get_x()
        width = artist.get_width()
        bin_min = x_left
        bin_max = x_left + width

        self.log_box.append(f"<b>Clicked bin:</b> {bin_min:.1f} to {bin_max:.1f}<br>")

        # Just a simple info dialog with the counts in that bin for each subset
        count_in_range = {}
        for subset, intensities in self.subset_to_intensities.items():
            sub_vals = intensities[(intensities >= bin_min) & (intensities < bin_max)]
            count_in_range[subset] = len(sub_vals)

        lines = []
        for subset, cnt in count_in_range.items():
            lines.append(f"{subset}: {cnt} intensities in that bin.")
        detail_str = "\n".join(lines)

        msg = f"You clicked bin {bin_min:.1f}–{bin_max:.1f}.\n\n{detail_str}"
        QMessageBox.information(self, "Histogram Bin Info", msg)

    # --------------------------------------------------------------------------
    # "Compare Subsets" button => do box plots, advanced stats
    # --------------------------------------------------------------------------
    def on_compare_subsets(self):
        """
        1) Gather whichever subsets are checked.
        2) Show a box plot for intensities in each subset side by side.
        3) Display quartiles, min, max, outliers in adv_stats_box.
        4) Possibly highlight if one subset is systematically darker or lighter.
        """
        chosen_subsets = []
        for cb in self.subset_checkboxes:
            if cb.isChecked():
                chosen_subsets.append(cb.text())

        if not chosen_subsets:
            QMessageBox.warning(self, "No subsets selected", "Please check at least one subset.")
            return

        # Prepare data for box plot
        data_for_boxplot = []
        subset_labels = []
        for s in chosen_subsets:
            arr = self.subset_to_intensities.get(s, None)
            if arr is not None and len(arr) > 0:
                data_for_boxplot.append(arr)
                subset_labels.append(s)

        if not data_for_boxplot:
            QMessageBox.information(self, "No Data", "No intensities found for the selected subsets.")
            return

        self.box_canvas.figure.clear()
        ax = self.box_canvas.figure.add_subplot(111)

        # Box plot
        ax.boxplot(data_for_boxplot, labels=subset_labels, showfliers=True)
        ax.set_title("Subset Comparison - Box Plot of Intensities")
        ax.set_ylabel("Pixel Intensity")

        self.box_canvas.figure.tight_layout()
        self.box_canvas.draw()

        # Advanced stats
        self.adv_stats_box.clear()
        self.adv_stats_box.append("<b>Advanced Stats</b> (min, 25%, median, 75%, max)\n")

        all_medians = []
        for i, subset_name in enumerate(subset_labels):
            arr = data_for_boxplot[i]
            stats = np.percentile(arr, [0, 25, 50, 75, 100])
            linestr = (
                f"{subset_name} => "
                f"min={stats[0]:.1f}, Q1={stats[1]:.1f}, median={stats[2]:.1f}, "
                f"Q3={stats[3]:.1f}, max={stats[4]:.1f}"
            )
            self.adv_stats_box.append(linestr)
            all_medians.append(stats[2])

        avg_median = np.mean(all_medians)
        self.adv_stats_box.append(f"\nAverage median across chosen subsets: {avg_median:.1f}")

        threshold = 20
        for i, med in enumerate(all_medians):
            if abs(med - avg_median) >= threshold:
                subset_name = subset_labels[i]
                self.adv_stats_box.append(
                    f"<b>Warning:</b> {subset_name} median differs from average by >= {threshold}!"
                )
