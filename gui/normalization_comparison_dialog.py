# File: gui/normalization_comparison_dialog.py

import os
import random
import numpy as np
from PIL import Image

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout,
    QSpinBox, QFileDialog, QMessageBox, QTextEdit
)
from PyQt5.QtCore import Qt

from gui.widgets import MplCanvas
from src.transforms_config import create_transform

# We'll use openpyxl to export .xlsx
# Make sure openpyxl is installed (pip install openpyxl)
import openpyxl


class NormalizationComparisonDialog(QDialog):
    """
    Dialog to compare original (no normalization) vs. min-max vs. z-score
    normalization on the same set of images. Then we can export the chart
    and a .xlsx with numeric data & summary stats.
    """
    def __init__(self, root_folder, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Normalization Comparison")
        self.root_folder = root_folder

        layout = QVBoxLayout(self)

        info_label = QLabel(
            f"<b>Experiment:</b><br>"
            f"We sample images from: <i>{self.root_folder}</i> and compare three methods:<br>"
            f"1) Original intensities (0..255) after resizing<br>"
            f"2) Min-Max normalization (0..1)<br>"
            f"3) Z-score normalization (also re-scaled 0..1 by default)<br><br>"
            f"Use the spinner to set how many images to sample. Then press <b>Run Comparison</b>.<br>"
            f"Below, you can <b>Export Chart</b> to a PNG or <b>Export to Excel</b> for numeric data.<br>"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Controls
        control_layout = QHBoxLayout()
        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(1, 300)
        self.sample_spin.setValue(30)
        control_layout.addWidget(QLabel("Images to Sample:"))
        control_layout.addWidget(self.sample_spin)

        self.btn_run = QPushButton("Run Comparison")
        self.btn_run.clicked.connect(self.on_run_comparison)
        control_layout.addWidget(self.btn_run)

        layout.addLayout(control_layout)

        # Canvas
        self.canvas = MplCanvas(self, width=5, height=4)
        layout.addWidget(self.canvas)

        # Buttons for exporting
        export_layout = QHBoxLayout()
        self.btn_export_png = QPushButton("Export Chart (PNG)")
        self.btn_export_png.clicked.connect(self.on_export_chart)

        self.btn_export_excel = QPushButton("Export to Excel")
        self.btn_export_excel.setToolTip("Export summary stats and histogram data to an Excel file.")
        self.btn_export_excel.clicked.connect(self.on_export_excel)

        export_layout.addWidget(self.btn_export_png)
        export_layout.addWidget(self.btn_export_excel)
        layout.addLayout(export_layout)

        # Text area for final remarks
        self.conclusion_text = QTextEdit()
        self.conclusion_text.setReadOnly(True)
        self.conclusion_text.setHtml(
            "<h3>Conclusion / Observations:</h3>"
            "1) <b>Original (green)</b> intensities typically range [0..255].<br>"
            "2) <b>Min-Max (blue)</b> always strictly in [0..1]. Slightly narrower distribution if images were already bright.<br>"
            "3) <b>Z-Score (red)</b> also re-scaled to [0..1], but normalizes the image's contrast differently.<br>"
            "<br>"
            "Depending on your model input assumptions, one or the other might help<br>"
            "achieve better training stability. Empirically testing is recommended!"
        )
        layout.addWidget(self.conclusion_text)

        self.setLayout(layout)
        self.resize(900, 700)

        # We'll store the intensities in memory for potential exports
        self.intensities_orig = []
        self.intensities_minmax = []
        self.intensities_zscore = []

    def on_run_comparison(self):
        # Gather images
        subfolders = ["Training", "Testing"] if any(x in self.root_folder for x in ["archive", "cleaned", "augmented"]) else [""]
        all_imgs = []
        for subset in subfolders:
            path_sub = os.path.join(self.root_folder, subset)
            if not os.path.isdir(path_sub):
                continue
            classes = [d for d in os.listdir(path_sub) if os.path.isdir(os.path.join(path_sub, d))]
            for c in classes:
                cpath = os.path.join(path_sub, c)
                images = [
                    os.path.join(cpath, f) for f in os.listdir(cpath)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
                all_imgs.extend(images)

        if not all_imgs:
            QMessageBox.warning(self, "No Images", "No images found in the selected folder.")
            return

        random.shuffle(all_imgs)
        sample_count = self.sample_spin.value()
        picked = all_imgs[:sample_count]

        # Build transforms for each method
        t_none = create_transform(normalization=None, image_size=(128,128), augment=False)
        t_minmax = create_transform(normalization="minmax", image_size=(128,128), augment=False)
        t_zscore = create_transform(normalization="zscore", image_size=(128,128), augment=False)

        # Clear old data
        self.intensities_orig = []
        self.intensities_minmax = []
        self.intensities_zscore = []

        error_count = 0

        for img_path in picked:
            try:
                pil_img = Image.open(img_path).convert('L')

                # Original
                orig_tensor = t_none(pil_img)
                # scale from 0..1 up to 0..255 just for histogram consistency
                # Actually, let's keep "original" in [0..255] by reading from the raw PIL directly
                arr_raw = np.array(pil_img.resize((128,128)), dtype=np.float32)
                self.intensities_orig.extend(arr_raw.flatten())

                # Minmax
                mm_tensor = t_minmax(pil_img)
                self.intensities_minmax.extend(mm_tensor.numpy().flatten())

                # Zscore
                zs_tensor = t_zscore(pil_img)
                self.intensities_zscore.extend(zs_tensor.numpy().flatten())

            except Exception as e:
                error_count += 1

        if error_count > 0:
            QMessageBox.information(
                self, "Errors",
                f"{error_count} images failed to load or process."
            )

        # Plot results
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)

        # For consistent bin range, let's set up a custom bin range for each "group":
        # - Original is in [0..255], so let's do bins=50 within that range.
        # - The normalized ones are in [0..1], also 50 bins is fine.

        # We'll just rely on auto-scaling. But we can do 0..255 vs 0..1
        ax.hist(self.intensities_orig, bins=50, alpha=0.4, label="Original", color="green")
        ax.hist(self.intensities_minmax, bins=50, alpha=0.4, label="Min-Max", color="blue")
        ax.hist(self.intensities_zscore, bins=50, alpha=0.4, label="Z-Score", color="red")

        ax.set_title("Distribution of Normalized Intensities\n(Original, Min-Max, Z-Score)")
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Count")
        ax.legend()
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def on_export_chart(self):
        fname, _ = QFileDialog.getSaveFileName(
            self, "Save Chart", "normalization_comparison.png", "PNG Files (*.png);;All Files (*)"
        )
        if not fname:
            return
        try:
            self.canvas.figure.savefig(fname, dpi=150)
            QMessageBox.information(self, "Exported", f"Chart saved to:\n{fname}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save chart:\n{str(e)}")

    def on_export_excel(self):
        """
        Exports:
          1) Summary stats for each method (Original, Min-Max, Z-Score)
          2) Hist bins & counts for each method.
        """
        if not (self.intensities_orig and self.intensities_minmax and self.intensities_zscore):
            QMessageBox.warning(self, "No Data", "Please run the comparison first.")
            return

        fname, _ = QFileDialog.getSaveFileName(
            self, "Save Excel", "normalization_comparison.xlsx", "Excel Files (*.xlsx);;All Files (*)"
        )
        if not fname:
            return

        try:
            wb = openpyxl.Workbook()

            # 1) Summary stats
            ws_summary = wb.active
            ws_summary.title = "SummaryStats"

            # Write header row
            headers = ["Method", "Min", "Max", "Median", "Mean", "Std"]
            ws_summary.append(headers)

            # We'll define a helper to compute stats
            def stats(arr):
                arr = np.array(arr, dtype=np.float32)
                return [
                    float(arr.min()),
                    float(arr.max()),
                    float(np.median(arr)),
                    float(arr.mean()),
                    float(arr.std())
                ]

            data_original = stats(self.intensities_orig)
            data_minmax   = stats(self.intensities_minmax)
            data_zscore   = stats(self.intensities_zscore)

            ws_summary.append(["Original"] + data_original)
            ws_summary.append(["Min-Max"] + data_minmax)
            ws_summary.append(["Z-Score"] + data_zscore)

            # 2) Hist bins & counts for each method in separate sheets
            ws_orig = wb.create_sheet("Histogram_Original")
            ws_minmax = wb.create_sheet("Histogram_Minmax")
            ws_zscore = wb.create_sheet("Histogram_Zscore")

            # We'll define a helper to fill a sheet with (bin_left, bin_right, count)
            def fill_hist_sheet(ws, arr, n_bins=50):
                arr = np.array(arr, dtype=np.float32)
                counts, bin_edges = np.histogram(arr, bins=n_bins)
                # bin_edges has length n_bins+1
                ws.append(["Bin Left", "Bin Right", "Count"])
                for i in range(len(counts)):
                    left = float(bin_edges[i])
                    right = float(bin_edges[i+1])
                    c = int(counts[i])
                    ws.append([left, right, c])

            fill_hist_sheet(ws_orig, self.intensities_orig, n_bins=50)
            fill_hist_sheet(ws_minmax, self.intensities_minmax, n_bins=50)
            fill_hist_sheet(ws_zscore, self.intensities_zscore, n_bins=50)

            # Finally, save
            wb.save(fname)
            QMessageBox.information(self, "Exported", f"Excel data saved to:\n{fname}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save Excel:\n{str(e)}")
