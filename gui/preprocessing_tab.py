# File: gui/preprocessing_tab.py

import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QFormLayout, QCheckBox,
    QLineEdit, QPushButton, QProgressBar, QMessageBox, QTextEdit,
    QHBoxLayout, QFileDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from logic.preprocessing import preprocess_entire_dataset


class PreprocessingWorker(QThread):
    """
    QThread to handle copying, filtering (blur/sharpen), histogram eq,
    resizing, etc. without blocking the UI.
    """
    progress = pyqtSignal(int, str)
    # We'll now emit 5 items:
    #  dict -> dist_info
    #  list -> error_log
    #  int  -> total_processed
    #  int  -> dups_removed
    #  float-> elapsed_sec
    finished = pyqtSignal(dict, list, int, int, float)

    def __init__(
        self,
        root_folder,
        output_folder,
        resize_val,
        delete_duplicates=False,
        apply_hist_eq=False,
        filter_mode="none",
        interpolation_mode="BILINEAR",
        parent=None
    ):
        super().__init__(parent)
        self.root_folder = root_folder
        self.output_folder = output_folder
        self.resize_val = resize_val
        self.delete_duplicates = delete_duplicates
        self.apply_hist_eq = apply_hist_eq
        self.filter_mode = filter_mode
        self.interpolation_mode = interpolation_mode

    def run(self):
        try:
            def progress_callback(current, total):
                pct = int((current / total) * 100)
                msg = f"Processing image {current} of {total}"
                self.progress.emit(pct, msg)

            self.progress.emit(0, "Starting Preprocessing...")
            (dist_info,
             error_log,
             total_processed,
             dups_removed,
             elapsed_sec) = preprocess_entire_dataset(
                root_dir=self.root_folder,
                dst_root=self.output_folder,
                resize_val=self.resize_val,
                delete_duplicates=self.delete_duplicates,
                apply_hist_eq=self.apply_hist_eq,
                filter_mode=self.filter_mode,
                interpolation=self.interpolation_mode,
                progress_fn=progress_callback
            )
            self.progress.emit(100, "All done!")
            self.finished.emit(dist_info, error_log, total_processed, dups_removed, elapsed_sec)
        except Exception as e:
            msg = f"Error: {str(e)}"
            self.progress.emit(0, msg)
            # For consistency, pass placeholders if an error occurs:
            self.finished.emit({}, [msg], 0, 0, 0.0)


class PreprocessingTab(QWidget):
    """
    Allows the user to:
      1. Pick a raw dataset folder
      2. Pick an output folder
      3. (Optional) Remove duplicates
      4. (Optional) Apply histogram eq
      5. (Optional) Filter mode (blur/sharpen)
      6. Resize images with chosen interpolation
      7. Produce a 'cleaned' version of the data
      8. Advanced Logging
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QVBoxLayout(self)

        doc_label = QLabel(
            "<h3>Dataset Preprocessing</h3>"
            "Steps:<br>"
            "1) Copy the raw dataset (Training & Testing).<br>"
            "2) (Optional) Remove duplicates.<br>"
            "3) (Optional) Blur/Sharpen filters.<br>"
            "4) (Optional) Histogram Equalization (CLAHE).<br>"
            "5) Resize images with chosen interpolation.<br>"
            "Your original data remains untouched."
        )
        doc_label.setWordWrap(True)
        main_layout.addWidget(doc_label)

        # Folder selection group
        folder_box = QGroupBox("Data Folders")
        folder_layout = QFormLayout()

        # Source folder
        self.root_edit = QLineEdit("data/archive")
        self.root_edit.setToolTip("Folder containing 'Training' & 'Testing' subfolders.")
        btn_src_browse = QPushButton("Browse...")
        btn_src_browse.clicked.connect(self.on_browse_src)

        h_src = QHBoxLayout()
        h_src.addWidget(self.root_edit)
        h_src.addWidget(btn_src_browse)
        folder_layout.addRow("Source Folder:", h_src)

        # Output folder
        self.out_edit = QLineEdit("cleaned_data")
        self.out_edit.setToolTip("Destination folder where cleaned data is placed.")
        btn_out_browse = QPushButton("Browse...")
        btn_out_browse.clicked.connect(self.on_browse_out)

        h_out = QHBoxLayout()
        h_out.addWidget(self.out_edit)
        h_out.addWidget(btn_out_browse)
        folder_layout.addRow("Output Folder:", h_out)

        folder_box.setLayout(folder_layout)
        main_layout.addWidget(folder_box)

        # BASIC SETTINGS group
        basic_box = QGroupBox("Basic Settings")
        basic_form = QFormLayout()

        self.resize_combo = QLineEdit("128")  # let user type or pick dimension
        basic_form.addRow("Resize (px):", self.resize_combo)

        # Interpolation mode
        from PyQt5.QtWidgets import QComboBox
        self.interp_combo = QComboBox()
        self.interp_combo.addItems(["BILINEAR", "BICUBIC", "LANCZOS"])
        basic_form.addRow("Interpolation:", self.interp_combo)

        basic_box.setLayout(basic_form)
        main_layout.addWidget(basic_box)

        # ADVANCED SETTINGS group
        adv_box = QGroupBox("Advanced")
        adv_form = QFormLayout()

        self.dup_chk = QCheckBox("Delete duplicates?")
        self.eq_chk = QCheckBox("Apply Histogram Equalization?")

        # Filter combo
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["none", "blur", "sharpen"])
        self.filter_combo.setToolTip("Apply blur or sharpen filter to each image")

        adv_form.addRow(self.dup_chk)
        adv_form.addRow(self.eq_chk)
        adv_form.addRow("Filter Mode:", self.filter_combo)

        # Advanced logging checkbox
        self.adv_log_chk = QCheckBox("Enable Advanced Logging")
        adv_form.addRow(self.adv_log_chk)

        adv_box.setLayout(adv_form)
        main_layout.addWidget(adv_box)

        # Action + Progress
        self.btn_preprocess = QPushButton("Start Preprocessing")
        self.btn_preprocess.setToolTip("Copies dataset, optional eq, filtering, duplicates, etc.")
        self.btn_preprocess.clicked.connect(self.on_start_preprocessing)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        main_layout.addWidget(self.btn_preprocess)
        main_layout.addWidget(self.progress_bar)

        # Log area
        self.log_box = QTextEdit()
        self.log_box.setToolTip("Distribution info & any errors appear here.")
        self.log_box.setReadOnly(True)
        main_layout.addWidget(self.log_box)

        self.setLayout(main_layout)
        self.worker = None

    def on_browse_src(self):
        dlg = QFileDialog(self)
        dlg.setFileMode(QFileDialog.Directory)
        if dlg.exec_():
            folder = dlg.selectedFiles()[0]
            self.root_edit.setText(folder)

    def on_browse_out(self):
        dlg = QFileDialog(self)
        dlg.setFileMode(QFileDialog.Directory)
        if dlg.exec_():
            folder = dlg.selectedFiles()[0]
            self.out_edit.setText(folder)

    def on_start_preprocessing(self):
        root_folder = self.root_edit.text().strip()
        if not os.path.isdir(root_folder):
            QMessageBox.warning(self, "Invalid Folder", f"'{root_folder}' is not a valid directory.")
            return

        output_folder = self.out_edit.text().strip()
        if not output_folder:
            QMessageBox.warning(self, "No Output Folder", "Please specify an output folder.")
            return

        try:
            resize_val = int(self.resize_combo.text().strip())
        except:
            QMessageBox.warning(self, "Invalid Resize", "Resize must be an integer.")
            return

        delete_dup = self.dup_chk.isChecked()
        apply_eq = self.eq_chk.isChecked()
        filter_mode = self.filter_combo.currentText()
        interp_mode = self.interp_combo.currentText()

        self.log_box.clear()
        self.progress_bar.setValue(0)

        self.worker = PreprocessingWorker(
            root_folder=root_folder,
            output_folder=output_folder,
            resize_val=resize_val,
            delete_duplicates=delete_dup,
            apply_hist_eq=apply_eq,
            filter_mode=filter_mode,
            interpolation_mode=interp_mode
        )
        self.worker.progress.connect(self.on_worker_progress)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_worker_progress(self, val, msg):
        self.progress_bar.setValue(val)
        self.progress_bar.setFormat(msg)

    def on_worker_finished(
        self,
        dist_info: dict,
        error_log: list,
        total_processed: int,
        dups_removed: int,
        elapsed_sec: float
    ):
        """
        dist_info: final distribution of images
        error_log: any errors encountered
        total_processed: how many images were processed
        dups_removed: how many duplicates removed
        elapsed_sec: total time spent
        """
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Finished")

        # Basic logs always
        self.log_box.append(f"<b>Total images processed:</b> {total_processed}")

        if dups_removed > 0:
            # "in one phrase" => you can keep it short
            self.log_box.append(f"<b>Duplicates removed in one go:</b> {dups_removed}")

        # Display distribution
        if dist_info:
            self.log_box.append("<br><b>Final distribution in output folder:</b>")
            for subset, sub_dict in dist_info.items():
                sub_total = sum(sub_dict.values())
                self.log_box.append(f"[{subset}] => {sub_total} images total")
                for cls, cnt in sub_dict.items():
                    self.log_box.append(f"   {cls}: {cnt}")
                self.log_box.append("")

        # Show any errors
        if error_log:
            self.log_box.append("<b>Errors during processing:</b>")
            for e in error_log:
                self.log_box.append(e)

        self.log_box.append(f"<i>Elapsed time: {elapsed_sec:.2f} seconds</i>\n")

        # If advanced logging is enabled, add extra explanation
        if self.adv_log_chk.isChecked():
            self.log_box.append("<hr><b>Advanced Logging Summary:</b>")
            self.log_box.append("1) We first copied the original dataset structure into the output folder.")
            if self.dup_chk.isChecked():
                self.log_box.append(
                    "2) We removed duplicates by comparing file hashes, which prevents repeated samples "
                    "from biasing our dataset."
                )
            if self.eq_chk.isChecked():
                self.log_box.append(
                    "3) We applied CLAHE-based Histogram Equalization to enhance contrast for each image."
                )
            fmode = self.filter_combo.currentText()
            if fmode != "none":
                self.log_box.append(
                    f"4) We applied the '{fmode}' filter to each image for potential smoothing or sharpening."
                )
            rz = self.resize_combo.text().strip()
            interp = self.interp_combo.currentText()
            self.log_box.append(
                f"5) Lastly, we resized all images to {rz}x{rz} using {interp} interpolation "
                "so that training uses a consistent input size."
            )
            self.log_box.append("Processing is complete. The dataset is now 'cleaned' and ready.\n")
