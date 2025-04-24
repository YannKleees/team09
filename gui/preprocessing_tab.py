# File: gui/preprocessing_tab.py

import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QFormLayout, QHBoxLayout,
    QLineEdit, QPushButton, QCheckBox, QProgressBar, QMessageBox, QTextEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from logic.preprocessing import preprocess_entire_dataset


class PreprocessingWorker(QThread):
    """
    QThread to handle copying, resizing, and normalization
    without blocking the UI.
    """
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict, list, int)  # dist_info, error_log, total_processed

    def __init__(self, root_folder, resize_val, norm_method, delete_duplicates=False, parent=None):
        super().__init__(parent)
        self.root_folder = root_folder
        self.resize_val = resize_val
        self.norm_method = norm_method
        self.delete_duplicates = delete_duplicates

    def run(self):
        try:
            self.progress.emit(0, "Copying dataset & preparing to preprocess...")
            dist_info, error_log, total_processed, dups_removed, elapsed_sec = preprocess_entire_dataset(
                root_dir=self.root_folder,
                dst_root="cleaned_data",
                resize_val=self.resize_val,
                norm_method=self.norm_method,
                delete_duplicates=self.delete_duplicates
            )
            self.progress.emit(100, "All done!")
            self.finished.emit(dist_info, error_log, total_processed)
        except Exception as e:
            msg = f"Error: {str(e)}"
            self.progress.emit(0, msg)
            self.finished.emit({}, [msg], 0)


class PreprocessingTab(QWidget):
    """
    Allows the user to:
      1. Specify a raw dataset folder
      2. Configure resizing & normalization (Basic)
      3. Optionally remove duplicates (Advanced)
      4. Produce a 'cleaned_data' folder
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QVBoxLayout(self)

        doc_label = QLabel(
            "<h3>Dataset Preprocessing</h3>"
            "Steps:<br>"
            "1) Copy the raw dataset (Training & Testing) to <i>cleaned_data</i>.<br>"
            "2) (Optional) Remove duplicates.<br>"
            "3) Resize & normalize images.<br>"
            "Your original data remains untouched."
        )
        doc_label.setWordWrap(True)
        main_layout.addWidget(doc_label)

        # -----------------------
        # BASIC SETTINGS group
        # -----------------------
        basic_box = QGroupBox("Basic Settings")
        basic_form = QFormLayout()

        self.root_edit = QLineEdit("data/archive")
        self.root_edit.setToolTip("Folder containing 'Training' & 'Testing' subfolders.")

        self.resize_combo = QLineEdit("128")  # let user type in or pick
        self.norm_combo = QLineEdit("zscore")  # or "minmax"

        basic_form.addRow("Source Folder:", self.root_edit)
        basic_form.addRow("Resize (px):", self.resize_combo)
        basic_form.addRow("Normalization:", self.norm_combo)
        basic_box.setLayout(basic_form)
        main_layout.addWidget(basic_box)

        # -----------------------
        # ADVANCED SETTINGS group
        # -----------------------
        adv_box = QGroupBox("Advanced")
        adv_form = QFormLayout()

        self.dup_chk = QCheckBox("Delete duplicates?")
        self.dup_chk.setToolTip("If checked, duplicates are removed in 'cleaned_data' before resizing.")

        adv_form.addRow(self.dup_chk)
        adv_box.setLayout(adv_form)
        main_layout.addWidget(adv_box)

        # -----------------------
        # ACTION + PROGRESS
        # -----------------------
        self.btn_preprocess = QPushButton("Start Preprocessing")
        self.btn_preprocess.setToolTip("Copies & normalizes dataset into 'cleaned_data'.")
        self.btn_preprocess.clicked.connect(self.on_start_preprocessing)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        main_layout.addWidget(self.btn_preprocess)
        main_layout.addWidget(self.progress_bar)

        # Log area
        self.log_box = QTextEdit()
        self.log_box.setToolTip("Distribution info & any errors will appear here.")
        self.log_box.setReadOnly(True)
        main_layout.addWidget(self.log_box)

        self.setLayout(main_layout)
        self.worker = None

    def on_start_preprocessing(self):
        root_folder = self.root_edit.text().strip()
        if not os.path.isdir(root_folder):
            QMessageBox.warning(self, "Invalid Folder", f"'{root_folder}' is not a valid directory.")
            return

        try:
            resize_val = int(self.resize_combo.text().strip())
        except:
            QMessageBox.warning(self, "Invalid Resize", "Resize must be an integer.")
            return

        norm_method = self.norm_combo.text().strip().lower()
        if norm_method not in ["zscore", "minmax", "none"]:
            QMessageBox.warning(self, "Invalid Normalization", "Use 'zscore', 'minmax', or 'none'.")
            return

        delete_dup = self.dup_chk.isChecked()

        self.log_box.clear()
        self.worker = PreprocessingWorker(
            root_folder=root_folder,
            resize_val=resize_val,
            norm_method=(None if norm_method == "none" else norm_method),
            delete_duplicates=delete_dup
        )
        self.worker.progress.connect(self.on_worker_progress)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_worker_progress(self, val, msg):
        self.progress_bar.setValue(val)
        self.progress_bar.setFormat(msg)

    def on_worker_finished(self, dist_info, error_log, total_processed):
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Finished")

        self.log_box.append(f"<b>Total images processed:</b> {total_processed}")
        if dist_info:
            self.log_box.append("<br><b>Distribution in 'cleaned_data':</b>")
            for subset in ["Training", "Testing"]:
                if subset in dist_info:
                    subset_dict = dist_info[subset]
                    if isinstance(subset_dict, dict):
                        sub_total = 0
                        for cls, cnt in subset_dict.items():
                            self.log_box.append(f"{subset}/{cls}: {cnt}")
                            sub_total += cnt
                        self.log_box.append(f"Subtotal in {subset}: {sub_total}\n")

        if error_log:
            self.log_box.append("<b>Errors:</b>")
            for e in error_log:
                self.log_box.append(e)
