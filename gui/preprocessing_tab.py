import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QFormLayout, QHBoxLayout,
    QLineEdit, QPushButton, QCheckBox, QComboBox,
    QProgressBar, QMessageBox, QTextEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from logic.preprocessing import preprocess_entire_dataset

class PreprocessingWorker(QThread):
    """
    QThread to handle the entire copying, resizing, normalization steps
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
        """
        Perform the entire dataset preprocessing pipeline.
        """
        try:
            self.progress.emit(0, "Copying dataset & preparing to preprocess...")
            dist_info, error_log, total_processed = preprocess_entire_dataset(
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
      2. Configure resizing & normalization
      3. Optionally remove duplicates
      4. Produce a 'cleaned_data' folder
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QVBoxLayout(self)

        doc_label = QLabel(
            "<h3>Dataset Preprocessing</h3>"
            "Group Ken9: Balázs Ács, Rayan Atieh, Rafal Dorsz, Valters Eglitis, Yann Klees, Ole Rotteveel<br>"
            "<b>Steps:</b><br>"
            "1) Copy the raw dataset (Training & Testing) to <i>cleaned_data</i>.<br>"
            "2) (Optional) Remove duplicates.<br>"
            "3) Resize & normalize images.<br>"
            "Your original data remains untouched."
        )
        doc_label.setWordWrap(True)
        main_layout.addWidget(doc_label)

        # Options Group
        opt_box = QGroupBox("Preprocessing Options")
        form_lay = QFormLayout()

        self.root_edit = QLineEdit("data/archive")
        self.root_edit.setToolTip("Folder containing 'Training' & 'Testing' subfolders.")

        self.resize_combo = QComboBox()
        self.resize_combo.addItems(["128", "160", "224"])
        self.resize_combo.setToolTip("Select target image size for resizing.")

        self.norm_combo = QComboBox()
        self.norm_combo.addItems(["minmax", "zscore"])
        self.norm_combo.setToolTip("Choose normalization method.")

        self.dup_chk = QCheckBox("Delete duplicates?")
        self.dup_chk.setToolTip("If checked, duplicates are removed in cleaned_data before resizing.")

        self.btn_preprocess = QPushButton("Preprocess Entire Dataset")
        self.btn_preprocess.setToolTip("Starts copying & normalizing the dataset into 'cleaned_data'.")
        self.btn_preprocess.clicked.connect(self.on_start_preprocessing)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        form_lay.addRow("Source Folder:", self.root_edit)
        form_lay.addRow("Resize:", self.resize_combo)
        form_lay.addRow("Normalization:", self.norm_combo)
        form_lay.addRow(self.dup_chk)
        form_lay.addRow(self.btn_preprocess)
        form_lay.addRow(self.progress_bar)

        opt_box.setLayout(form_lay)
        main_layout.addWidget(opt_box)

        # Log area
        self.log_box = QTextEdit()
        self.log_box.setToolTip("Detailed logs about distribution and errors.")
        self.log_box.setReadOnly(True)
        main_layout.addWidget(self.log_box)

        self.setLayout(main_layout)
        self.worker = None

    def on_start_preprocessing(self):
        root_folder = self.root_edit.text().strip()
        if not os.path.isdir(root_folder):
            QMessageBox.warning(self, "Invalid Folder", f"'{root_folder}' is not a valid directory.")
            return

        resize_val = int(self.resize_combo.currentText())
        norm_method = self.norm_combo.currentText()
        delete_dup = self.dup_chk.isChecked()

        self.log_box.clear()
        self.worker = PreprocessingWorker(
            root_folder=root_folder,
            resize_val=resize_val,
            norm_method=norm_method,
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
