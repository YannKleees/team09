import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QHBoxLayout,
    QLineEdit, QPushButton, QFormLayout, QCheckBox, QComboBox,
    QProgressBar, QMessageBox, QTextEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from logic.preprocessing import preprocess_entire_dataset

class PreprocessingWorker(QThread):
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
            self.progress.emit(0, f"Error: {str(e)}")
            self.finished.emit({}, [f"Error: {str(e)}"], 0)

class PreprocessingTab(QWidget):
    """
    A single page with:
      - Project & group info
      - Single-button approach to produce cleaned_data
      - Distribution + error logs
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QVBoxLayout(self)

        # Intro + Group Info
        doc_label = QLabel(
            "<b>Dataset Preprocessing</b><br>"
            "Group Number: Group Ken9<br>"
            "Members: Balázs Ács, Rayan Atieh, Rafal Dorsz, Valters Eglitis, Yann Klees, Ole Rotteveel<br><br>"
            "Click one button to copy the raw dataset (Training & Testing),"
            " optionally remove duplicates, then resize & normalize into <i>cleaned_data</i>.<br>"
            "<b>Raw data remains untouched.</b>"
        )
        doc_label.setWordWrap(True)
        main_layout.addWidget(doc_label)

        # A group box for user options
        opt_box = QGroupBox("Preprocessing Options")
        form_lay = QFormLayout()

        self.root_edit = QLineEdit("data/archive")

        self.resize_combo = QComboBox()
        self.resize_combo.addItems(["128", "160", "224"])

        self.norm_combo = QComboBox()
        self.norm_combo.addItems(["minmax", "zscore"])

        self.dup_chk = QCheckBox("Delete duplicates (in cleaned_data)?")

        self.btn_preprocess = QPushButton("Preprocess Entire Dataset")
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

        # Log area for distribution & errors
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        main_layout.addWidget(self.log_box)

        self.worker = None
        self.setLayout(main_layout)

    def on_start_preprocessing(self):
        root_folder = self.root_edit.text().strip()
        if not os.path.isdir(root_folder):
            QMessageBox.warning(self, "Invalid Folder", f"{root_folder} is not valid.")
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
        """
        Show final distribution & any errors in log_box,
        also finalize the progress bar.
        """
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Finished")

        self.log_box.append(f"Total images processed: {total_processed}")
        if dist_info:
            self.log_box.append("\n<b>Distribution in cleaned_data:</b>")
            for subset in ["Training", "Testing"]:
                if subset in dist_info:
                    subset_dict = dist_info[subset]
                    if isinstance(subset_dict, dict):
                        total_sub = 0
                        for cls, cnt in subset_dict.items():
                            self.log_box.append(f"{subset}/{cls}: {cnt}")
                            total_sub += cnt
                        self.log_box.append(f"Subtotal in {subset}: {total_sub}\n")

        if error_log:
            self.log_box.append("\n<b>Errors:</b>")
            for e in error_log:
                self.log_box.append(e)

        QMessageBox.information(self, "Done", "Preprocessing is complete!")

