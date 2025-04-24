# File: gui/main_window.py

import sys
from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QStyleFactory, QApplication,
    QDialog, QVBoxLayout, QLabel, QRadioButton, QHBoxLayout,
    QPushButton
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

# Import your existing tabs
from gui.dashboard_tab import DashboardTab
from gui.preprocessing_tab import PreprocessingTab
from gui.augmentation_tab import AugmentationTab
from gui.compare_tab import CompareTab
from gui.training_tab import TrainingTab
from gui.auto_search_tab import AutoSearchTab
from gui.advanced_eda_tab import AdvancedEdaTab


class ModeSelectionDialog(QDialog):
    """
    A small dialog that lets the user pick "Data Science" vs. "Clinical" mode.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose Application Mode")

        self.selected_mode = None

        layout = QVBoxLayout(self)

        label = QLabel("Select how you want to use this application:")
        layout.addWidget(label)

        self.radio_data = QRadioButton("Data Science / Developer Mode")
        self.radio_data.setChecked(True)  # default
        self.radio_clinical = QRadioButton("Clinical / Inference Mode")

        layout.addWidget(self.radio_data)
        layout.addWidget(self.radio_clinical)

        # Buttons row
        btn_layout = QHBoxLayout()
        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(self.on_ok)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.on_cancel)

        btn_layout.addWidget(btn_ok)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

        self.setLayout(layout)
        self.setModal(True)

    def on_ok(self):
        if self.radio_data.isChecked():
            self.selected_mode = "data_science"
        elif self.radio_clinical.isChecked():
            self.selected_mode = "clinical"
        else:
            self.selected_mode = "data_science"  # fallback
        self.accept()  # close dialog with accept state

    def on_cancel(self):
        self.reject()  # or sys.exit() if you prefer to quit if user cancels


class MainWindow(QMainWindow):
    def __init__(self, mode="data_science"):
        """
        If you already know the mode, pass it in directly.
        Otherwise, you can show ModeSelectionDialog inside here.
        """
        super().__init__()
        self.setWindowTitle("Brain Tumor MRI Toolkit")
        self.setWindowIcon(QIcon())  # set a custom icon if desired
        self.resize(1200, 800)

        # --------------------------------------------------
        # If you want to prompt for mode here, you can do so:
        # --------------------------------------------------
        # If the mode wasn’t pre-passed in, launch the dialog:
        if mode not in ["data_science", "clinical"]:
            dlg = ModeSelectionDialog(self)
            if dlg.exec_() == QDialog.Accepted:
                mode = dlg.selected_mode
            else:
                # user canceled => exit or default to data_science
                mode = "data_science"
        self.mode = mode

        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)

        # Always add the Dashboard tab
        self.tab_widget.addTab(DashboardTab(), "Dashboard")

        # If in clinical mode, we might only want certain tabs:
        if self.mode == "clinical":
            # Example: show just the CompareTab
            self.tab_widget.addTab(CompareTab(), "Comparison")

            # (Optional) If you have a specialized "InferenceTab", you’d add it here.
            # e.g. self.tab_widget.addTab(InferenceTab(), "Inference")

        else:
            # In data_science mode, add all your advanced tabs
            self.tab_widget.addTab(PreprocessingTab(), "Preprocessing")
            self.tab_widget.addTab(AugmentationTab(), "Augmentation")
            self.tab_widget.addTab(CompareTab(), "Comparison")
            self.tab_widget.addTab(TrainingTab(), "Training")
            self.tab_widget.addTab(AutoSearchTab(), "Auto Search")
            self.tab_widget.addTab(AdvancedEdaTab(), "Advanced EDA")

        self.setCentralWidget(self.tab_widget)
        self._setup_modern_style()

    def _setup_modern_style(self):
        self.setStyle(QStyleFactory.create("Fusion"))
        qss = """
        QMainWindow { background-color: #f9f9f9; }
        QWidget { font-family: "Segoe UI"; background-color: #ffffff; font-size: 10pt; }
        QTabWidget::pane { background-color: #ffffff; border: 1px solid #ccc; }
        QTabBar::tab {
            background: #f3f3f3; border: 1px solid #ccc; padding: 8px 14px; margin-right: 2px;
            border-top-left-radius: 6px; border-top-right-radius: 6px; min-width: 120px;
        }
        QTabBar::tab:selected { background: #ffffff; border-bottom: 1px solid #ffffff; }
        QTabBar::tab:hover { background-color: #e8e8e8; }
        QGroupBox {
            border: 1px solid #ddd; border-radius: 8px; margin-top: 10px; padding: 10px;
        }
        QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 8px; }
        QPushButton {
            background-color: #0078d7; color: #ffffff; border: none;
            padding: 8px 14px; border-radius: 6px;
        }
        QPushButton:hover { background-color: #005fa3; }
        QProgressBar {
            border: 1px solid #aaa; border-radius: 4px; text-align: center; height: 20px;
        }
        QProgressBar::chunk { background-color: #0078d7; width: 10px; }
        QTextEdit, QLineEdit, QSpinBox, QComboBox {
            border: 1px solid #ccc; border-radius: 4px; padding: 4px;
        }
        QLabel { font-size: 10pt; }
        """
        self.setStyleSheet(qss)


# If you have a top-level script that launches the app:
def main():
    app = QApplication(sys.argv)
    # Here, you can forcibly pop up the mode selection before building MainWindow:
    dlg = ModeSelectionDialog()
    if dlg.exec_() == QDialog.Accepted:
        mode = dlg.selected_mode
    else:
        mode = "data_science"  # default or exit

    window = MainWindow(mode=mode)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

