import sys
from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QStyleFactory, QApplication,
    QPushButton, QLabel, QLineEdit, QHBoxLayout, QFileDialog,
    QTextEdit, QMessageBox, QVBoxLayout, QGroupBox, QFormLayout,
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
import torch

# Import existing tabs
from gui.dashboard_tab import DashboardTab
from gui.preprocessing_tab import PreprocessingTab
from gui.augmentation_tab import AugmentationTab
from gui.compare_tab import CompareTab
from gui.training_tab import TrainingTab
from gui.auto_search_tab import AutoSearchTab
from gui.advanced_eda_tab import AdvancedEdaTab
from gui.inference_tab import InferenceTab 



class MainWindow(QMainWindow):
    """
    A single-application window that houses all the tabs:
      - Dashboard
      - Preprocessing
      - Augmentation
      - Comparison
      - Training
      - Auto Search
      - Advanced EDA
      - Inference (merged here)
    No mode selection, no placeholders.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Brain Tumor MRI Toolkit")
        self.setWindowIcon(QIcon())  # If you have an icon, set it here or remove entirely
        self.resize(1200, 800)

        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)

        # Add all tabs unconditionally
        self.tab_widget.addTab(DashboardTab(), "Dashboard")
        self.tab_widget.addTab(PreprocessingTab(), "Preprocessing")
        self.tab_widget.addTab(AugmentationTab(), "Augmentation")
        self.tab_widget.addTab(CompareTab(), "Comparison")
        self.tab_widget.addTab(TrainingTab(), "Training")
        self.tab_widget.addTab(AutoSearchTab(), "Hyperparameter Search")
        self.tab_widget.addTab(AdvancedEdaTab(), "Advanced EDA")
        self.tab_widget.addTab(InferenceTab(), "Inference")

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


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
