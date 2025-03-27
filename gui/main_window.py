import sys
from PyQt5.QtWidgets import QMainWindow, QTabWidget, QStyleFactory
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

from gui.dashboard_tab import DashboardTab
from gui.preprocessing_tab import PreprocessingTab
from gui.augmentation_tab import AugmentationTab
from gui.compare_tab import CompareTab
from gui.training_tab import TrainingTab

class MainWindow(QMainWindow):
    """
    The main application window that contains all tabs:
      1. Dashboard
      2. Preprocessing
      3. Augmentation
      4. Comparison
      5. Training
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Brain Tumor MRI Toolkit - Modern UI")
        # If you have a custom icon file, add its path here
        self.setWindowIcon(QIcon())
        self.resize(1200, 800)

        # Create QTabWidget and add each Tab
        tab_widget = QTabWidget()
        tab_widget.setTabPosition(QTabWidget.North)
        tab_widget.addTab(DashboardTab(),    "Dashboard")
        tab_widget.addTab(PreprocessingTab(), "Preprocessing")
        tab_widget.addTab(AugmentationTab(),  "Augmentation")
        tab_widget.addTab(CompareTab(),       "Comparison")
        tab_widget.addTab(TrainingTab(),      "Training")

        self.setCentralWidget(tab_widget)

        # Apply our modern, white-themed style
        self._setup_modern_style()

    def _setup_modern_style(self):
        # We start with Fusion style as a base
        self.setStyle(QStyleFactory.create("Fusion"))

        # Define a refined QSS for a “white + minimal + modern” look
        qss = """
        QMainWindow {
            background-color: #f9f9f9;
        }
        QWidget {
            font-family: "Segoe UI";
            color: #333;
            background-color: #ffffff;
            font-size: 10pt;
        }
        QTabWidget::pane {
            background-color: #ffffff;
            border: 1px solid #ccc;
        }
        QTabBar::tab {
            background: #f3f3f3;
            border: 1px solid #ccc;
            padding: 8px 14px;
            margin-right: 2px;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            min-width: 100px;
        }
        QTabBar::tab:selected {
            background: #ffffff;
            border-bottom: 1px solid #ffffff;
        }
        QTabBar::tab:hover {
            background-color: #e8e8e8;
        }
        QGroupBox {
            border: 1px solid #ddd;
            border-radius: 8px;
            margin-top: 10px;
            padding: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 8px;
            background-color: transparent;
            font-weight: bold;
        }
        QPushButton {
            background-color: #0078d7;
            color: #ffffff;
            border: none;
            padding: 8px 14px;
            border-radius: 6px;
        }
        QPushButton:hover {
            background-color: #005fa3;
        }
        QProgressBar {
            border: 1px solid #aaa;
            border-radius: 4px;
            text-align: center;
            height: 20px;
        }
        QProgressBar::chunk {
            background-color: #0078d7;
            width: 10px;
        }
        QTextEdit, QLineEdit, QSpinBox, QComboBox {
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 4px;
        }
        QLabel {
            font-size: 10pt;
        }
        """
        self.setStyleSheet(qss)
