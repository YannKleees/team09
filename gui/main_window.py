import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QStyleFactory
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt

from gui.preprocessing_tab import PreprocessingTab
from gui.augmentation_tab import AugmentationTab
from gui.compare_tab import CompareTab

class MainWindow(QMainWindow):
    """
    Main window with 3 tabs:
      1) Preprocessing
      2) Augmentation
      3) Comparison
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MRI Preprocessing & Augmentation - Refactored GUI")
        self.resize(1200, 800)

        tab_widget = QTabWidget()
        tab_widget.addTab(PreprocessingTab(), "Preprocessing")
        tab_widget.addTab(AugmentationTab(), "Augmentation")
        tab_widget.addTab(CompareTab(),      "Comparison")

        self.setCentralWidget(tab_widget)

        QApplication.setStyle(QStyleFactory.create("Fusion"))
        self._setup_palette()

    def _setup_palette(self):
        """A pastel color theme."""
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(240, 248, 255))  # AliceBlue
        palette.setColor(QPalette.WindowText, Qt.black)
        palette.setColor(QPalette.Base, QColor(255, 250, 240))  # FloralWhite
        palette.setColor(QPalette.AlternateBase, QColor(245, 222, 179))  # Wheat
        palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 225))
        palette.setColor(QPalette.ToolTipText, Qt.black)
        palette.setColor(QPalette.Text, Qt.black)
        palette.setColor(QPalette.Button, QColor(224, 255, 255))  # LightCyan
        palette.setColor(QPalette.ButtonText, Qt.black)
        palette.setColor(QPalette.Highlight, QColor(173, 216, 230)) # LightBlue
        palette.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(palette)
