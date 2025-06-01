"""
auto_search_tab.py

Updated HPC search that removes the old "disable_aug" and "adv_aug" parameters,
removes any references to "resize", and also removes "double" from the HPC model variants
so we don't train it. We only keep 'binary', 'simple', 'threeclass'.
Includes a set of checkboxes for model variants, plus typical HPC space
(dropout, weight_decay, earlystop, class_weight).
Everything else (progress display, final training, HPC info) is the same.
"""

import os
import json
import random
import copy
import torch
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QGroupBox, QFormLayout,
    QLabel, QPushButton, QComboBox, QLineEdit, QSpinBox,
    QProgressBar, QTextEdit, QMessageBox, QHBoxLayout, QTableWidget,
    QTableWidgetItem, QHeaderView, QFileDialog, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from gui.widgets import MplCanvas

from src.experiment import Experiment

_HPC_RESULTS_JSON = "hpc_results.json"


def save_hpc_results_json(all_results, path=_HPC_RESULTS_JSON):
    """
    Saves HPC combos (confmat/ROC/metrics) to a JSON file so they persist across sessions.
    """
    import json
    data_to_save = []
    for rec in all_results:
        # Convert confusion matrix, if any
        confmat_tensor = rec.get("confmat", None)
        if isinstance(confmat_tensor, torch.Tensor):
            cmat_list = confmat_tensor.cpu().tolist()
        else:
            cmat_list = None

        # Convert any ROC arrays to lists
        roc_obj = rec.get("roc", None)
        if roc_obj:
            fpr = roc_obj.get("fpr", [])
            tpr = roc_obj.get("tpr", [])
            if isinstance(fpr, np.ndarray):
                fpr = fpr.tolist()
            if isinstance(tpr, np.ndarray):
                tpr = tpr.tolist()
            roc_obj = {
                "fpr": fpr,
                "tpr": tpr,
                "auc": roc_obj.get("auc", None)
            }
        else:
            roc_obj = None

        data_to_save.append({
            "combo_str":     rec.get("combo_str", ""),
            "val_acc":       rec.get("val_acc", 0.0),
            "model_path":    rec.get("model_path", ""),
            "model_variant": rec.get("model_variant", ""),
            "confmat_list":  cmat_list,
            "roc":           roc_obj,
            "metrics":       rec.get("metrics", None)
        })

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2)
    except Exception as e:
        print(f"[WARN] Could not save HPC results to {path}: {e}")


def load_hpc_results_json(path=_HPC_RESULTS_JSON):
    if not os.path.isfile(path):
        return []
    if os.path.getsize(path) == 0:
        print(f"[WARN] {path} is empty. Return no HPC combos.")
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                print(f"[WARN] {path} is empty (or whitespace). Return no HPC combos.")
                return []
            data = json.loads(content)
    except Exception as e:
        print(f"[WARN] Could not load HPC combos from {path}: {e}")
        return []

    results = []
    for item in data:
        cmat_list = item.get("confmat_list", None)
        if cmat_list is not None:
            cmat_tensor = torch.tensor(cmat_list, dtype=torch.int32)
        else:
            cmat_tensor = None

        results.append({
            "combo_str":     item.get("combo_str", ""),
            "val_acc":       item.get("val_acc", 0.0),
            "model_path":    item.get("model_path", ""),
            "model_variant": item.get("model_variant", ""),
            "confmat":       cmat_tensor,
            "roc":           item.get("roc", None),
            "metrics":       item.get("metrics", None)
        })
    return results


class HPCSearchWorker(QThread):
    """
    Conduct HPC with partial training for each combo (no "double" variant).
    We store best val_acc, confmat, ROC, metrics, plus the saved model path.
    """
    progress = pyqtSignal(int)
    log_msg = pyqtSignal(str)
    combo_epoch_update = pyqtSignal(int, int, float, float)  # combo_idx, epoch, train_acc, val_acc
    finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_root = "cleaned_data"
        self.model_variants = []
        self.dropout_list = []
        self.weight_decay_list = []
        self.earlystop_list = []
        self.class_weight_list = []
        self.batch_size = 16
        self.lr = 0.001
        self.mini_epochs = 2
        self.device = "cpu"

        self.best_acc = -1.0
        self.best_params_str = ""
        self.all_results = []

    def setup_hpc(
        self,
        data_root,
        model_variants,
        dropout_list,
        weight_decay_list,
        earlystop_list,
        class_weight_list,
        batch_size,
        lr,
        mini_epochs,
        device="cpu"
    ):
        self.data_root = data_root
        self.model_variants = model_variants  # e.g. ["binary","simple","threeclass"] (no double)
        self.dropout_list = dropout_list
        self.weight_decay_list = weight_decay_list
        self.earlystop_list = earlystop_list
        self.class_weight_list = class_weight_list
        self.batch_size = batch_size
        self.lr = lr
        self.mini_epochs = mini_epochs
        self.device = device

        self.best_acc = -1.0
        self.best_params_str = ""
        self.all_results = []

    def run(self):
        import itertools
        param_space = list(itertools.product(
            self.model_variants,
            self.dropout_list,
            self.weight_decay_list,
            self.earlystop_list,
            self.class_weight_list
        ))
        total_combos = len(param_space)
        if total_combos == 0:
            self.log_msg.emit("No HPC combos specified!")
            self.finished.emit()
            return

        self.log_msg.emit(f"Starting HPC with {total_combos} combos, each ~{self.mini_epochs} epoch(s)...")

        os.makedirs("hpc_checkpoints", exist_ok=True)
        from src.experiment import Experiment

        for combo_idx, combo in enumerate(param_space):
            model_var   = combo[0]
            drop        = combo[1]
            wdec        = combo[2]
            estop       = combo[3]
            cw          = combo[4]

            combo_str = (
                f"model={model_var}, drop={drop}, "
                f"wdec={wdec}, eStop={estop}, cw={cw}"
            )
            self.log_msg.emit(f"\n[Combo {combo_idx+1}/{total_combos}] => {combo_str}")

            try:
                exp = Experiment(
                    data_root=self.data_root,
                    model_variant=model_var,
                    input_size=128,
                    batch_size=self.batch_size,
                    lr=self.lr,
                    epochs=self.mini_epochs,
                    disable_augment=True,   # removed references to adv_aug
                    adv_aug=False,          # removed references to adv_aug
                    dropout=drop,
                    weight_decay=wdec,
                    earlystop_patience=estop,
                    use_class_weight=cw,
                    device=self.device
                )
                exp.load_data()
                if hasattr(exp, "train_dataset") and getattr(exp.train_dataset, "is_empty", False):
                    self.log_msg.emit("  -> Skipping: dataset is EMPTY.")
                    self.all_results.append({
                        "combo_str": combo_str,
                        "val_acc": 0.0,
                        "confmat": None,
                        "roc": None,
                        "metrics": None,
                        "model_path": None,
                        "model_variant": model_var
                    })
                    continue

                exp._setup_optimizer_criterion()

            except Exception as e:
                self.log_msg.emit(f"Error building experiment: {e}")
                self.all_results.append({
                    "combo_str": combo_str,
                    "val_acc": 0.0,
                    "confmat": None,
                    "roc": None,
                    "metrics": None,
                    "model_path": None,
                    "model_variant": model_var
                })
                continue

            best_val_for_combo = 0.0
            best_state_dict = None
            best_confmat = None
            best_roc = None
            best_metrics = None
            no_improve_count = 0

            for ep in range(self.mini_epochs):
                train_loss, train_acc, _ = exp._train_one_epoch()
                val_metrics = exp._evaluate()
                val_acc = val_metrics["accuracy"]

                if val_acc > best_val_for_combo:
                    best_val_for_combo = val_acc
                    import copy
                    best_state_dict = copy.deepcopy(exp.model.state_dict())
                    best_confmat = val_metrics.get("confusion_matrix")
                    best_roc = {
                        "fpr": val_metrics.get("roc_fpr"),
                        "tpr": val_metrics.get("roc_tpr"),
                        "auc": val_metrics.get("roc_auc")
                    }
                    best_metrics = exp._compute_class_metrics(val_metrics)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if estop > 0 and no_improve_count >= estop:
                        self.log_msg.emit(f"  -> EarlyStop triggered at epoch {ep+1}")
                        break

                self.combo_epoch_update.emit(combo_idx, ep+1, train_acc, val_acc)
                self.log_msg.emit(f"  Ep {ep+1}/{self.mini_epochs}: train={train_acc:.1f}%, val={val_acc:.1f}%")

            if best_state_dict is not None:
                model_path = os.path.join("hpc_checkpoints", f"hpc_model_combo_{combo_idx+1}.pth")
                torch.save(best_state_dict, model_path)
            else:
                model_path = None

            self.all_results.append({
                "combo_str": combo_str,
                "val_acc": best_val_for_combo,
                "confmat": best_confmat,
                "roc": best_roc,
                "metrics": best_metrics,
                "model_path": model_path,
                "model_variant": model_var
            })

            if best_val_for_combo > self.best_acc:
                self.best_acc = best_val_for_combo
                self.best_params_str = combo_str

            pct = int(100*(combo_idx+1) / total_combos)
            self.progress.emit(pct)

        self.log_msg.emit(f"\nHPC DONE. Best val_acc={self.best_acc:.2f}% => {self.best_params_str}")
        self.finished.emit()


class FinalTrainWorker(QThread):
    """
    Performs final training on the best HPC combo.
    """
    finishedDetailed = pyqtSignal(float, dict, object, dict)
    progress = pyqtSignal(int)
    log_msg = pyqtSignal(str)
    epoch_update = pyqtSignal(int, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.best_params_str = ""
        self.data_root = "cleaned_data"
        self.epochs = 10
        self.batch = 32
        self.lr = 0.001
        self.folds = 1
        self.device = "cpu"

    def setup_final(self, best_params_str, data_root, epochs, batch, lr, folds, device="cpu"):
        self.best_params_str = best_params_str
        self.data_root = data_root
        self.epochs = epochs
        self.batch = batch
        self.lr = lr
        self.folds = folds
        self.device = device

    def run(self):
        if not self.best_params_str:
            self.log_msg.emit("No HPC combo chosen for final training.")
            self.finishedDetailed.emit(0.0, {}, None, {})
            return

        # parse param string, e.g. "model=..., drop=..., wdec=..., eStop=..., cw=..."
        parts = self.best_params_str.split(",")
        param_dict = {}
        for p in parts:
            if "=" in p:
                kv = p.split("=")
                if len(kv) == 2:
                    k, v = kv
                    param_dict[k.strip()] = v.strip()

        model_var = param_dict.get("model", "binary")
        drop       = float(param_dict.get("drop", "0.0"))
        wdec       = float(param_dict.get("wdec", "0.0"))
        estop      = int(param_dict.get("eStop", "0"))
        cw         = (param_dict.get("cw", "False") == "True")

        self.log_msg.emit(f"[Final Training] Using => {self.best_params_str}")
        self.log_msg.emit(f"Folds={self.folds}, epochs={self.epochs}, batch={self.batch}, lr={self.lr}")

        try:
            exp = Experiment(
                data_root=self.data_root,
                model_variant=model_var,
                input_size=128,
                batch_size=self.batch,
                lr=self.lr,
                epochs=self.epochs,
                disable_augment=True,
                adv_aug=False,
                dropout=drop,
                weight_decay=wdec,
                earlystop_patience=estop,
                use_class_weight=cw,
                device=self.device,
                folds=self.folds
            )
            exp.set_logger(self.log_msg.emit)
        except Exception as e:
            self.log_msg.emit(f"Failed building final experiment: {e}")
            self.progress.emit(0)
            self.finishedDetailed.emit(0.0, {}, None, {})
            return

        exp.load_data()
        if hasattr(exp, "train_dataset") and getattr(exp.train_dataset, "is_empty", False):
            self.log_msg.emit("Train dataset is empty => abort.")
            self.finishedDetailed.emit(0.0, {}, None, {})
            return

        exp._setup_optimizer_criterion()

        final_acc = 0.0
        if self.folds <= 1:
            best_val = 0.0
            no_improve = 0
            for ep in range(self.epochs):
                tr_loss, tr_acc, _ = exp._train_one_epoch()
                val_metrics = exp._evaluate()
                val_acc = val_metrics["accuracy"]

                if val_acc > best_val:
                    best_val = val_acc
                    no_improve = 0
                else:
                    no_improve += 1

                self.epoch_update.emit(ep+1, tr_acc, val_acc)
                pct = int(100*(ep+1)/self.epochs)
                self.progress.emit(pct)
                self.log_msg.emit(f"Ep {ep+1}: train={tr_acc:.1f}%, val={val_acc:.1f}%")

                if estop>0 and no_improve>=estop:
                    self.log_msg.emit(f"[EarlyStop at ep {ep+1}]")
                    break

            final_acc = best_val
        else:
            avg_acc, _, _, _ = exp.run_kfold(
                folds=self.folds,
                epochs=self.epochs
            )
            final_acc = avg_acc
            self.log_msg.emit(f"K-Fold => avg_val_acc={avg_acc:.2f}%")
            self.progress.emit(100)

        final_roc = exp.final_roc
        final_confmat = exp.final_confmat
        final_metrics = exp.final_class_metrics
        self.finishedDetailed.emit(final_acc, final_roc, final_confmat, final_metrics)


class AutoSearchTab(QWidget):
    """
    HPC search for (binary, simple, threeclass) but NOT double,
    plus final training of the best combo, storing confusion matrix & ROC.
    """

    hpc_updated = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setStyleSheet("""
            QWidget#AutoSearchTab {
                background-color: #f9f9f9;
            }
            QGroupBox {
                border: 1px solid #cccccc;
                border-radius: 6px;
                margin-top: 10px;
                padding: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 4px 10px;
                color: #0078D7;
            }
            QLabel {
                font-size: 10pt;
            }
            QPushButton {
                background-color: #0078d7;
                color: #ffffff;
                border: none;
                padding: 6px 14px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #005fa3;
            }
            QLineEdit, QTextEdit, QSpinBox, QComboBox {
                font-size: 9pt;
                padding: 4px;
            }
            QProgressBar {
                height: 20px;
                border-radius: 4px;
                text-align: center;
            }
            QTableWidget {
                font-size: 9pt;
            }
        """)
        self.setObjectName("AutoSearchTab")

        main_layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        cont_layout = QVBoxLayout(container)
        cont_layout.setContentsMargins(15,15,15,15)
        cont_layout.setSpacing(20)

        # HPC group
        hpc_group = QGroupBox("Hyperparameter Search (HPC)")
        hpc_form = QFormLayout(hpc_group)

        # Dataset folder
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["cleaned_data", "unprocessed_data", "augmented_data"])
        hpc_form.addRow("Dataset Folder:", self.dataset_combo)

        # Model variants => checkboxes, but NO double.
        self.model_ck_binary = QCheckBox("binary")
        self.model_ck_simple  = QCheckBox("simple")
        self.model_ck_threeclass = QCheckBox("threeclass")

        model_ck_layout = QHBoxLayout()
        model_ck_layout.addWidget(self.model_ck_binary)
        model_ck_layout.addWidget(self.model_ck_simple)
        model_ck_layout.addWidget(self.model_ck_threeclass)
        hpc_form.addRow(QLabel("Model Variants:"), model_ck_layout)

        # Dropout
        self.dropout_edit = QLineEdit("0.0,0.3")
        hpc_form.addRow("Dropout List:", self.dropout_edit)

        # Weight decay
        self.wdec_edit = QLineEdit("0.0,0.0001")
        hpc_form.addRow("Weight Decay List:", self.wdec_edit)

        # Early stop
        self.es_edit = QLineEdit("0,3")
        hpc_form.addRow("EarlyStop List:", self.es_edit)

        # Class Weight
        self.cw_edit = QLineEdit("False,True")
        hpc_form.addRow("Class Weight (List):", self.cw_edit)

        # batch size
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1,512)
        self.batch_spin.setValue(16)
        hpc_form.addRow("Batch Size:", self.batch_spin)

        # LR
        self.lr_edit = QLineEdit("0.001")
        hpc_form.addRow("Learning Rate:", self.lr_edit)

        # HPC mini-epochs
        self.mini_epoch_spin = QSpinBox()
        self.mini_epoch_spin.setRange(1,20)
        self.mini_epoch_spin.setValue(2)
        hpc_form.addRow("Mini-Epochs (partial):", self.mini_epoch_spin)

        self.transferlearn_model = QComboBox()
        self.transferlearn_model.addItems(["None","ResNet18"])
        hpc_form.addRow("Transfer-learning:", self.transferlearn_model)

        self.transferlearn_mode = QComboBox()
        self.transferlearn_mode.addItems(["finetuning","feature extraction"])
        hpc_form.addRow("Transfer-learning:", self.transferlearn_mode)

        self.btn_hpc = QPushButton("Run HPC Search")
        self.btn_hpc.clicked.connect(self.on_run_hpc)
        hpc_form.addRow(self.btn_hpc)

        cont_layout.addWidget(hpc_group)

        # HPC progress & log
        self.hpc_progress = QProgressBar()
        self.hpc_progress.setValue(0)
        self.hpc_log = QTextEdit()
        self.hpc_log.setReadOnly(True)
        self.hpc_log.setMinimumHeight(180)

        cont_layout.addWidget(self.hpc_progress)
        cont_layout.addWidget(self.hpc_log)

        self.hpc_canvas = MplCanvas(self, width=8, height=4, dpi=100)
        self.hpc_canvas.setMinimumSize(800, 400)
        self.hpc_toolbar = NavigationToolbar(self.hpc_canvas, self)

        cont_layout.addWidget(QLabel("<b>Current HPC Combo: Train vs. Val Accuracy</b>"))
        cont_layout.addWidget(self.hpc_canvas)
        cont_layout.addWidget(self.hpc_toolbar)

        # HPC Results Overview
        results_group = QGroupBox("HPC Results Overview")
        rg_layout = QVBoxLayout(results_group)

        self.hpc_table = QTableWidget()
        self.hpc_table.setColumnCount(4)
        self.hpc_table.setHorizontalHeaderLabels(["Rank","Combo","Val Acc","Model?"])
        self.hpc_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.hpc_table.setMinimumHeight(800)  # <--- bigger table
        rg_layout.addWidget(self.hpc_table)

        h_btn_row = QHBoxLayout()
        self.btn_show_info = QPushButton("Show HPC Info (Selected)")
        self.btn_show_info.clicked.connect(self.on_show_hpc_info)
        h_btn_row.addWidget(self.btn_show_info)

        self.btn_delete_model = QPushButton("Delete HPC Model (Selected)")
        self.btn_delete_model.clicked.connect(self.on_delete_model)
        h_btn_row.addWidget(self.btn_delete_model)

        rg_layout.addLayout(h_btn_row)

        self.hpc_info_text = QTextEdit()
        self.hpc_info_text.setReadOnly(True)
        self.hpc_info_text.setMinimumHeight(200)  # <--- bigger info text
        rg_layout.addWidget(self.hpc_info_text)

        self.hpc_bar_canvas = MplCanvas(self, width=8, height=3, dpi=100)
        self.hpc_bar_canvas.setMinimumSize(800, 450)  # <--- larger
        self.hpc_bar_toolbar = NavigationToolbar(self.hpc_bar_canvas, self)
        rg_layout.addWidget(QLabel("<b>HPC Combos: Validation Accuracy</b>"))
        rg_layout.addWidget(self.hpc_bar_canvas)
        rg_layout.addWidget(self.hpc_bar_toolbar)

        self.hpc_auc_canvas = MplCanvas(self, width=8, height=3, dpi=100)
        self.hpc_auc_canvas.setMinimumSize(800, 450)  # <--- larger
        self.hpc_auc_toolbar = NavigationToolbar(self.hpc_auc_canvas, self)
        rg_layout.addWidget(QLabel("<b>HPC Combos: ROC AUC (Binary Only)</b>"))
        rg_layout.addWidget(self.hpc_auc_canvas)
        rg_layout.addWidget(self.hpc_auc_toolbar)

        cont_layout.addWidget(results_group)

        # Final Training group
        final_box = QGroupBox("Final Training (Use HPC Combo)")
        final_form = QFormLayout(final_box)

        self.final_epochs_spin = QSpinBox()
        self.final_epochs_spin.setRange(1,200)
        self.final_epochs_spin.setValue(10)
        final_form.addRow("Final Epochs:", self.final_epochs_spin)

        self.final_batch_spin = QSpinBox()
        self.final_batch_spin.setRange(1,512)
        self.final_batch_spin.setValue(32)
        final_form.addRow("Final Batch Size:", self.final_batch_spin)

        self.final_lr_edit = QLineEdit("0.001")
        final_form.addRow("Final LR:", self.final_lr_edit)

        self.final_folds_spin = QSpinBox()
        self.final_folds_spin.setRange(1,10)
        self.final_folds_spin.setValue(1)
        final_form.addRow("Folds (K-Fold):", self.final_folds_spin)

        self.btn_final = QPushButton("Train Final Model")
        self.btn_final.setEnabled(False)
        self.btn_final.clicked.connect(self.on_train_final)
        final_form.addRow(self.btn_final)

        cont_layout.addWidget(final_box)

        self.final_progress = QProgressBar()
        self.final_progress.setValue(0)
        self.final_log = QTextEdit()
        self.final_log.setReadOnly(True)
        self.final_log.setMinimumHeight(100)
        cont_layout.addWidget(self.final_progress)
        cont_layout.addWidget(self.final_log)

        self.final_canvas = MplCanvas(container, width=8, height=4)
        self.final_canvas.setMinimumSize(800, 400)
        self.final_toolbar = NavigationToolbar(self.final_canvas, self)
        cont_layout.addWidget(QLabel("<b>Final Training: Train vs. Val Accuracy</b>"))
        cont_layout.addWidget(self.final_canvas)
        cont_layout.addWidget(self.final_toolbar)

        final_res_g = QGroupBox("Final Results: Confusion Matrix & ROC")
        fres_layout = QHBoxLayout(final_res_g)

        self.confmat_canvas = MplCanvas(self, width=4, height=3)
        self.confmat_canvas.setMinimumSize(400, 300)
        self.confmat_toolbar = NavigationToolbar(self.confmat_canvas, self)
        v_left = QVBoxLayout()
        v_left.addWidget(QLabel("<b>Confusion Matrix</b>"))
        v_left.addWidget(self.confmat_canvas)
        v_left.addWidget(self.confmat_toolbar)
        fres_layout.addLayout(v_left)

        self.roc_canvas = MplCanvas(self, width=4, height=3)
        self.roc_canvas.setMinimumSize(400, 300)
        self.roc_toolbar = NavigationToolbar(self.roc_canvas, self)
        v_right = QVBoxLayout()
        v_right.addWidget(QLabel("<b>ROC Curve (Binary/Double)</b>"))
        v_right.addWidget(self.roc_canvas)
        v_right.addWidget(self.roc_toolbar)
        fres_layout.addLayout(v_right)

        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        fres_layout.addWidget(self.metrics_text)

        cont_layout.addWidget(final_res_g)

        scroll.setWidget(container)
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

        self.hpc_worker = None
        self.current_combo_idx = -1
        self.hpc_train_acc = []
        self.hpc_val_acc = []
        self.best_params_str = ""
        self.all_results = []

        self.final_worker = None
        self.final_epoch_list = []
        self.final_train_acc_list = []
        self.final_val_acc_list = []
        self.final_roc_data = None
        self.final_confmat_data = None
        self.final_metrics_data = None

        # load existing HPC combos from JSON
        self._load_existing_hpc_results()

    def _load_existing_hpc_results(self):
        loaded = load_hpc_results_json()
        if loaded:
            self.all_results = []
            for item in loaded:
                rec = {
                    "combo_str":     item.get("combo_str",""),
                    "val_acc":       item.get("val_acc",0.0),
                    "model_path":    item.get("model_path",""),
                    "model_variant": item.get("model_variant",""),
                    "metrics":       item.get("metrics",None),
                    "roc":           item.get("roc",None),
                    "confmat":       item.get("confmat",None)
                }
                self.all_results.append(rec)

            self._update_hpc_table()
            self._plot_hpc_combo_chart()
            self._plot_hpc_auc_chart()

            best = max(self.all_results, key=lambda r: r["val_acc"], default=None)
            if best:
                self.best_params_str = best["combo_str"]
                self.hpc_log.append(f"Loaded HPC combos from JSON. Best => {self.best_params_str}")
                self.btn_final.setEnabled(True)

    def _save_current_hpc_results(self):
        save_hpc_results_json(self.all_results)

    def on_run_hpc(self):
        self.hpc_log.clear()
        self.hpc_progress.setValue(0)
        self.hpc_info_text.clear()
        self.hpc_table.clearContents()
        self.hpc_table.setRowCount(0)
        self.hpc_train_acc.clear()
        self.hpc_val_acc.clear()
        self._plot_hpc_training_chart()
        self._plot_hpc_combo_chart()
        self._plot_hpc_auc_chart()

        data_root = self.dataset_combo.currentText().strip()
        if not os.path.isdir(data_root):
            QMessageBox.warning(self, "Invalid Folder", f"'{data_root}' is not valid.")
            return

        chosen_models = []
        if self.model_ck_binary.isChecked():
            chosen_models.append("binary")
        if self.model_ck_simple.isChecked():
            chosen_models.append("simple")
        if self.model_ck_threeclass.isChecked():
            chosen_models.append("threeclass")
        if not chosen_models:
            QMessageBox.warning(self, "No Models", "Please check at least one model variant.")
            return

        def parse_csv_floats(text):
            res = []
            for t in text.split(","):
                t = t.strip()
                try:
                    val = float(t)
                    res.append(val)
                except:
                    pass
            return res

        def parse_csv_ints(text):
            res = []
            for t in text.split(","):
                t = t.strip()
                try:
                    val = int(t)
                    res.append(val)
                except:
                    pass
            return res

        def parse_csv_bools(text):
            res = []
            for t in text.split(","):
                t = t.strip().lower()
                if t in ("true","1"):
                    res.append(True)
                elif t in ("false","0"):
                    res.append(False)
            return res

        dropout_list = parse_csv_floats(self.dropout_edit.text())
        wdec_list = parse_csv_floats(self.wdec_edit.text())
        es_list = parse_csv_ints(self.es_edit.text())
        cw_list = parse_csv_bools(self.cw_edit.text())

        if not (dropout_list and wdec_list and es_list and cw_list):
            QMessageBox.critical(self, "Input Error", "One of the HPC lists is empty or invalid.")
            return

        batch_size = self.batch_spin.value()
        try:
            lr = float(self.lr_edit.text().strip())
        except:
            QMessageBox.critical(self, "LR Error", "LR must be float.")
            return

        mep = self.mini_epoch_spin.value()
        print(self.transferlearn_model.currentText())
        if self.transferlearn_model.currentText() != "None":
            from src.transfer_learning import transfer_learning
            tf = transfer_learning(batch_size=bs,num_epochs=mep,lr=lr,data_root=root_dir,disable_augment=disaug_list,adv_aug=advaug_list)
            tf.epoch_msg.connect(self._on_hpc_log)
            os.makedirs("hpc_checkpoints", exist_ok=True)
            if self.transferlearn_mode.currentText() == "finetuning":
                finetuned=tf.finetuning(mode=model_list[0])
                torch.save(finetuned, os.path.join("hpc_checkpoints", f"finetuned_resnet18.pth"))
            elif self.transferlearn_mode.currentText() == "feature extraction":
                feat_extracted=tf.feat_extractor(mode=model_list[0])
                torch.save(feat_extracted, os.path.join("hpc_checkpoints", f"feat_extracted_resnet18.pth"))
        else:
            from .auto_search_tab import HPCSearchWorker
            self.hpc_worker = HPCSearchWorker()
            self.hpc_worker.setup_hpc(
                data_root=root_dir,
                model_variants=model_list,
                resize_list=resize_list,
                disable_aug_list=disaug_list,
                adv_aug_list=advaug_list,
                dropout_list=do_list,
                weight_decay_list=wdec_list,
                earlystop_list=es_list,
                class_weight_list=cw_list,
                batch_size=bs,
                lr=lr,
                mini_epochs=mep,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            self.hpc_worker.progress.connect(self._on_hpc_progress)
            self.hpc_worker.log_msg.connect(self._on_hpc_log)
            self.hpc_worker.combo_epoch_update.connect(self._on_hpc_combo_epoch)
            self.hpc_worker.finished.connect(self._on_hpc_finished)

        from .auto_search_tab import HPCSearchWorker
        self.hpc_worker = HPCSearchWorker()
        self.hpc_worker.setup_hpc(
            data_root=data_root,
            model_variants=chosen_models,
            dropout_list=dropout_list,
            weight_decay_list=wdec_list,
            earlystop_list=es_list,
            class_weight_list=cw_list,
            batch_size=batch_size,
            lr=lr,
            mini_epochs=mep,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.hpc_worker.progress.connect(self._on_hpc_progress)
        self.hpc_worker.log_msg.connect(self._on_hpc_log)
        self.hpc_worker.combo_epoch_update.connect(self._on_hpc_combo_epoch)
        self.hpc_worker.finished.connect(self._on_hpc_finished)

        self.btn_hpc.setEnabled(False)
        self.hpc_worker.start()

    def _on_hpc_progress(self, val):
        self.hpc_progress.setValue(val)

    def _on_hpc_log(self, msg):
        self.hpc_log.append(msg)

    def _on_hpc_combo_epoch(self, combo_idx, epoch_idx, tr_acc, val_acc):
        if combo_idx != self.current_combo_idx:
            self.current_combo_idx = combo_idx
            self.hpc_train_acc.clear()
            self.hpc_val_acc.clear()

        self.hpc_train_acc.append(tr_acc)
        self.hpc_val_acc.append(val_acc)
        self._plot_hpc_training_chart()

    def _plot_hpc_training_chart(self):
        self.hpc_canvas.figure.clear()
        ax = self.hpc_canvas.figure.add_subplot(111)
        ax.set_title("HPC Combo Training (Partial)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True)

        e_range = range(1, len(self.hpc_train_acc)+1)
        ax.plot(e_range, self.hpc_train_acc, marker='o', label="Train")
        ax.plot(e_range, self.hpc_val_acc, marker='s', label="Val")
        ax.legend()

        self.hpc_canvas.figure.tight_layout()
        self.hpc_canvas.draw()

    def _on_hpc_finished(self):
        self.hpc_log.append("=== HPC Finished ===")
        self.hpc_progress.setValue(100)
        self.btn_hpc.setEnabled(True)

        if self.hpc_worker:
            new_results = self.hpc_worker.all_results
            best_params_str = self.hpc_worker.best_params_str
            best_acc = self.hpc_worker.best_acc

            old_list = load_hpc_results_json()
            combined = old_list + new_results
            deduped = []
            seen = set()
            for item in combined:
                key = (item.get("combo_str",""), item.get("model_path",""))
                if key not in seen:
                    seen.add(key)
                    deduped.append(item)

            self.all_results = deduped
            self._save_current_hpc_results()

            self.best_params_str = best_params_str
            self.hpc_log.append(f"Best HPC => {best_params_str} (val_acc ~ {best_acc:.1f}%)")

            self._update_hpc_table()
            self._plot_hpc_combo_chart()
            self._plot_hpc_auc_chart()

            self.hpc_updated.emit()
            if self.best_params_str:
                self.btn_final.setEnabled(True)

        self.hpc_worker = None

    def _update_hpc_table(self):
        sorted_list = sorted(self.all_results, key=lambda x: x["val_acc"], reverse=True)
        self.hpc_table.setRowCount(len(sorted_list))
        for i, rec in enumerate(sorted_list):
            rank_item = QTableWidgetItem(str(i+1))
            combo_item = QTableWidgetItem(rec["combo_str"])
            acc_item = QTableWidgetItem(f"{rec['val_acc']:.2f}")
            has_model = "Yes" if rec["model_path"] else "No"
            model_item = QTableWidgetItem(has_model)

            self.hpc_table.setItem(i, 0, rank_item)
            self.hpc_table.setItem(i, 1, combo_item)
            self.hpc_table.setItem(i, 2, acc_item)
            self.hpc_table.setItem(i, 3, model_item)

    def _plot_hpc_combo_chart(self):
        self.hpc_bar_canvas.figure.clear()
        ax = self.hpc_bar_canvas.figure.add_subplot(111)
        if not self.all_results:
            ax.text(0.5, 0.5, "No HPC results yet", ha='center', va='center')
        else:
            xvals = range(len(self.all_results))
            acc_vals = [r["val_acc"] for r in self.all_results]
            best_idx = max(range(len(acc_vals)), key=lambda i: acc_vals[i])
            colors = ["steelblue"] * len(acc_vals)
            colors[best_idx] = "red"
            ax.bar(xvals, acc_vals, color=colors)
            ax.set_title("HPC Combos Validation Accuracy")
            ax.set_xlabel("Combo Index")
            ax.set_ylabel("Val Acc (%)")
            ax.set_xticks(xvals)
            ax.set_xticklabels([str(i+1) for i in xvals])
            ax.grid(axis='y', alpha=0.3)

        self.hpc_bar_canvas.figure.tight_layout()
        self.hpc_bar_canvas.draw()

    def _plot_hpc_auc_chart(self):
        self.hpc_auc_canvas.figure.clear()
        ax = self.hpc_auc_canvas.figure.add_subplot(111)
        if not self.all_results:
            ax.text(0.5, 0.5, "No HPC results yet", ha='center', va='center')
        else:
            auc_vals = []
            for rec in self.all_results:
                roc_obj = rec.get("roc", None)
                if roc_obj and roc_obj.get("auc", None) is not None:
                    auc_vals.append(float(roc_obj["auc"]))
                else:
                    auc_vals.append(0.0)

            xvals = range(len(auc_vals))
            best_idx = max(range(len(auc_vals)), key=lambda i: auc_vals[i])
            colors = ["purple"] * len(auc_vals)
            colors[best_idx] = "orange"
            ax.bar(xvals, auc_vals, color=colors)
            ax.set_title("HPC Combos ROC AUC (Binary Only)")
            ax.set_xlabel("Combo Index")
            ax.set_ylabel("AUC")
            ax.set_xticks(xvals)
            ax.set_xticklabels([str(i+1) for i in xvals])
            ax.grid(axis='y', alpha=0.3)

        self.hpc_auc_canvas.figure.tight_layout()
        self.hpc_auc_canvas.draw()

    def on_show_hpc_info(self):
        row = self.hpc_table.currentRow()
        if row < 0:
            self.hpc_info_text.setPlainText("No HPC row selected.")
            return

        combo_str = self.hpc_table.item(row, 1).text()
        val_acc_s = self.hpc_table.item(row, 2).text()
        try:
            val_acc = float(val_acc_s)
        except:
            val_acc = 0.0

        found = None
        for rec in self.all_results:
            if rec["combo_str"] == combo_str and abs(rec["val_acc"]-val_acc) < 1e-4:
                found = rec
                break

        if not found:
            self.hpc_info_text.setPlainText("Could not find HPC record.")
            return

        cm_str = self._format_confmat(found["confmat"])
        roc_str = self._format_roc(found["roc"])
        mets_str = self._format_metrics(found["metrics"])

        html = f"""
        <div style="border:2px solid #444;padding:10px;border-radius:6px;">
          <h3 style="margin-top:0;">HPC Info Card</h3>
          <p><b>Hyperparams:</b> {combo_str}</p>
          <p><b>Val Accuracy:</b> {val_acc:.2f}%</p>
          <p><b>Confusion Matrix:</b><br>{cm_str}</p>
          <p><b>ROC Info:</b><br>{roc_str}</p>
          <p><b>Class Metrics:</b><br>{mets_str}</p>
          <hr><em>Model file: {found['model_path'] if found['model_path'] else 'No checkpoint'}</em>
        </div>
        """
        self.hpc_info_text.setHtml(html)

    def _format_confmat(self, confmat):
        if confmat is None:
            return "N/A"
        arr = confmat.cpu().numpy()
        rows = []
        for row in arr:
            row_s = " ".join(f"{int(x):3d}" for x in row)
            rows.append(row_s)
        return f"<pre>{chr(10).join(rows)}</pre>"

    def _format_roc(self, roc_dict):
        if not roc_dict or roc_dict.get("fpr") is None:
            return "N/A"
        auc_v = roc_dict.get("auc", None)
        if auc_v is None:
            return "AUC = N/A"
        return f"AUC = {auc_v:.3f}"

    def _format_metrics(self, metrics):
        if not metrics:
            return "N/A"
        lines = []
        p = metrics.get("precision", [])
        r = metrics.get("recall", [])
        f1 = metrics.get("f1", [])
        sup = metrics.get("support", [])
        for i in range(len(p)):
            lines.append(f"Class {i}: p={p[i]:.3f}, r={r[i]:.3f}, f1={f1[i]:.3f}, sup={sup[i]}")
        micro = metrics.get("micro_f1",None)
        macro = metrics.get("macro_f1",None)
        if micro is not None:
            lines.append(f"Micro-F1: {micro:.3f}")
        if macro is not None:
            lines.append(f"Macro-F1: {macro:.3f}")
        return f"<pre>{chr(10).join(lines)}</pre>"

    def on_delete_model(self):
        row = self.hpc_table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "No row selected", "Select an HPC row first.")
            return

        combo_str = self.hpc_table.item(row, 1).text()
        val_acc_s = self.hpc_table.item(row, 2).text()
        try:
            val_acc = float(val_acc_s)
        except:
            val_acc = 0.0

        found_index = None
        for i, rec in enumerate(self.all_results):
            if rec["combo_str"] == combo_str and abs(rec["val_acc"] - val_acc) < 1e-5:
                found_index = i
                break

        if found_index is None:
            QMessageBox.information(self, "Not Found", "No matching HPC combo in memory.")
            return

        found = self.all_results[found_index]
        mpath = found["model_path"]
        if mpath and os.path.isfile(mpath):
            try:
                os.remove(mpath)
                self.hpc_log.append(f"Deleted HPC model file: {mpath}")
            except OSError as e:
                QMessageBox.critical(self, "Error", f"Could not delete: {e}")

        self.all_results.pop(found_index)
        self.hpc_log.append("Removed HPC combo from list.")

        self._save_current_hpc_results()
        self._update_hpc_table()
        self._plot_hpc_combo_chart()
        self._plot_hpc_auc_chart()

    def on_train_final(self):
        self.final_log.clear()
        self.final_progress.setValue(0)
        self.final_epoch_list.clear()
        self.final_train_acc_list.clear()
        self.final_val_acc_list.clear()
        self._plot_final_chart()

        self.final_roc_data = None
        self.final_confmat_data = None
        self.final_metrics_data = None
        self.metrics_text.clear()
        self._plot_final_confmat()
        self._plot_final_roc()

        if not self.best_params_str:
            QMessageBox.information(self, "No HPC Combo", "No best HPC combo to train.")
            return

        root_folder = self.dataset_combo.currentText().strip()
        if not os.path.isdir(root_folder):
            QMessageBox.warning(self, "Invalid Folder", f"{root_folder} not valid.")
            return

        ep = self.final_epochs_spin.value()
        bs = self.final_batch_spin.value()
        try:
            lr = float(self.final_lr_edit.text().strip())
        except:
            QMessageBox.critical(self, "LR Error", "Final LR must be float.")
            return
        folds = self.final_folds_spin.value()

        from .auto_search_tab import FinalTrainWorker
        self.final_worker = FinalTrainWorker()
        self.final_worker.setup_final(
            best_params_str=self.best_params_str,
            data_root=root_folder,
            epochs=ep,
            batch=bs,
            lr=lr,
            folds=folds,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.final_worker.log_msg.connect(self._on_final_log)
        self.final_worker.progress.connect(self._on_final_progress)
        self.final_worker.epoch_update.connect(self._on_final_epoch)
        self.final_worker.finishedDetailed.connect(self._on_final_finished)

        self.btn_final.setEnabled(False)
        self.final_worker.start()

    def _on_final_log(self, msg):
        self.final_log.append(msg)

    def _on_final_progress(self, val):
        self.final_progress.setValue(val)

    def _on_final_epoch(self, ep, train_acc, val_acc):
        self.final_epoch_list.append(ep)
        self.final_train_acc_list.append(train_acc)
        self.final_val_acc_list.append(val_acc)
        self._plot_final_chart()

    def _on_final_finished(self, final_acc, roc_dict, confmat, metrics):
        self.final_log.append(f"Final training done => acc={final_acc:.2f}%")
        self.final_progress.setValue(100)
        self.btn_final.setEnabled(True)

        self.final_roc_data = roc_dict
        self.final_confmat_data = confmat
        self.final_metrics_data = metrics

        self._plot_final_confmat()
        self._plot_final_roc()
        self._show_final_metrics()

        self.final_worker = None

    def _plot_final_chart(self):
        self.final_canvas.figure.clear()
        ax = self.final_canvas.figure.add_subplot(111)
        ax.set_title("Final Train vs. Val Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True)
        ax.plot(self.final_epoch_list, self.final_train_acc_list, marker='o', label="Train")
        ax.plot(self.final_epoch_list, self.final_val_acc_list, marker='s', label="Val")
        ax.legend()
        self.final_canvas.figure.tight_layout()
        self.final_canvas.draw()

    def _plot_final_confmat(self):
        self.confmat_canvas.figure.clear()
        ax = self.confmat_canvas.figure.add_subplot(111)
        if self.final_confmat_data is None:
            ax.text(0.5, 0.5, "No ConfMat found", ha='center', va='center')
        else:
            cm = self.final_confmat_data.cpu().numpy().astype(float)
            im = ax.imshow(cm, cmap="Blues")
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(int(cm[i,j])), ha='center', va='center')
            self.confmat_canvas.figure.colorbar(im, ax=ax)
        self.confmat_canvas.figure.tight_layout()
        self.confmat_canvas.draw()

    def _plot_final_roc(self):
        self.roc_canvas.figure.clear()
        ax = self.roc_canvas.figure.add_subplot(111)
        if not self.final_roc_data or self.final_roc_data.get("fpr") is None:
            ax.text(0.5, 0.5, "ROC not available", ha='center', va='center')
        else:
            fpr = self.final_roc_data["fpr"]
            tpr = self.final_roc_data["tpr"]
            auc_v = self.final_roc_data["auc"] if self.final_roc_data["auc"] else 0.0
            ax.plot(fpr, tpr, color='red', label=f"AUC={auc_v:.2f}")
            ax.plot([0,1],[0,1],"--", color='gray')
            ax.set_title("ROC Curve")
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.legend()
        self.roc_canvas.figure.tight_layout()
        self.roc_canvas.draw()

    def _show_final_metrics(self):
        self.metrics_text.clear()
        if not self.final_metrics_data:
            self.metrics_text.append("No class metrics found.")
            return
        prec = self.final_metrics_data.get("precision",[])
        rec = self.final_metrics_data.get("recall",[])
        f1s = self.final_metrics_data.get("f1",[])
        sup = self.final_metrics_data.get("support",[])
        self.metrics_text.append("<b>Class Metrics</b> (Precision/Recall/F1/Support)")
        for i in range(len(prec)):
            self.metrics_text.append(
                f"Class {i}: p={prec[i]:.3f}, r={rec[i]:.3f}, f1={f1s[i]:.3f}, sup={sup[i]}"
            )
        micro = self.final_metrics_data.get("micro_f1",None)
        macro = self.final_metrics_data.get("macro_f1",None)
        if micro is not None:
            self.metrics_text.append(f"\nMicro-F1: {micro:.3f}")
        if macro is not None:
            self.metrics_text.append(f"Macro-F1: {macro:.3f}")
