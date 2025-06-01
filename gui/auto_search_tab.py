import os
import json
import itertools
import copy
import torch
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QGroupBox, QFormLayout,
    QLabel, QPushButton, QComboBox, QLineEdit, QSpinBox,
    QProgressBar, QTextEdit, QMessageBox, QHBoxLayout, QTableWidget,
    QTableWidgetItem, QHeaderView, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from gui.widgets import MplCanvas

from src.experiment import Experiment

_HPC_RESULTS_JSON = "hpc_results.json"

# ------------------------------------------------------------------------------------
# Helper functions for (de)serialising HPC results
# ------------------------------------------------------------------------------------
def _load_existing_hpc_results(self):
    """
    Load any persisted HPC results at start-up, repopulate the table/plots,
    and if at least one combo exists, enable the “Train Final Model” button.
    """
    self.all_results = load_hpc_results_json()
    if not self.all_results:
        return

    # Refresh the HPC table and charts
    self._update_hpc_table()
    self._plot_hpc_combo_chart()
    self._plot_hpc_auc_chart()

    # Select the best combo by accuracy so we can let the user click "Train Final Model"
    best_rec = max(self.all_results, key=lambda r: r["val_acc"])
    self.best_params_str = best_rec["combo_str"]
    self.btn_final.setEnabled(True)

def _save_current_hpc_results(self):
    """
    Handy wrapper for re-saving all HPC combos after deleting or modifying entries.
    """
    save_hpc_results_json(self.all_results)

def save_hpc_results_json(all_results, path=_HPC_RESULTS_JSON):
    """
    Persist the list of HPC result dictionaries so they survive application restarts.
    Tensor and ndarray fields are converted to vanilla Python lists first.
    """
    serialisable = []
    for rec in all_results:
        cmat = rec.get("confmat")
        if isinstance(cmat, torch.Tensor):
            cmat = cmat.cpu().tolist()

        roc = rec.get("roc")
        if roc:
            roc = {
                "fpr": (roc.get("fpr") or []),
                "tpr": (roc.get("tpr") or []),
                "auc": roc.get("auc"),
            }
            # Convert np.ndarrays
            for k in ("fpr", "tpr"):
                if isinstance(roc[k], np.ndarray):
                    roc[k] = roc[k].tolist()

        serialisable.append({
            "combo_str":     rec.get("combo_str", ""),
            "val_acc":       rec.get("val_acc", 0.0),
            "model_path":    rec.get("model_path", ""),
            "model_variant": rec.get("model_variant", ""),
            "confmat_list":  cmat,
            "roc":           roc,
            "metrics":       rec.get("metrics", None)
        })

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2)
    except Exception as e:
        print(f"[WARN] Could not save HPC results to {path}: {e}")


def load_hpc_results_json(path=_HPC_RESULTS_JSON):
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Could not load {path}: {e}")
        return []

    restored = []
    for item in data:
        cmat = item.get("confmat_list")
        if cmat is not None:
            cmat = torch.tensor(cmat, dtype=torch.int32)

        restored.append({
            "combo_str":     item.get("combo_str"),
            "val_acc":       item.get("val_acc", 0.0),
            "model_path":    item.get("model_path"),
            "model_variant": item.get("model_variant"),
            "confmat":       cmat,
            "roc":           item.get("roc"),
            "metrics":       item.get("metrics")
        })
    return restored

# ------------------------------------------------------------------------------------
# Worker threads (unchanged)
# ------------------------------------------------------------------------------------
class HPCSearchWorker(QThread):
    """Grid-search across model variant × dropout × weight-decay × early-stop × class-weight."""
    progress = pyqtSignal(int)
    log_msg = pyqtSignal(str)
    combo_epoch_update = pyqtSignal(int, int, float, float)
    finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # default values
        self.setup_hpc(
            data_root="cleaned_data", model_variants=[],
            dropout_list=[], weight_decay_list=[],
            earlystop_list=[], class_weight_list=[],
            batch_size=16, lr=1e-3, mini_epochs=2,
            device="cpu"
        )

        self.best_acc = -1.0
        self.best_params_str = ""
        self.all_results = []

    # ------------------------------------------------------------------
    def setup_hpc(
        self, data_root, model_variants, dropout_list, weight_decay_list,
        earlystop_list, class_weight_list, batch_size, lr, mini_epochs,
        device="cpu"
    ):
        self.data_root = data_root
        self.model_variants = model_variants
        self.dropout_list = dropout_list
        self.weight_decay_list = weight_decay_list
        self.earlystop_list = earlystop_list
        self.class_weight_list = class_weight_list
        self.batch_size = batch_size
        self.lr = lr
        self.epochs  = mini_epochs
        self.folds   = 1 
        self.device = device
        self.batch        = batch_size      
        self.batch_size   = batch_size
        self.mini_epochs  = mini_epochs 
        self.epochs       = mini_epochs
        self.best_acc = -1.0
        self.best_params_str = ""
        self.all_results = []

    # ------------------------------------------------------------------------------------
    # HPCSearchWorker.run
    # ------------------------------------------------------------------------------------
    def run(self):
        """
        Main grid-search loop.

        Uses the new Experiment signature (no disable_augment / adv_aug).
        """
        param_space = list(itertools.product(
            self.model_variants,
            self.dropout_list,
            self.weight_decay_list,
            self.earlystop_list,
            self.class_weight_list,
        ))
        total = len(param_space)
        if total == 0:
            self.log_msg.emit("No HPC combinations to evaluate.")
            self.finished.emit()
            return

        self.log_msg.emit(
            f"Starting HPC search across {total} combos "
            f"(each {self.mini_epochs} epoch{'s' if self.mini_epochs > 1 else ''})"
        )
        os.makedirs("hpc_checkpoints", exist_ok=True)

        for combo_idx, (model_var, drop, wdec, estop, cw) in enumerate(param_space):
            combo_str = (
                f"model={model_var}, drop={drop}, "
                f"wdec={wdec}, eStop={estop}, cw={cw}"
            )
            self.log_msg.emit(f"\n[{combo_idx + 1}/{total}] {combo_str}")

            try:
                exp = Experiment(
                    data_root=self.data_root,
                    model_variant=model_var,
                    input_size=128,
                    batch_size=self.batch,
                    lr=self.lr,
                    epochs=self.epochs,
                    dropout=drop,
                    weight_decay=wdec,
                    earlystop_patience=estop,
                    use_class_weight=cw,
                    device=self.device,
                    folds=self.folds,
                    )
                exp.set_logger(self.log_msg.emit)
                exp.load_data()
                if getattr(exp.train_dataset, "is_empty", False):
                    raise RuntimeError("train dataset empty")
                exp._setup_optimizer_criterion()

            except Exception as e:
                self.log_msg.emit(f"  !! Build failed: {e}")
                self._cache_result(combo_str, model_var, 0.0, None, None, None, None)
                self.progress.emit(int(100 * (combo_idx + 1) / total))
                continue

            best_val, best_state, best_confmat, best_roc, best_metrics = \
                self._train_partial(exp, combo_idx, estop)

            model_path = None
            if best_state is not None:
                model_path = os.path.join("hpc_checkpoints",
                                        f"hpc_combo_{combo_idx + 1}.pth")
                torch.save(best_state, model_path)

            self._cache_result(combo_str, model_var,
                            best_val, best_confmat, best_roc,
                            best_metrics, model_path)

            if best_val > self.best_acc:
                self.best_acc = best_val
                self.best_params_str = combo_str

            self.progress.emit(int(100 * (combo_idx + 1) / total))

        self.log_msg.emit(
            f"\nHPC finished. Best val_acc={self.best_acc:.2f}%: {self.best_params_str}"
        )
        self.finished.emit()



    # ------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
    
    # ------------------------------------------------------------------------------------
    def _save_current_hpc_results(self):
        """
        Convenience wrapper used by on_delete_model().
        """
        save_hpc_results_json(self.all_results)

    def _train_partial(self, exp, combo_idx, estop):
        best_val = 0.0
        best_state = best_confmat = best_roc = best_metrics = None
        patience = 0
        for ep in range(self.mini_epochs):
            tr_loss, tr_acc, _ = exp._train_one_epoch()
            val_metrics = exp._evaluate()
            val_acc = val_metrics["accuracy"]

            self.combo_epoch_update.emit(combo_idx, ep+1, tr_acc, val_acc)
            self.log_msg.emit(f"  Ep {ep+1}: train={tr_acc:.1f}%, val={val_acc:.1f}%")

            if val_acc > best_val:
                best_val = val_acc
                patience = 0
                best_state = copy.deepcopy(exp.model.state_dict())
                best_confmat = val_metrics.get("confusion_matrix")
                best_roc = {
                    "fpr": val_metrics.get("roc_fpr"),
                    "tpr": val_metrics.get("roc_tpr"),
                    "auc": val_metrics.get("roc_auc")
                }
                best_metrics = exp._compute_class_metrics(val_metrics)
            else:
                patience += 1
                if estop and patience >= estop:
                    self.log_msg.emit(f"  -> Early-stopping at epoch {ep+1}")
                    break
        return best_val, best_state, best_confmat, best_roc, best_metrics

    # ------------------------------------------------------------------
    def _cache_result(self, combo_str, model_var, val_acc,
                      confmat, roc, metrics, model_path):
        self.all_results.append({
            "combo_str": combo_str,
            "val_acc": val_acc,
            "model_path": model_path,
            "model_variant": model_var,
            "confmat": confmat,
            "roc": roc,
            "metrics": metrics
        })


class FinalTrainWorker(QThread):
    """Full-length training on the best combo (unchanged)."""
    finishedDetailed = pyqtSignal(float, dict, object, dict)
    progress = pyqtSignal(int)
    log_msg = pyqtSignal(str)
    epoch_update = pyqtSignal(int, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_final("", "cleaned_data", 10, 32, 1e-3, 1, "cpu")

    # ------------------------------------------------------------------
    def setup_final(self, best_params_str, data_root, epochs,
                    batch, lr, folds, device="cpu"):
        self.best_params_str = best_params_str
        self.data_root = data_root
        self.epochs = epochs
        self.batch = batch
        self.lr = lr
        self.folds = folds
        self.device = device

    # ------------------------------------------------------------------
    def run(self):
        if not self.best_params_str:
            self.log_msg.emit("Final training aborted: no HPC combo.")
            self.finishedDetailed.emit(0.0, {}, None, {})
            return

        # Parse hyper-parameter string
        pmap = {k.strip(): v.strip() for k, v in
                (piece.split("=") for piece in self.best_params_str.split(",") if "=" in piece)}

        model_var = pmap.get("model", "binary")
        drop  = float(pmap.get("drop", 0.0))
        wdec  = float(pmap.get("wdec", 0.0))
        estop = int(pmap.get("eStop", 0))
        cw    = (pmap.get("cw", "False") == "True")

        self.log_msg.emit(f"[Final] {self.best_params_str}")
        self.log_msg.emit(f"folds={self.folds}, epochs={self.epochs}, "
                          f"batch={self.batch}, lr={self.lr}")

        try:
            exp = Experiment(
                data_root=self.data_root,
                model_variant=model_var,
                input_size=128,
                batch_size=self.batch,
                lr=self.lr,
                epochs=self.epochs,
                dropout=drop,
                weight_decay=wdec,
                earlystop_patience=estop,
                use_class_weight=cw,
                device=self.device,
                folds=self.folds,
                )
            exp.set_logger(self.log_msg.emit)
            exp.load_data()
            if getattr(exp.train_dataset, "is_empty", False):
                raise RuntimeError("train dataset empty")
            exp._setup_optimizer_criterion()

        except Exception as e:
            self.log_msg.emit(f"Build failed: {e}")
            self.finishedDetailed.emit(0.0, {}, None, {})
            return

        # Single split or k-fold
        if self.folds <= 1:
            self._run_single_split(exp, estop)
        else:
            self._run_kfold(exp)

    # ------------------------------------------------------------------
    def _run_single_split(self, exp, estop):
        best_val = 0.0
        patience = 0
        for ep in range(self.epochs):
            tr_loss, tr_acc, _ = exp._train_one_epoch()
            val_metrics = exp._evaluate()
            val_acc = val_metrics["accuracy"]

            self.epoch_update.emit(ep+1, tr_acc, val_acc)
            self.progress.emit(int(100*(ep+1)/self.epochs))
            self.log_msg.emit(f"Ep {ep+1}: train={tr_acc:.1f}%, val={val_acc:.1f}%")

            if val_acc > best_val:
                best_val = val_acc
                patience = 0
            else:
                patience += 1
                if estop and patience >= estop:
                    self.log_msg.emit(f"Early-stop at ep {ep+1}")
                    break

        self.finishedDetailed.emit(
            best_val, exp.final_roc, exp.final_confmat, exp.final_class_metrics
        )

    # ------------------------------------------------------------------
    def _run_kfold(self, exp):
        avg_acc, _, _, _ = exp.run_kfold(self.folds, self.epochs)
        self.log_msg.emit(f"K-fold avg val_acc={avg_acc:.2f}%")
        self.progress.emit(100)
        self.finishedDetailed.emit(
            avg_acc, exp.final_roc, exp.final_confmat, exp.final_class_metrics
        )

# ------------------------------------------------------------------------------------
# Main GUI class
# ------------------------------------------------------------------------------------
class AutoSearchTab(QWidget):
    """
    End-to-end tab:
      * Hyper-parameter grid-search (binary / simple / threeclass) **or** transfer-learning.
      * Final full-run training of the best combo.
    """

    hpc_updated = pyqtSignal()   # emitted after every grid-search completion

    # ------------------------------------------------------------------
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("AutoSearchTab")
        self._build_ui()
        self._wire_events()

        # Runtime state
        self.hpc_worker = None
        self.final_worker = None
        self.current_combo_idx = -1
        self.hpc_train_acc = []
        self.hpc_val_acc = []
        self.best_params_str = ""
        self.all_results = []

        self.final_epoch_list = []
        self.final_train_acc_list = []
        self.final_val_acc_list = []
        self.final_roc_data = None
        self.final_confmat_data = None
        self.final_metrics_data = None

        # Load persisted HPC results (if any)
        self._load_existing_hpc_results()

        # Disable TL widgets initially
        self._on_transfer_checkbox_changed(Qt.Unchecked)

    # ------------------------------------------------------------------
    # █████████████████████████  UI BUILDING  █████████████████████████
    # ------------------------------------------------------------------
    def _load_existing_hpc_results(self):
        """
        Load any persisted HPC results at start-up, repopulate the table/plots,
        and, if at least one combo exists, enable the “Train Final Model” button.
        """
        self.all_results = load_hpc_results_json()
        if not self.all_results:
            return

        # Refresh UI
        self._update_hpc_table()
        self._plot_hpc_combo_chart()
        self._plot_hpc_auc_chart()

        # Pre-select best combo
        best_rec = max(self.all_results, key=lambda r: r["val_acc"])
        self.best_params_str = best_rec["combo_str"]
        self.btn_final.setEnabled(True)

    def _plot_hpc_training_chart(self):
        self.hpc_canvas.figure.clear()
        ax = self.hpc_canvas.figure.add_subplot(111)
        if not self.hpc_train_acc:
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center")
        else:
            ax.plot(self.hpc_train_acc, label="Train", marker="o")
            ax.plot(self.hpc_val_acc,   label="Val",   marker="s")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title("Current Combo")
            ax.grid(True)
            ax.legend()
        self.hpc_canvas.figure.tight_layout()
        self.hpc_canvas.draw()

    def _build_ui(self):
        # ---------- Main containers ----------
        main_layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        cont_layout = QVBoxLayout(container)
        cont_layout.setContentsMargins(15, 15, 15, 15)
        cont_layout.setSpacing(20)

        # ---------- 1. HPC settings ----------
        hpc_group = QGroupBox("Hyper-parameter Search (HPC)")
        hpc_form  = QFormLayout(hpc_group)

        # Dataset folder
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["cleaned_data", "unprocessed_data", "augmented_data"])
        hpc_form.addRow("Dataset Folder:", self.dataset_combo)

        # Model-variant check-boxes  ← lpconv added
        self.model_ck_binary      = QCheckBox("binary")
        self.model_ck_simple      = QCheckBox("simple")
        self.model_ck_threeclass  = QCheckBox("threeclass")
        self.model_ck_lpconv      = QCheckBox("lpconv")       # NEW
        self.model_ck_transfer    = QCheckBox("transfer")

        model_ck_layout = QHBoxLayout()
        for ck in (
            self.model_ck_binary,
            self.model_ck_simple,
            self.model_ck_threeclass,
            self.model_ck_lpconv,       # NEW
            self.model_ck_transfer,
        ):
            model_ck_layout.addWidget(ck)
        hpc_form.addRow(QLabel("Model Variants:"), model_ck_layout)

        # … (the remainder of _build_ui is **unchanged**)


        # Classic HPC hyper-params
        self.dropout_edit = QLineEdit("0.0,0.3")
        self.wdec_edit    = QLineEdit("0.0,0.0001")
        self.es_edit      = QLineEdit("0,3")
        self.cw_edit      = QLineEdit("False,True")

        hpc_form.addRow("Dropout List:",      self.dropout_edit)
        hpc_form.addRow("Weight Decay List:", self.wdec_edit)
        hpc_form.addRow("Early-stop List:",   self.es_edit)
        hpc_form.addRow("Class-weight List:", self.cw_edit)

        self.batch_spin    = QSpinBox(); self.batch_spin.setRange(1,512); self.batch_spin.setValue(16)
        self.lr_edit       = QLineEdit("0.001")
        self.mini_epoch_spin = QSpinBox(); self.mini_epoch_spin.setRange(1,20); self.mini_epoch_spin.setValue(2)

        hpc_form.addRow("Batch Size:",         self.batch_spin)
        hpc_form.addRow("Learning Rate:",      self.lr_edit)
        hpc_form.addRow("Mini-epochs (HPC):",  self.mini_epoch_spin)

        # Transfer-learning specific widgets (enabled when ck_transfer ticked)
        self.transferlearn_model = QComboBox()
        self.transferlearn_model.addItems(["None", "ResNet18", "EfficientNetB0"])
        self.transferlearn_mode  = QComboBox()
        self.transferlearn_mode.addItems(["finetuning", "feature extraction"])
        hpc_form.addRow("Base Model (TL):", self.transferlearn_model)
        hpc_form.addRow("TL Mode:",         self.transferlearn_mode)

        # Launch button
        self.btn_hpc = QPushButton("Run Search")
        hpc_form.addRow(self.btn_hpc)

        cont_layout.addWidget(hpc_group)

        # ---------- 2. HPC live progress ----------
        self.hpc_progress = QProgressBar(); self.hpc_progress.setValue(0)
        self.hpc_log = QTextEdit(); self.hpc_log.setReadOnly(True); self.hpc_log.setMinimumHeight(180)

        cont_layout.addWidget(self.hpc_progress)
        cont_layout.addWidget(self.hpc_log)

        # Live combo accuracy plot
        self.hpc_canvas  = MplCanvas(self, width=8, height=4, dpi=100)
        self.hpc_toolbar = NavigationToolbar(self.hpc_canvas, self)
        cont_layout.addWidget(QLabel("<b>Current Combo: Train vs Val Acc</b>"))
        cont_layout.addWidget(self.hpc_canvas)
        cont_layout.addWidget(self.hpc_toolbar)

        # ---------- 3. HPC results table & charts ----------
        results_group = QGroupBox("HPC Results Overview")
        rg_layout = QVBoxLayout(results_group)

        self.hpc_table = QTableWidget()
        self.hpc_table.setColumnCount(4)
        self.hpc_table.setHorizontalHeaderLabels(["Rank", "Combo", "Val Acc", "Model?"])
        self.hpc_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.hpc_table.setMinimumHeight(600)
        rg_layout.addWidget(self.hpc_table)

        h_btns = QHBoxLayout()
        self.btn_show_info   = QPushButton("Show Info")
        self.btn_delete_model= QPushButton("Delete Model")
        h_btns.addWidget(self.btn_show_info); h_btns.addWidget(self.btn_delete_model)
        rg_layout.addLayout(h_btns)

        self.hpc_info_text = QTextEdit(); self.hpc_info_text.setReadOnly(True); self.hpc_info_text.setMinimumHeight(240)
        rg_layout.addWidget(self.hpc_info_text)

        self.hpc_bar_canvas = MplCanvas(self, width=8, height=3, dpi=100)
        self.hpc_bar_toolbar= NavigationToolbar(self.hpc_bar_canvas, self)
        rg_layout.addWidget(QLabel("<b>Validation Accuracy</b>"))
        rg_layout.addWidget(self.hpc_bar_canvas); rg_layout.addWidget(self.hpc_bar_toolbar)

        self.hpc_auc_canvas = MplCanvas(self, width=8, height=3, dpi=100)
        self.hpc_auc_toolbar= NavigationToolbar(self.hpc_auc_canvas, self)
        rg_layout.addWidget(QLabel("<b>ROC AUC (binary)</b>"))
        rg_layout.addWidget(self.hpc_auc_canvas); rg_layout.addWidget(self.hpc_auc_toolbar)

        cont_layout.addWidget(results_group)

        # ---------- 4. Final training ----------
        final_box = QGroupBox("Final Training (best HPC combo)")
        final_form = QFormLayout(final_box)

        self.final_epochs_spin = QSpinBox(); self.final_epochs_spin.setRange(1,200); self.final_epochs_spin.setValue(10)
        self.final_batch_spin  = QSpinBox(); self.final_batch_spin.setRange(1,512); self.final_batch_spin.setValue(32)
        self.final_lr_edit     = QLineEdit("0.001")
        self.final_folds_spin  = QSpinBox(); self.final_folds_spin.setRange(1,10); self.final_folds_spin.setValue(1)

        final_form.addRow("Epochs:",   self.final_epochs_spin)
        final_form.addRow("Batch:",    self.final_batch_spin)
        final_form.addRow("LR:",       self.final_lr_edit)
        final_form.addRow("Folds:",    self.final_folds_spin)

        self.btn_final = QPushButton("Train Final Model"); self.btn_final.setEnabled(False)
        final_form.addRow(self.btn_final)
        cont_layout.addWidget(final_box)

        self.final_progress = QProgressBar(); self.final_progress.setValue(0)
        self.final_log = QTextEdit(); self.final_log.setReadOnly(True); self.final_log.setMinimumHeight(100)
        cont_layout.addWidget(self.final_progress); cont_layout.addWidget(self.final_log)

        self.final_canvas  = MplCanvas(container, width=8, height=4)
        self.final_toolbar = NavigationToolbar(self.final_canvas, self)
        cont_layout.addWidget(QLabel("<b>Final Train vs Val Acc</b>"))
        cont_layout.addWidget(self.final_canvas); cont_layout.addWidget(self.final_toolbar)

        # ConfMat + ROC + metrics
        final_res_g = QGroupBox("Final Results")
        fres_layout = QHBoxLayout(final_res_g)

        self.confmat_canvas  = MplCanvas(self, width=4, height=3)
        self.confmat_toolbar = NavigationToolbar(self.confmat_canvas, self)
        left = QVBoxLayout()
        left.addWidget(QLabel("<b>Confusion Matrix</b>"))
        left.addWidget(self.confmat_canvas); left.addWidget(self.confmat_toolbar)
        fres_layout.addLayout(left)

        self.roc_canvas  = MplCanvas(self, width=4, height=3)
        self.roc_toolbar = NavigationToolbar(self.roc_canvas, self)
        right = QVBoxLayout()
        right.addWidget(QLabel("<b>ROC Curve</b>"))
        right.addWidget(self.roc_canvas); right.addWidget(self.roc_toolbar)
        fres_layout.addLayout(right)

        self.metrics_text = QTextEdit(); self.metrics_text.setReadOnly(True)
        fres_layout.addWidget(self.metrics_text)

        cont_layout.addWidget(final_res_g)

        # ---------- Finalise ----------
        scroll.setWidget(container)
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

    # ------------------------------------------------------------------
    # █████████████████████████  EVENT WIRING  █████████████████████████
    # ------------------------------------------------------------------
    def _wire_events(self):
        self.btn_hpc.clicked.connect(self.on_run_hpc)
        self.btn_show_info.clicked.connect(self.on_show_hpc_info)
        self.btn_delete_model.clicked.connect(self.on_delete_model)
        self.btn_final.clicked.connect(self.on_train_final)
        self.model_ck_transfer.stateChanged.connect(self._on_transfer_checkbox_changed)

    # ------------------------------------------------------------------
    # █████████████████████████  TRANSFER-LEARNING UI LOGIC  ███████████
    # ------------------------------------------------------------------
    def _on_transfer_checkbox_changed(self, state):
        """
        Enable/disable the TL controls and (optionally) grey-out the
        classic variant check-boxes – now also includes lpconv.
        """
        enabled = state == Qt.Checked
        self.transferlearn_model.setEnabled(enabled)
        self.transferlearn_mode.setEnabled(enabled)

        # Grey-out all non-TL variants
        for ck in (
            self.model_ck_binary,
            self.model_ck_simple,
            self.model_ck_threeclass,
            self.model_ck_lpconv,       # NEW
        ):
            ck.setEnabled(not enabled)


    # ------------------------------------------------------------------
    # █████████████████████████  RUN BUTTON  ███████████████████████████
    # ------------------------------------------------------------------
    def on_run_hpc(self):
        # -------- UI reset (identical) --------
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

        # -------- Basic checks (identical) ----
        data_root = self.dataset_combo.currentText().strip()
        if not os.path.isdir(data_root):
            QMessageBox.warning(self, "Dataset error", f"Folder '{data_root}' not found.")
            return
        batch_size = self.batch_spin.value()
        try:
            lr = float(self.lr_edit.text())
        except ValueError:
            QMessageBox.warning(self, "LR error", "Learning-rate must be a float.")
            return
        mini_epochs = self.mini_epoch_spin.value()

        transfer_selected = self.model_ck_transfer.isChecked()

        # ------------------------------------------------------------------
        # TL path (unchanged)
        # ------------------------------------------------------------------
        if transfer_selected:
            if self.transferlearn_model.currentText() == "None":
                QMessageBox.warning(self, "Transfer-learning",
                                    "Please choose a base model (not 'None').")
                return
            self._run_transfer_learning(
                data_root=data_root,
                batch_size=batch_size,
                lr=lr,
                num_epochs=mini_epochs
            )
            return

        # ------------------------------------------------------------------
        # Standard HPC path – now includes lpconv
        # ------------------------------------------------------------------
        chosen_models = []
        if self.model_ck_binary.isChecked():
            chosen_models.append("binary")
        if self.model_ck_simple.isChecked():
            chosen_models.append("simple")
        if self.model_ck_threeclass.isChecked():
            chosen_models.append("threeclass")
        if self.model_ck_lpconv.isChecked():            # NEW
            chosen_models.append("lpconv")

        if not chosen_models:
            QMessageBox.warning(
                self,
                "No model variant",
                "Tick at least one model variant or select transfer-learning.",
            )
            return

        # … (parsing of CSV hyper-params & launching HPC worker is unchanged)

        # Parse CSV helpers ------------
        def csv_floats(text):
            vals = []
            for s in text.split(","):
                try: vals.append(float(s.strip()))
                except: pass
            return vals
        def csv_ints(text):
            vals = []
            for s in text.split(","):
                try: vals.append(int(s.strip()))
                except: pass
            return vals
        def csv_bools(text):
            vals = []
            for s in text.split(","):
                s = s.strip().lower()
                if s in ("true", "1"):   vals.append(True)
                elif s in ("false", "0"): vals.append(False)
            return vals

        dropout_list = csv_floats(self.dropout_edit.text())
        wdec_list    = csv_floats(self.wdec_edit.text())
        es_list      = csv_ints(self.es_edit.text())
        cw_list      = csv_bools(self.cw_edit.text())

        if not (dropout_list and wdec_list and es_list and cw_list):
            QMessageBox.critical(self, "Input error",
                                 "HPC parameter lists must not be empty.")
            return

        # Launch worker ---------------
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
            mini_epochs=mini_epochs,
            device=("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.hpc_worker.progress.connect(self._on_hpc_progress)
        self.hpc_worker.log_msg.connect(self._on_hpc_log)
        self.hpc_worker.combo_epoch_update.connect(self._on_hpc_combo_epoch)
        self.hpc_worker.finished.connect(self._on_hpc_finished)

        self.btn_hpc.setEnabled(False)
        self.hpc_worker.start()

    # ------------------------------------------------------------------
    def _run_transfer_learning(self, data_root, batch_size, lr, num_epochs):
        """
        Executes transfer-learning outside the grid search.
        The helper `transfer_learning` no longer needs `disable_augment`
        or `adv_aug` flags.
        """
        base_model = self.transferlearn_model.currentText()
        mode       = self.transferlearn_mode.currentText()

        self.hpc_log.append(f"*** Transfer-learning: {base_model} ({mode}) ***")
        self.hpc_progress.setValue(0)
        self.btn_hpc.setEnabled(False)

        try:
            from src.transfer_learning import transfer_learning
            tl_runner = transfer_learning(
                data_root=data_root,
                batch_size=batch_size,
                num_epochs=num_epochs,
                lr=lr,
                model_name=base_model,
            )
            tl_runner.epoch_msg.connect(self._on_hpc_log)
            os.makedirs("hpc_checkpoints", exist_ok=True)

            if mode == "finetuning":
                state_dict, val_acc = tl_runner.finetuning()
                tag = "finetune"
            else:
                state_dict, val_acc = tl_runner.feat_extractor()
                tag = "feat"
        except Exception as e:
            QMessageBox.critical(self, "Transfer-learning failed", str(e))
            self.btn_hpc.setEnabled(True)
            return

        # Save checkpoint
        model_path = os.path.join("hpc_checkpoints", f"TL_{base_model}_{tag}.pth")
        torch.save(state_dict, model_path)
        self.hpc_log.append(f"Saved TL model to {model_path}")

        # Record pseudo-combo for the overview table
        combo_str = f"TL-{base_model} ({tag})"
        self.all_results.append({
            "combo_str": combo_str,
            "val_acc":  val_acc,
            "model_path": model_path,
            "model_variant": "transfer",
            "confmat": None,
            "roc": None,
            "metrics": None
        })
        save_hpc_results_json(self.all_results)

        self.hpc_progress.setValue(100)
        self.hpc_log.append("*** Transfer-learning finished ***")
        self.btn_hpc.setEnabled(True)

        # Refresh UI
        self._update_hpc_table()
        self._plot_hpc_combo_chart()
        self._plot_hpc_auc_chart()

        # Enable final-train button
        self.best_params_str = combo_str
        self.btn_final.setEnabled(True)

    # ------------------------------------------------------------------
    # █████████████████████████  WORKER CALLBACKS  █████████████████████
    # ------------------------------------------------------------------
    def _on_hpc_progress(self, val):               self.hpc_progress.setValue(val)
    def _on_hpc_log(self, msg):                    self.hpc_log.append(msg)

    def _on_hpc_combo_epoch(self, combo_idx, epoch_idx, tr_acc, val_acc):
        if combo_idx != self.current_combo_idx:
            self.current_combo_idx = combo_idx
            self.hpc_train_acc.clear(); self.hpc_val_acc.clear()
        self.hpc_train_acc.append(tr_acc); self.hpc_val_acc.append(val_acc)
        self._plot_hpc_training_chart()

    def _on_hpc_finished(self):
        self.hpc_log.append("=== HPC finished ===")
        self.hpc_progress.setValue(100)
        self.btn_hpc.setEnabled(True)

        if self.hpc_worker:
            self.all_results.extend(self.hpc_worker.all_results)
            # Deduplicate
            uniq = {}
            for rec in self.all_results:
                uniq[(rec["combo_str"], rec["model_path"])] = rec
            self.all_results = list(uniq.values())
            save_hpc_results_json(self.all_results)

            self.best_params_str = self.hpc_worker.best_params_str
            self.hpc_log.append(f"Best combo: {self.best_params_str}")
            if self.best_params_str:
                self.btn_final.setEnabled(True)

            self._update_hpc_table()
            self._plot_hpc_combo_chart()
            self._plot_hpc_auc_chart()
            self.hpc_updated.emit()

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
