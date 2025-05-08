# File: logic/auto_experiment.py

import itertools
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from src.experiment import Experiment


class AutoSearchWorker(QThread):
    """
    A QThread that runs a hyperparameter search on the specified HPC ranges.
    After finishing, it exposes `best_params`, `best_acc`, and a full log of combos in `searcher`.
    """

    progress = pyqtSignal(int)   # emits integer for progress [0..100]
    log_msg = pyqtSignal(str)    # emits str messages
    finished = pyqtSignal()      # emitted when search is done

    def __init__(self, parent=None):
        super().__init__(parent)
        self.searcher = None  # We store HPC results here

        # The user-provided HPC space
        self.data_root = "cleaned_data"
        self.resize_list = [128]
        self.disable_augment_list = [False]
        self.adv_aug_list = [False]
        self.dropout_list = [0.0]
        self.weight_decay_list = [0.0]
        self.earlystop_list = [0]
        self.class_weight_list = [False]
        self.batch_size = 16
        self.lr = 0.001
        self.epochs = 5
        self.num_classes = 2
        self.device = "cpu"

    def setup_search(
        self,
        data_root,
        resize_list,
        disable_augment_list,
        adv_aug_list,
        dropout_list,
        weight_decay_list,
        earlystop_list,
        class_weight_list,
        batch_size,
        lr,
        epochs,
        num_classes,
        device="cpu"
    ):
        """
        Called by the UI to pass in HPC settings. We'll store them, so run() can do the search.
        """
        self.data_root = data_root
        self.resize_list = resize_list
        self.disable_augment_list = disable_augment_list
        self.adv_aug_list = adv_aug_list
        self.dropout_list = dropout_list
        self.weight_decay_list = weight_decay_list
        self.earlystop_list = earlystop_list
        self.class_weight_list = class_weight_list
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.num_classes = num_classes
        self.device = device

        # Prepare a simple container to store HPC results
        self.searcher = HPCResults()

    def run(self):
        """
        Execute a brute-force HPC search, calling a mini-train method for each combo.
        We store the best combo by val_acc in self.searcher.best_params and best_acc.
        """
        # Build the HPC combos in Pythonic style
        combos = list(itertools.product(
            self.resize_list,
            self.disable_augment_list,
            self.adv_aug_list,
            self.dropout_list,
            self.weight_decay_list,
            self.earlystop_list,
            self.class_weight_list
        ))
        total = len(combos)
        if total == 0:
            self.log_msg.emit("[AutoSearchWorker] No HPC combos specified!")
            self.finished.emit()
            return

        self.log_msg.emit(f"[AutoSearchWorker] Searching over {total} combos...")

        best_acc = -1.0
        best_params_str = ""
        for idx, combo in enumerate(combos):
            resize_val = combo[0]
            disable_aug = combo[1]
            adv_aug = combo[2]
            dropout = combo[3]
            w_decay = combo[4]
            earlystop = combo[5]
            cw = combo[6]

            # Create an Experiment with these HPC
            # For demonstration, we do a short training or direct skip
            # We'll do a small 'simulate_train' or a real partial train

            val_acc = self._simulate_train(
                resize_val=resize_val,
                disable_aug=disable_aug,
                adv_aug=adv_aug,
                dropout=dropout,
                weight_decay=w_decay,
                earlystop=earlystop,
                class_weight=cw
            )

            # Build param string
            combo_str = (
                f"resize={resize_val}, disable_aug={disable_aug}, adv_aug={adv_aug}, "
                f"dropout={dropout}, w_decay={w_decay}, earlystop={earlystop}, cw={cw}"
            )
            # store in HPCResults
            self.searcher.add_result(combo_str, val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                best_params_str = combo_str

            # progress
            pct = int(100*(idx+1)/total)
            self.progress.emit(pct)
            self.log_msg.emit(f"[{idx+1}/{total}] HPC combo => {combo_str}, val_acc={val_acc:.2f}")

        self.searcher.best_acc = best_acc
        self.searcher.best_params = best_params_str

        self.log_msg.emit(f"[AutoSearchWorker] Done! Best Acc={best_acc:.2f} => {best_params_str}")
        self.finished.emit()

    def _simulate_train(
        self,
        resize_val,
        disable_aug,
        adv_aug,
        dropout,
        weight_decay,
        earlystop,
        class_weight
    ):
        """
        A quick approach: either do a *short real training*, or in this example,
        we'll just generate a random number to simulate 'val_acc'.
        In your real code, you might build an `Experiment` and call run_single_training.
        """
        import random
        # If you want to do partial real training, you might do:
        # exp = Experiment(
        #     data_root=self.data_root,
        #     model_variant="binary" if self.num_classes==2 else "multi",
        #     input_size=resize_val,
        #     batch_size=self.batch_size,
        #     lr=self.lr,
        #     epochs=1,  # maybe fewer epochs to speed HPC
        #     disable_augment=disable_aug,
        #     adv_aug=adv_aug,
        #     dropout=dropout,
        #     weight_decay=weight_decay,
        #     earlystop_patience=earlystop,
        #     use_class_weight=class_weight,
        #     device=self.device
        # )
        # exp.load_data()
        # best_acc, _, _ = exp.run_single_training(epochs=1)
        # return best_acc

        # For demonstration, just return a random val_acc
        return random.uniform(60, 98)


class HPCResults:
    """
    Holds the best HPC param string & val_acc, and a list of all combos.
    """
    def __init__(self):
        self.best_params = ""
        self.best_acc = 0.0
        self.results = []  # list of (param_str, val_acc)

    def add_result(self, param_str, val_acc):
        self.results.append((param_str, val_acc))
