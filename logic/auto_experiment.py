import os
import time
import itertools
import shutil

from PyQt5.QtCore import QThread, pyqtSignal

from logic.preprocessing import preprocess_entire_dataset
from src.experiment import Experiment


##############################################################################
# Single-step Workers
##############################################################################
class StepAWorker(QThread):
    """
    Runs Step A only: pipeline search (resize, norm, augment).
    Saves best pipeline to 'best_pipeline.txt'.
    """
    log_msg = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, data_root, device="cpu", parent=None):
        super().__init__(parent)
        self.data_root = data_root
        self.device = device

    def run(self):
        self.log_msg.emit("Running Step A (Pipeline Search)...")
        # Only pipeline hyperparams; fix model hyperparams
        searcherA = HyperparamSearcher(
            data_root=self.data_root,
            resize_list=[128, 160, 224],
            disable_augment_list=[False, True],
            adv_aug_list=[False, True],
            # fixed model side
            dropout_list=[0.0],
            weight_decay_list=[0.0],
            earlystop_list=[0],
            class_weight_list=[False],
            batch_size=16,
            lr=1e-3,
            epochs=3,
            device=self.device,
            reprocess_mode="once_per_pipeline"
        )
        bestA_params, bestA_acc, _ = searcherA.run_search_with_logging(log_func=self.log_callback)
        self.log_msg.emit(f"\nStep A Completed. Best Pipeline => {bestA_params} ({bestA_acc:.2f}%)")

        # Save in best_pipeline.txt
        with open("best_pipeline.txt", "w", encoding="utf-8") as f:
            f.write(f"{bestA_params}\n")
            f.write(f"{bestA_acc}\n")

        self.finished.emit()

    def log_callback(self, msg):
        self.log_msg.emit(msg)


class StepBWorker(QThread):
    """
    Runs Step B only: search model hyperparams, reading best pipeline from 'best_pipeline.txt'.
    Saves best combo in 'best_model.txt'.
    """
    log_msg = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, data_root, device="cpu", parent=None):
        super().__init__(parent)
        self.data_root = data_root
        self.device = device

    def run(self):
        self.log_msg.emit("Running Step B (Model Hyperparams)...")

        # Load best pipeline
        if not os.path.isfile("best_pipeline.txt"):
            self.log_msg.emit("[ERROR] 'best_pipeline.txt' not found. Run Step A first.")
            self.finished.emit()
            return

        with open("best_pipeline.txt", "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
            best_pipeline_str = lines[0]

        pipe_dict = self._parse_pipeline_str(best_pipeline_str)
        chosen_resize = int(pipe_dict["resize"])
        chosen_norm = pipe_dict["norm"]
        chosen_disable_aug = (pipe_dict["disable_aug"] == "True")
        chosen_adv_aug = (pipe_dict["adv_aug"] == "True")

        # Now search model hyperparams, fix pipeline
        searcherB = HyperparamSearcher(
            data_root=self.data_root,
            resize_list=[chosen_resize],
            disable_augment_list=[chosen_disable_aug],
            adv_aug_list=[chosen_adv_aug],
            dropout_list=[0.0, 0.3, 0.5],
            weight_decay_list=[0.0, 0.0001, 0.001],
            earlystop_list=[0, 3],
            class_weight_list=[False, True],
            batch_size=16,
            lr=1e-3,
            epochs=5,
            device=self.device,
            reprocess_mode="none"
        )
        bestB_params, bestB_acc, _ = searcherB.run_search_with_logging(log_func=self.log_callback)
        self.log_msg.emit(f"\nStep B Completed. Best => {bestB_params} ({bestB_acc:.2f}%)")

        with open("best_model.txt", "w", encoding="utf-8") as f:
            f.write(f"{bestB_params}\n")
            f.write(f"{bestB_acc}\n")

        self.finished.emit()

    def log_callback(self, msg):
        self.log_msg.emit(msg)

    def _parse_pipeline_str(self, combo_str):
        """
        e.g. 'resize=224, norm=minmax, disable_aug=False, adv_aug=False, dropout=0.0, ...'
        """
        d = {}
        parts = combo_str.split(",")
        for p in parts:
            p = p.strip()
            kv = p.split("=")
            if len(kv) == 2:
                k = kv[0].strip()
                v = kv[1].strip()
                d[k] = v
        return d


class StepCWorker(QThread):
    """
    Step C: final refinement pass.
    Reads 'best_model.txt' and does a smaller search around it, saving in 'best_final.txt'.
    """
    log_msg = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, data_root, device="cpu", parent=None):
        super().__init__(parent)
        self.data_root = data_root
        self.device = device

    def run(self):
        self.log_msg.emit("Running Step C (Refinement)...")

        if not os.path.isfile("best_model.txt"):
            self.log_msg.emit("[ERROR] 'best_model.txt' not found. Run Step B first.")
            self.finished.emit()
            return

        with open("best_model.txt", "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
            best_model_str = lines[0]

        combo_dict = self._parse_combo(best_model_str)

        resize_val = int(combo_dict["resize"])
        norm_str = combo_dict["norm"]
        dis_aug = (combo_dict["disable_aug"] == "True")
        adv_aug = (combo_dict["adv_aug"] == "True")

        best_dropout = float(combo_dict["dropout"])
        best_wd = float(combo_dict["w_decay"])
        best_es = int(combo_dict["earlystop"])
        best_cw = (combo_dict["cw"] == "True")

        # define smaller search around them
        dropout_list = [best_dropout, best_dropout + 0.1]
        dropout_list = [x for x in dropout_list if x >= 0]

        if best_wd <= 0:
            wd_list = [0.0, 1e-5, 1e-4]
        else:
            wd_list = [best_wd / 2, best_wd, best_wd * 2]

        es_list = [best_es, best_es + 2]
        cw_list = [best_cw, not best_cw]

        self.log_msg.emit(
            "Refinement combos => dropout={}, wd={}, earlystop={}, cw={}".format(
                dropout_list, wd_list, es_list, cw_list
            )
        )

        searcherC = HyperparamSearcher(
            data_root=self.data_root,
            resize_list=[resize_val],
            disable_augment_list=[dis_aug],
            adv_aug_list=[adv_aug],
            dropout_list=dropout_list,
            weight_decay_list=wd_list,
            earlystop_list=es_list,
            class_weight_list=cw_list,
            batch_size=16,
            lr=1e-3,
            epochs=6,
            device=self.device,
            reprocess_mode="none"
        )
        bestC_params, bestC_acc, _ = searcherC.run_search_with_logging(log_func=self.log_callback)
        self.log_msg.emit(f"\nStep C Completed. Best => {bestC_params} ({bestC_acc:.2f}%)")

        with open("best_final.txt", "w", encoding="utf-8") as f:
            f.write(f"{bestC_params}\n{bestC_acc}\n")

        self.finished.emit()

    def log_callback(self, msg):
        self.log_msg.emit(msg)

    def _parse_combo(self, combo_str):
        d = {}
        for part in combo_str.split(","):
            part = part.strip()
            kv = part.split("=")
            if len(kv) == 2:
                k = kv[0].strip()
                v = kv[1].strip()
                d[k] = v
        return d


##############################################################################
# Single unified full-search Worker
##############################################################################
class AutoSearchWorker(QThread):
    """
    Runs a single big HyperparamSearcher for "Run Full Auto Search."
    """
    progress = pyqtSignal(int)
    log_msg = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.searcher = None

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
        num_classes=4,
        device="cpu"
    ):
        self.searcher = HyperparamSearcher(
            data_root=data_root,
            resize_list=resize_list,
            disable_augment_list=disable_augment_list,
            adv_aug_list=adv_aug_list,
            dropout_list=dropout_list,
            weight_decay_list=weight_decay_list,
            earlystop_list=earlystop_list,
            class_weight_list=class_weight_list,
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            num_classes=num_classes,
            device=device
        )

    def log_callback(self, msg):
        self.log_msg.emit(msg)

    def run(self):
        if not self.searcher:
            self.log_msg.emit("No HyperparamSearcher set up!")
            self.finished.emit()
            return

        best_params, best_acc, results = self.searcher.run_search_with_logging(
            log_func=self.log_callback
        )
        self.log_msg.emit("\n=== AutoSearch Finished ===")
        self.log_msg.emit(f"Best Validation Accuracy: {best_acc:.2f}%")
        self.log_msg.emit(f"Best Params: {best_params}")

        self.log_msg.emit("\n--- All Results ---")
        for combo_str, val_acc, status in results:
            self.log_msg.emit(f"   {combo_str} => val={val_acc:.2f}% ({status})")

        self.progress.emit(100)
        self.finished.emit()


##############################################################################
# Stepwise A->B
##############################################################################
class StepwiseSearchManager:
    """
    Demonstration of a 2-step approach:
      Step A: search pipeline hyperparams
      Step B: fix pipeline, vary model hyperparams
    """
    def __init__(self, data_root, device="cpu", log_func=print):
        self.data_root = data_root
        self.device = device
        self.log_func = log_func

    def run_all_steps(self):
        self.log_func("==== Step A: Searching pipeline hyperparams ====")
        searcherA = HyperparamSearcher(
            data_root=self.data_root,
            resize_list=[128, 160, 224],
            disable_augment_list=[False, True],
            adv_aug_list=[False, True],
            # fix model side
            dropout_list=[0.0],
            weight_decay_list=[0.0],
            earlystop_list=[0],
            class_weight_list=[False],
            batch_size=16,
            lr=1e-3,
            epochs=3,
            device=self.device,
            reprocess_mode="once_per_pipeline"
        )
        bestA_params, bestA_acc, _ = searcherA.run_search_with_logging(log_func=self.log_func)
        self.log_func(f"Step A best => {bestA_params} ({bestA_acc:.2f}%)")

        # parse pipeline
        bestA_kv = self._parse_combo_str(bestA_params)
        chosen_resize = int(bestA_kv["resize"])
        chosen_norm = bestA_kv["norm"]
        chosen_disable_aug = (bestA_kv["disable_aug"] == "True")
        chosen_adv_aug = (bestA_kv["adv_aug"] == "True")

        self.log_func("\n==== Step B: Searching model hyperparams ====")
        searcherB = HyperparamSearcher(
            data_root=self.data_root,
            resize_list=[chosen_resize],
            disable_augment_list=[chosen_disable_aug],
            adv_aug_list=[chosen_adv_aug],
            dropout_list=[0.0, 0.3, 0.5],
            weight_decay_list=[0.0, 0.0001, 0.001],
            earlystop_list=[0, 3],
            class_weight_list=[False, True],
            batch_size=16,
            lr=1e-3,
            epochs=5,
            device=self.device,
            reprocess_mode="none"
        )
        bestB_params, bestB_acc, _ = searcherB.run_search_with_logging(log_func=self.log_func)
        self.log_func(f"Step B best => {bestB_params} ({bestB_acc:.2f}%)")

        self.log_func("\n=== Stepwise Search Completed ===")

    def _parse_combo_str(self, combo_str):
        d = {}
        for p in combo_str.split(","):
            p = p.strip()
            kv = p.split("=")
            if len(kv) == 2:
                k = kv[0].strip()
                v = kv[1].strip()
                d[k] = v
        return d


##############################################################################
# The Master HyperparamSearcher
##############################################################################
class HyperparamSearcher:
    """
    Searches combos of:
      - resize_val
      - norm_str
      - disable_augment
      - adv_aug
      - dropout
      - weight_decay
      - earlystop
      - use_class_weight
    Returns (best_params_string, best_acc, results).
    """
    def __init__(
        self,
        data_root,
        resize_list=[128, 160],
        disable_augment_list=[False, True],
        adv_aug_list=[False, True],
        dropout_list=[0.0, 0.3],
        weight_decay_list=[0.0, 0.0001],
        earlystop_list=[0, 3],
        class_weight_list=[False, True],
        batch_size=16,
        lr=1e-3,
        epochs=5,
        num_classes=4,
        device="cpu",

        # Reprocessing mode
        reprocess_mode="once_per_pipeline",
        base_temp_dir="tmp_AutoSearch_experiments"
    ):
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

        self.reprocess_mode = reprocess_mode
        self.base_temp_dir = base_temp_dir

        self.results = []
        self.best_acc = 0.0
        self.best_params = None

        # pipeline cache if reprocess_mode="once_per_pipeline"
        self.pipeline_cache = {}

    def run_search_with_logging(self, log_func=print):
        combos = itertools.product(
            self.resize_list,
            self.disable_augment_list,
            self.adv_aug_list,
            self.dropout_list,
            self.weight_decay_list,
            self.earlystop_list,
            self.class_weight_list
        )

        os.makedirs(self.base_temp_dir, exist_ok=True)

        for combo in combos:
            (
                resize_val,
                norm_str,
                disable_aug,
                adv_aug,
                dropout,
                w_decay,
                earlystop,
                use_class_weight
            ) = combo

            combo_str = (
                f"resize={resize_val}, norm={norm_str}, "
                f"disable_aug={disable_aug}, adv_aug={adv_aug}, "
                f"dropout={dropout}, w_decay={w_decay}, "
                f"earlystop={earlystop}, cw={use_class_weight}"
            )
            log_func(f"\n=== Testing Combo: {combo_str} ===")

            t0 = time.time()

            # Decide on data folder
            if self.reprocess_mode == "none":
                out_dir = self.data_root
                log_func("  >> Skipping preprocessing (reprocess_mode='none').")
            else:
                pipeline_key = (resize_val, norm_str)
                if self.reprocess_mode == "always":
                    out_dir = os.path.join(
                        self.base_temp_dir,
                        f"pipeline_{resize_val}_{norm_str}_{int(time.time())}"
                    )
                    os.makedirs(out_dir, exist_ok=True)
                    log_func(f"  >> Reprocessing into: {out_dir}")
                    try:
                        _, _, total, dups_removed, elapsed_sec = preprocess_entire_dataset(
                            root_dir=self.data_root,
                            dst_root=out_dir,
                            resize_val=resize_val,
                            norm_method=(None if norm_str == "none" else norm_str),
                            delete_duplicates=True
                        )
                        log_func(
                            f"    [Preproc time: {elapsed_sec:.2f}s, images={total}, "
                            f"dups_removed={dups_removed}]"
                        )
                    except Exception as e:
                        log_func(f"  !! Preprocessing error: {e}")
                        self.results.append((combo_str, 0.0, "error"))
                        continue

                elif self.reprocess_mode == "once_per_pipeline":
                    if pipeline_key not in self.pipeline_cache:
                        out_dir = os.path.join(
                            self.base_temp_dir,
                            f"pipeline_{resize_val}_{norm_str}"
                        )
                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir, exist_ok=True)
                            log_func(f"  >> Preprocessing into: {out_dir}")
                            try:
                                _, _, total, dups_removed, elapsed_sec = preprocess_entire_dataset(
                                    root_dir=self.data_root,
                                    dst_root=out_dir,
                                    resize_val=resize_val,
                                    norm_method=(None if norm_str == "none" else norm_str),
                                    delete_duplicates=True
                                )
                                log_func(
                                    f"    [Preproc time: {elapsed_sec:.2f}s, images={total}, "
                                    f"dups_removed={dups_removed}]"
                                )
                            except Exception as e:
                                log_func(f"  !! Preprocessing error: {e}")
                                self.results.append((combo_str, 0.0, "error"))
                                continue
                        else:
                            log_func("  >> Reusing existing pipeline folder.")
                        self.pipeline_cache[pipeline_key] = out_dir
                    else:
                        out_dir = self.pipeline_cache[pipeline_key]
                        log_func("  >> Using cached pipeline folder, skip reprocessing.")
                else:
                    # fallback
                    out_dir = self.data_root
                    log_func("  [Warning] reprocess_mode not recognized; skipping preprocess.")

            # Train
            best_val_acc = 0.0
            try:
                exp = Experiment(
                    data_root=out_dir,
                    batch_size=self.batch_size,
                    lr=self.lr,
                    epochs=self.epochs,
                    disable_augment=disable_aug,
                    adv_aug=adv_aug,
                    dropout=dropout,
                    weight_decay=w_decay,
                    earlystop_patience=earlystop,
                    use_class_weight=use_class_weight,
                    input_size=resize_val,
                    device=self.device
                )
                exp.load_data()

                for ep in range(self.epochs):
                    start_ep = time.time()
                    train_loss, train_acc, _ = exp.train_one_epoch(ep)
                    val_metrics = exp.evaluate(ep)
                    end_ep = time.time()

                    val_acc = val_metrics["accuracy"]
                    overfit_gap = train_acc - val_acc
                    overfit_note = f"(gap={overfit_gap:.1f}%)" if overfit_gap > 5.0 else ""

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        exp.reset_earlystop()
                    else:
                        exp.update_earlystop()

                    log_func(
                        f"    [Epoch {ep+1}/{self.epochs}] "
                        f"TrainLoss={train_loss:.4f}, TrainAcc={train_acc:.2f}%, "
                        f"ValAcc={val_acc:.2f}% {overfit_note}, "
                        f"Time={end_ep - start_ep:.2f}s"
                    )
                    if exp.earlystop_triggered:
                        log_func(f"    [Early Stopping triggered at epoch {ep+1}]")
                        break

                log_func(f"  >> Final Val Acc: {best_val_acc:.2f}%")
                if best_val_acc > self.best_acc:
                    self.best_acc = best_val_acc
                    self.best_params = combo_str

                self.results.append((combo_str, best_val_acc, "ok"))
            except Exception as e:
                log_func(f"  !! Training error: {e}")
                self.results.append((combo_str, 0.0, "error"))

            total_time = time.time() - t0
            log_func(f"  [Combo done in {total_time:.2f}s] => BestVal={best_val_acc:.2f}%")

        return self.best_params, self.best_acc, self.results
