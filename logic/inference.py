import os
import csv
import torch
import torch.nn.functional as F
from PIL import Image

from src.models import get_model
from src.transforms_config import create_transform

class InferenceModel:
    """
    Enhanced helper class for loading a trained model checkpoint
    and performing single-image or batch (folder) prediction.
    For 'double' variant: first we classify (NoTumor vs Tumor).
       If Tumor, we do the 3-class head to differentiate
       (Pituitary / Meningioma / Glioma).
    """

    def __init__(
        self,
        checkpoint_path,
        model_variant="double",
        input_size=128,
        normalization="zscore",
        device="cpu",
        dropout=0.0
    ):
        self.device = torch.device(device)

        # Build the chosen model
        self.model, _ = get_model(
            model_variant=model_variant,
            input_size=input_size,
            dropout=dropout
        )
        self.model.to(self.device)
        self.model.eval()

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model_variant = model_variant

        # Build transform (same logic as training)
        self.transform = create_transform(
            image_size=(input_size, input_size),
            augment=False
        )

    def predict(self, pil_img):
        """
        Single-image prediction.
        For 'double': we do binary => if "tumor",
                      then do threeclass => classify Pituitary / Meningioma / Glioma.

        Returns a dict containing:
          {
            "prediction": "NoTumor" or "Pituitary"/"Meningioma"/"Glioma",
            "binary_probs": [float, float],        # raw probabilities for no/tumor
            "binary_perc":  [str, str],            # e.g. ["85.2%", "14.8%"]
            "multi_probs":  [floats or None],      # 3-class probabilities if tumor
            "multi_perc":   [str or None],         # 3-class percentages if tumor
          }

        If not double, returns a simpler structure with:
          {
            "class_index": <int>,
            "probabilities": [float...]
          }
        """
        x = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.model_variant == "double":
                # 1) Binary (NoTumor vs Tumor)
                out_bin = self.model(x, mode='binary')
                bin_probs = F.softmax(out_bin, dim=1)[0]  # shape [2]
                bin_probs_float = [float(p) for p in bin_probs]
                bin_probs_percent = [f"{p*100:.1f}%" for p in bin_probs_float]

                # Check if tumor (binary index=1)
                if bin_probs[1] > bin_probs[0]:
                    # 2) Three-class => Pituitary / Meningioma / Glioma
                    out_3c = self.model(x, mode='threeclass')
                    multi_probs = F.softmax(out_3c, dim=1)[0]  # shape [3]
                    multi_probs_float = [float(p) for p in multi_probs]
                    multi_probs_percent = [f"{p*100:.1f}%" for p in multi_probs_float]

                    subtype_idx = multi_probs.argmax().item()
                    # Tumor labels
                    tumor_labels = ["Pituitary", "Meningioma", "Glioma"]
                    subtype_label = tumor_labels[subtype_idx]

                    return {
                        "prediction": subtype_label,
                        "binary_probs": bin_probs_float,
                        "binary_perc": bin_probs_percent,
                        "multi_probs": multi_probs_float,
                        "multi_perc": multi_probs_percent
                    }
                else:
                    # NoTumor
                    return {
                        "prediction": "NoTumor",
                        "binary_probs": bin_probs_float,
                        "binary_perc": bin_probs_percent,
                        "multi_probs": None,
                        "multi_perc": None
                    }
            else:
                # Single-head (binary, threeclass, or multi)
                out = self.model(x)
                probs = F.softmax(out, dim=1)[0]
                probs_float = [float(p) for p in probs]
                class_idx = probs.argmax().item()
                return {
                    "class_index": class_idx,
                    "probabilities": probs_float
                }

    def predict_batch(
        self,
        image_paths=None,
        folder_path=None,
        export_csv=None
    ):
        """
        Batch inference on multiple images.

        Each returned dict has the same keys as `predict()`.
        - If double => "prediction", "binary_probs", "multi_probs" etc.
        - Otherwise => "class_index", "probabilities".
        """
        # Gather the file list from either `image_paths` or `folder_path`.
        file_list = []
        if image_paths:
            file_list = list(image_paths)
        elif folder_path:
            if not os.path.isdir(folder_path):
                raise ValueError(f"Folder not found: {folder_path}")
            all_files = os.listdir(folder_path)
            # pick images only
            for f in all_files:
                ext = f.lower()
                if ext.endswith(".png") or ext.endswith(".jpg") or ext.endswith(".jpeg"):
                    file_list.append(os.path.join(folder_path, f))
        else:
            raise ValueError("Must provide image_paths or folder_path for batch inference.")

        results = []
        for img_path in file_list:
            if not os.path.isfile(img_path):
                # skip invalid
                continue
            try:
                pil_img = Image.open(img_path).convert('L')
            except Exception as e:
                results.append({
                    "file": img_path,
                    "error": f"Could not open: {e}"
                })
                continue

            # Single predict
            pred_dict = self.predict(pil_img)
            pred_dict["file"] = img_path
            results.append(pred_dict)

        # optionally export CSV
        if export_csv:
            self._export_results_csv(results, export_csv)

        return results

    def _predict_internal(self, pil_img):
        """
        Internal helper that just returns the full record:
        For 'double': {prediction, binary_probs, multi_probs}
        For 'binary': {class_index, probabilities}
        For 'multi':  {class_index, probabilities}
        """
        x = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.model_variant == "double":
                out_bin = self.model(x, mode='binary')
                bin_probs = F.softmax(out_bin, dim=1)[0]
                if bin_probs[1] > bin_probs[0]:
                    out_multi = self.model(x, mode='multi')
                    multi_probs = F.softmax(out_multi, dim=1)[0]
                    subtype_idx = multi_probs.argmax().item()
                    return {
                        "prediction": f"tumor_subtype_{subtype_idx}",
                        "binary_probs": bin_probs.tolist(),
                        "multi_probs": multi_probs.tolist()
                    }
                else:
                    return {
                        "prediction": "no_tumor",
                        "binary_probs": bin_probs.tolist(),
                        "multi_probs": None
                    }
            else:
                out = self.model(x)
                probs = F.softmax(out, dim=1)[0]
                pred_idx = probs.argmax().item()
                return {
                    "class_index": int(pred_idx),
                    "probabilities": probs.tolist()
                }

    def _export_results_csv(self, results, csv_path):
        """
        Exports batch inference results to a CSV file.
        For double, we store columns for prediction, binary & multi probabilities, etc.
        """
        fieldnames = [
            "file",
            "prediction",
            "binary_probs", "binary_perc",
            "multi_probs",  "multi_perc",
            "class_index",  "probabilities",
            "error"
        ]
        try:
            with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in results:
                    writer.writerow(row)
        except Exception as e:
            print(f"[WARN] Could not export CSV to {csv_path}: {e}")
