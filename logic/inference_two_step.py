# logic/inference_two_step.py

import os
import torch
import torch.nn.functional as F
from PIL import Image

from src.models import get_model
from src.transforms_config import create_transform


class BinaryInferenceModel:
    """
    Single-head HPC model that does binary classification:
      0 => NoTumor
      1 => Tumor
    """
    def __init__(self, checkpoint_path, input_size=128, device="cpu", dropout=0.0):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Binary checkpoint not found: {checkpoint_path}")
        self.device = torch.device(device)

        # Build a "binary" net => get_model("binary", input_size=..., dropout=...)
        self.model, _ = get_model("binary", input_size=input_size, dropout=dropout)
        self.model.to(self.device)
        self.model.eval()

        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)

        self.transform = create_transform(
            image_size=(input_size, input_size),
            augment=False
        )

    def predict(self, pil_img):
        x = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)  # shape [1,2]
            probs = F.softmax(logits, dim=1)[0]
            pred_idx = int(probs.argmax().item())
        return {
            "prediction": pred_idx,  # 0 => no tumor, 1 => tumor
            "prob": probs.cpu().tolist()
        }


class ThreeClassInferenceModel:
    """
    Single-head HPC model that does 3-class:
      0 => Pituitary
      1 => Meningioma
      2 => Glioma
    """
    def __init__(self, checkpoint_path, input_size=128, device="cpu", dropout=0.0):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Threeclass checkpoint not found: {checkpoint_path}")
        self.device = torch.device(device)

        self.model, _ = get_model("threeclass", input_size=input_size, dropout=dropout)
        self.model.to(self.device)
        self.model.eval()

        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)

        self.transform = create_transform(
            image_size=(input_size, input_size),
            augment=False
        )
        self.labels = ["Pituitary", "Meningioma", "Glioma"]

    def predict(self, pil_img):
        x = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)[0]
            idx = int(probs.argmax().item())
        return {
            "prediction_idx": idx,
            "prediction_label": self.labels[idx],
            "prob": probs.cpu().tolist()
        }


class TwoStepDoublePipeline:
    """
    Combines a binary HPC model + a threeclass HPC model
    to replicate a 'double' approach without training a separate net.

    Steps:
      1) Binary => no tumor => done
         If tumor => classify pituitary/meningioma/glioma
    """
    def __init__(
        self,
        binary_checkpoint,
        three_checkpoint,
        input_size=128,
        device="cpu",
        dropout=0.0
    ):
        self.binary_model = BinaryInferenceModel(
            checkpoint_path=binary_checkpoint,
            input_size=input_size,
            device=device,
            dropout=dropout
        )
        self.three_model = ThreeClassInferenceModel(
            checkpoint_path=three_checkpoint,
            input_size=input_size,
            device=device,
            dropout=dropout
        )

    def predict_image(self, pil_img):
        bin_res = self.binary_model.predict(pil_img)  # {prediction=0 or 1, prob=[pNo, pTumor]}
        if bin_res["prediction"] == 0:
            # No Tumor => done
            return {
                "final_prediction": "NoTumor",
                "binary_prob": bin_res["prob"],
                "threeclass_prob": None
            }
        else:
            # Tumor => check 3-class
            three_res = self.three_model.predict(pil_img)
            return {
                "final_prediction": three_res["prediction_label"],
                "binary_prob": bin_res["prob"],
                "threeclass_prob": three_res["prob"]
            }
