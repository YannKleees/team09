# File: logic/inference.py

import os
import torch
from PIL import Image
import torch.nn.functional as F

from src.models import get_model
from src.transforms_config import create_transform

class InferenceModel:
    """
    Simple helper class for loading a trained model checkpoint
    and performing a single-image prediction.
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

        # Build the chosen model; note that for 'double' we don't pass num_classes
        self.model, _ = get_model(
            model_variant=model_variant,
            input_size=input_size,
            dropout=dropout
        )
        self.model.to(self.device)
        self.model.eval()

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model_variant = model_variant

        # Build transform (same logic as training)
        norm_str = (None if normalization == "none" else normalization)
        self.transform = create_transform(
            image_size=(input_size, input_size),
            normalization=norm_str,
            augment=False
        )

    def predict(self, pil_img):
        """
        Returns a dict with predictions and probabilities:
         - For 'double': 
              {
                "prediction": "no_tumor" OR "tumor_subtype_X",
                "binary_probs": [...],
                "multi_probs": [...]
              }
         - For 'multi':
              {
                "class_index": int,
                "probabilities": [...]
              }
         - For 'binary':
              {
                "class_index": 0 or 1,
                "probabilities": [...]
              }
        """
        x = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.model_variant == "double":
                out_bin = self.model(x, mode='binary')
                bin_probs = F.softmax(out_bin, dim=1)[0]  # shape [2]
                if bin_probs[1] > bin_probs[0]:
                    # It's a tumor => run multi
                    out_multi = self.model(x, mode='multi')
                    multi_probs = F.softmax(out_multi, dim=1)[0]  # shape [3]
                    subtype_idx = multi_probs.argmax().item()
                    return {
                        "prediction": f"tumor_subtype_{subtype_idx}",
                        "binary_probs": bin_probs.tolist(),
                        "multi_probs": multi_probs.tolist()
                    }
                else:
                    return {
                        "prediction": "no_tumor",
                        "binary_probs": bin_probs.tolist()
                    }

            else:
                # single-head (binary or multi)
                out = self.model(x)
                probs = F.softmax(out, dim=1)[0]
                pred_idx = probs.argmax().item()
                return {
                    "class_index": int(pred_idx),
                    "probabilities": probs.tolist()
                }
