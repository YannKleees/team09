import os
from PIL import Image
from torch.utils.data import Dataset

class BrainTumorDataset(Dataset):
    """
    A custom dataset that can handle multiple model variants:
      - binary: 2-class => 0=NoTumor, 1=Tumor (Pituitary/Meningioma/Glioma)
      - threeclass: 3-class => 0=Pituitary, 1=Meningioma, 2=Glioma (no NoTumor)
      - simple (or multi): 4-class => 0=NoTumor, 1=Pituitary, 2=Meningioma, 3=Glioma
      - double: still store (0=NoTumor,1=Pituitary,2=Meningioma,3=Glioma). The
        "double" model will read 0 => no tumor, else => yes tumor in the
        training logic.  (We do not do separate label logic here.)
    """

    def __init__(self, root_dir, subset="Training", transform=None, model_variant="simple"):
        super().__init__()
        self.root_dir = root_dir
        self.subset = subset
        self.transform = transform
        self.model_variant = model_variant.lower()

        self.data_path = os.path.join(self.root_dir, self.subset)
        if not os.path.isdir(self.data_path):
            raise FileNotFoundError(f"{self.data_path} does not exist. Please check your folder names.")

        self.samples = []
        self.classes = []

        if self.model_variant in ("simple", "multi", "lpconv"):
            # numeric labels => 0=NoTumor, 1=Pituitary, 2=Meningioma, 3=Glioma
            label_map = {
                "notumor": 0,
                "pituitary": 1,
                "meningioma": 2,
                "glioma": 3
            }
            self.classes = ["NoTumor", "Pituitary", "Meningioma", "Glioma"]
            self._gather_subfolders(label_map)

        elif self.model_variant == "binary":
            self.classes = ["NoTumor", "Tumor"]
            self._gather_binary()

        elif self.model_variant == "threeclass":
                label_map = {
                    "pituitary": 0,
                    "meningioma": 1,
                    "glioma": 2
                }
                self.classes = ["Pituitary", "Meningioma", "Glioma"]
                self._gather_subfolders(label_map)

        elif self.model_variant == "double":
            label_map = {
                "notumor": 0,
                "pituitary": 1,
                "meningioma": 2,
                "glioma": 3
            }
            self.classes = ["NoTumor", "Pituitary", "Meningioma", "Glioma"]
            self._gather_subfolders(label_map)

        else:
            raise ValueError(f"Unknown model_variant: {self.model_variant}")

    def _gather_subfolders(self, label_map):
        """ Helper that scans self.data_path for subfolders in label_map. """
        for subf in os.listdir(self.data_path):
            sf_path = os.path.join(self.data_path, subf)
            if not os.path.isdir(sf_path):
                continue
            sf_lower = subf.lower()
            if sf_lower in label_map:
                label_idx = label_map[sf_lower]
                for fname in os.listdir(sf_path):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        full_path = os.path.join(sf_path, fname)
                        self.samples.append((full_path, label_idx))

    def _gather_binary(self):
        """
        In 'binary' variant: 0=NoTumor, else 1=Tumor.
        So if subfolder is exactly 'NoTumor', label=0, otherwise label=1.
        """
        for subf in os.listdir(self.data_path):
            sf_path = os.path.join(self.data_path, subf)
            if not os.path.isdir(sf_path):
                continue

            sf_lower = subf.lower()
            if sf_lower == "notumor":
                label_idx = 0
            else:
                label_idx = 1

            for fname in os.listdir(sf_path):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    full_path = os.path.join(sf_path, fname)
                    self.samples.append((full_path, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def is_empty(self):
        return (len(self.samples) == 0)
