# File: src/dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset

class BrainTumorDataset(Dataset):
    """
    Expects structure:
      data_root/
         Training/
             ClassA/
             ClassB/
             ...
         Testing/
             ClassA/
             ClassB/
             ...
    Each class folder has .png/.jpg images.

    If we are told model_variant=='binary', then we unify:
       "notumor" => label 0
       anything else => label 1
    """

    def __init__(self, root_dir, subset="Training", transform=None, model_variant="multi"):
        super().__init__()
        self.root_dir = root_dir
        self.subset = subset
        self.transform = transform
        self.model_variant = model_variant

        self.data_path = os.path.join(self.root_dir, self.subset)
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"{self.data_path} does not exist.")

        # Gather all class names (folder names)
        all_classes = sorted([
            d for d in os.listdir(self.data_path)
            if os.path.isdir(os.path.join(self.data_path, d))
        ])

        if self.model_variant == "binary":
            self.classes = ["notumor", "tumor"]
            self.orig_class_to_label = {}
            for c in all_classes:
                # adapt this check if your folder name is something else (e.g. "no_tumor")
                if c.lower().startswith("notumor"):
                    self.orig_class_to_label[c] = 0
                else:
                    self.orig_class_to_label[c] = 1
        else:
            # normal multi/double
            self.classes = all_classes
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.samples = []
        for class_name in all_classes:
            class_dir = os.path.join(self.data_path, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(class_dir, fname)
                    if self.model_variant == "binary":
                        label_idx = self.orig_class_to_label[class_name]
                    else:
                        label_idx = self.class_to_idx[class_name]
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
        """Returns True if this dataset has zero samples."""
        return (len(self.samples) == 0)
