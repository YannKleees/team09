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
         Testing/
             ClassA/
             ClassB/
    Each class folder has .png/.jpg images.
    """
    def __init__(self, root_dir, subset="Training", transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.subset = subset
        self.transform = transform

        self.data_path = os.path.join(self.root_dir, self.subset)
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"{self.data_path} does not exist.")

        self.classes = sorted([
            d for d in os.listdir(self.data_path)
            if os.path.isdir(os.path.join(self.data_path, d))
        ])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.data_path, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(class_dir, fname)
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
