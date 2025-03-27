import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

def _normalize_minmax():
    def f(img_tensor):
        min_val = img_tensor.min()
        max_val = img_tensor.max()
        if max_val - min_val > 1e-5:
            img_tensor = (img_tensor - min_val) / (max_val - min_val)
        else:
            img_tensor = img_tensor - min_val
        return img_tensor
    return f

def _normalize_zscore():
    def f(img_tensor):
        mean_val = img_tensor.mean()
        std_val = img_tensor.std()
        if std_val < 1e-5:
            std_val = 1.0
        return (img_tensor - mean_val) / std_val
    return f

def create_transform(
    image_size=(128, 128),
    normalization="zscore",
    augment=False,
    rotation_range=(0, 15),
    brightness=(0.95, 1.05),
    contrast=(0.95, 1.05),
    gamma_range=(0.9, 1.1)
):
    """
    Creates a torchvision transform pipeline with flexible resizing,
    optional mild augmentation, and normalization.
    """
    transform_list = [
        T.Resize(image_size, interpolation=InterpolationMode.LANCZOS),
        T.ToTensor()
    ]

    if augment:
        # random rotation
        transform_list.append(
            T.RandomApply(
                [T.RandomRotation(rotation_range, interpolation=InterpolationMode.BICUBIC)],
                p=0.5
            )
        )
        # mild brightness/contrast
        transform_list.append(
            T.ColorJitter(brightness=brightness, contrast=contrast)
        )
        # (optional) random gamma, etc.

    # Normalization
    if normalization == "zscore":
        transform_list.append(T.Lambda(_normalize_zscore()))
    elif normalization == "minmax":
        transform_list.append(T.Lambda(_normalize_minmax()))
    else:
        raise ValueError("Normalization must be 'zscore' or 'minmax'")

    return T.Compose(transform_list)
