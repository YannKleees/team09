import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

def _normalize_minmax():
    """
    Min-max scale each image to [0..1].
    """
    def f(img_tensor):
        min_val = img_tensor.min()
        max_val = img_tensor.max()
        if max_val - min_val > 1e-5:
            img_tensor = (img_tensor - min_val) / (max_val - min_val)
        else:
            # If the image is basically constant, force all to 0
            img_tensor = img_tensor - min_val
        return img_tensor.clamp(0, 1)
    return f

def _normalize_zscore_rebound():
    """
    (1) Compute z-score (subtract mean, divide by std)
    (2) Then re-scale the result to [0..1] so that
        after standardizing, we don't have negative or >1 pixel values.

    NOTE: This is not a "pure" z-score in the final pixel domain
          because it is forcibly mapped back to [0..1].
          It *does* keep the shape of a distribution, but clipped
          and scaled into a [0..1] range for certain visualization or
          neural net convenience.
    """
    def f(img_tensor):
        mean_val = img_tensor.mean()
        std_val = img_tensor.std()
        if std_val < 1e-5:
            std_val = 1.0

        z = (img_tensor - mean_val) / std_val  # standard
        # Re-scale to [0..1]
        min_z = z.min()
        max_z = z.max()
        if max_z - min_z > 1e-5:
            z = (z - min_z) / (max_z - min_z)
        else:
            z = z - min_z
        return z.clamp(0, 1)
    return f

def _normalize_zscore_pure():
    """
    A "pure" z-score transform with no rebound to [0..1].
    This means pixel values could be negative or above 1.
    """
    def f(img_tensor):
        mean_val = img_tensor.mean()
        std_val = img_tensor.std()
        if std_val < 1e-5:
            std_val = 1.0
        z = (img_tensor - mean_val) / std_val
        return z
    return f

def create_transform(
    image_size=(128, 128),
    normalization="zscore",  # can be 'zscore', 'minmax', or None
    augment=False,
    rotation_range=(0, 15),
    brightness=(0.95, 1.05),
    contrast=(0.95, 1.05),
    pure_zscore=False
):
    """
    Create a torchvision transform pipeline:
      - Resizing
      - Optional augmentation
      - Normalization = 'zscore', 'minmax', or None
      - If 'pure_zscore=True' => no bounding to [0..1] after standardizing.
    """
    transform_list = [
        T.Resize(image_size, interpolation=InterpolationMode.LANCZOS),
        T.ToTensor()
    ]

    if augment:
        transform_list.append(
            T.RandomApply(
                [T.RandomRotation(rotation_range, interpolation=InterpolationMode.BICUBIC)],
                p=0.5
            )
        )
        transform_list.append(T.ColorJitter(brightness=brightness, contrast=contrast))

    # Handle normalization argument
    if normalization is None:
        # "No normalization" => do nothing more
        pass
    elif normalization == "zscore":
        if pure_zscore:
            transform_list.append(T.Lambda(_normalize_zscore_pure()))
        else:
            transform_list.append(T.Lambda(_normalize_zscore_rebound()))
    elif normalization == "minmax":
        transform_list.append(T.Lambda(_normalize_minmax()))
    else:
        raise ValueError("Normalization must be 'zscore', 'minmax', or None")

    return T.Compose(transform_list)
