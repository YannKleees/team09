"""
File: src/transforms_config.py

Provides a function `create_transform` that constructs a PyTorch
transformation pipeline for grayscale MRI images.
"""

import torchvision.transforms as T

def create_transform(
    image_size=(128, 128),
    augment=False,
    adv_aug=False,
    strong_aug=False,
    rotation_deg=0,
    brightness_range=(1.0, 1.0)
):
    """
    Build a transform pipeline (no explicit intensity normalization here).

    Args:
        image_size (tuple): (width, height) to resize/crop.
        augment (bool): If True, apply mild data augmentation (rotation, brightness).
        adv_aug (bool): If True, apply advanced augmentation like RandomResizedCrop.
        strong_aug (bool): If True, apply extra transforms like RandomPerspective.
        rotation_deg (float): Max degrees for RandomRotation if augment=True.
        brightness_range (tuple): Brightness/contrast jitter range if augment=True.

    Returns:
        torchvision.transforms.Compose: A composed transform to apply to images.
    """
    transform_list = []

    # If we're not using advanced augmentation, do a direct Resize
    if not adv_aug:
        transform_list.append(T.Resize(image_size))

    # Mild augmentations
    if augment:
        if rotation_deg > 0:
            transform_list.append(T.RandomRotation(degrees=rotation_deg))
        if brightness_range != (1.0, 1.0):
            transform_list.append(
                T.ColorJitter(
                    brightness=brightness_range,
                    contrast=brightness_range
                )
            )

    # If adv_aug is True, we do a RandomResizedCrop
    if adv_aug:
        transform_list.append(T.RandomResizedCrop(size=image_size, scale=(0.8, 1.0)))

    # If strong_aug is True, add perspective warp
    if strong_aug:
        transform_list.append(T.RandomPerspective(distortion_scale=0.2, p=0.5))

    # Finally convert to tensor
    transform_list.append(T.ToTensor())

    return T.Compose(transform_list)
