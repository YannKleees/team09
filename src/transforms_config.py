# File: src/transforms_config.py

import torchvision.transforms as T

def create_transform(
    image_size=(128,128),
    normalization="zscore",
    augment=False,
    adv_aug=False,
    strong_aug=False,
    rotation_deg=0,            # [ADDED]
    brightness_range=(1.0,1.0) # [ADDED]
):
    """
    Build a transform pipeline with optional rotation_deg & brightness_range.
    """
    transform_list = []

    if not adv_aug:
        transform_list.append(T.Resize(image_size))

    if augment:
        if rotation_deg > 0:
            transform_list.append(T.RandomRotation(degrees=rotation_deg))
        # apply brightness_range if it's not (1.0,1.0)
        if brightness_range != (1.0,1.0):
            transform_list.append(T.ColorJitter(
                brightness=brightness_range,
                contrast=brightness_range
            ))

    if adv_aug:
        transform_list.append(T.RandomResizedCrop(size=image_size, scale=(0.8,1.0)))

    if strong_aug:
        transform_list.append(T.RandomPerspective(distortion_scale=0.2, p=0.5))

    transform_list.append(T.ToTensor())

    if normalization == "zscore":
        transform_list.append(T.Normalize(mean=[0.5], std=[0.5]))
    elif normalization == "minmax":
        transform_list.append(T.Normalize(mean=[0.5], std=[0.5]))
    # else no normalization if "none"

    return T.Compose(transform_list)
