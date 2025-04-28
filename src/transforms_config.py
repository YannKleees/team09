# File: src/transforms_config.py

import torchvision.transforms as T

def create_transform(
    image_size=(128,128),
    augment=False,
    adv_aug=False,
    strong_aug=False,
    rotation_deg=0,
    brightness_range=(1.0,1.0)
):
    """
    Build a transform pipeline WITHOUT any intensity normalization.
    We only do possible resizing / data augmentation.
    """
    transform_list = []

    if not adv_aug:
        transform_list.append(T.Resize(image_size))

    if augment:
        if rotation_deg > 0:
            transform_list.append(T.RandomRotation(degrees=rotation_deg))
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

    return T.Compose(transform_list)
