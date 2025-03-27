import os
import random
import shutil
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

from src.data_utils import rotate_image_pil
from src.transforms_config import create_transform

def preview_augmentation(
    pil_image: Image.Image,
    rotate_deg: float = 15,
    do_rotate: bool = True,
    do_brightness_contrast: bool = True,
    do_gamma: bool = True,
    do_minmax: bool = False
):
    """
    For a single PIL image, apply random rotation, brightness/contrast,
    gamma, and optional min-max normalization, returning a torch.Tensor.
    Used purely for the AugmentationTab preview (GUI).
    """
    # Basic resizing for preview
    base_resize = T.Resize((224, 224), interpolation=InterpolationMode.LANCZOS)
    base_t = T.ToTensor()(base_resize(pil_image))
    temp_pil = T.ToPILImage()(base_t).convert('L')

    # Rotation
    if do_rotate:
        angle = random.uniform(-rotate_deg, rotate_deg)
        temp_pil = rotate_image_pil(temp_pil, angle=angle)

    # Brightness & contrast
    if do_brightness_contrast:
        color_jit = T.ColorJitter(
            brightness=(0.85, 1.15),
            contrast=(0.85, 1.15)
        )
        temp_pil = color_jit(temp_pil)

    # Gamma
    if do_gamma:
        g = random.uniform(0.85, 1.15)
        temp_pil = F.adjust_gamma(temp_pil, gamma=g, gain=1)

    out_tens = T.ToTensor()(temp_pil)

    # optional minmax
    if do_minmax:
        mn, mx = out_tens.min(), out_tens.max()
        if (mx - mn) > 1e-5:
            out_tens = (out_tens - mn)/(mx - mn)

    return out_tens

def build_bulk_transform(
    rotate_deg=15,
    do_rotate=True,
    do_brightness_contrast=True,
    do_gamma=True
):
    """
    Build a random transform pipeline for the bulk augmentation
    (similar to preview, but can be run multiple times).
    """
    # We'll replicate the same logic as preview
    # i.e., random rotation, brightness/contrast, gamma
    def augment_func(pil_img: Image.Image):
        # rotation
        temp = pil_img
        if do_rotate:
            angle = random.uniform(-rotate_deg, rotate_deg)
            temp = rotate_image_pil(temp, angle=angle)

        # brightness/contrast
        if do_brightness_contrast:
            color_jit = T.ColorJitter(
                brightness=(0.85, 1.15),
                contrast=(0.85, 1.15)
            )
            temp = color_jit(temp)

        # gamma
        if do_gamma:
            g = random.uniform(0.85, 1.15)
            temp = F.adjust_gamma(temp, gamma=g, gain=1)

        return temp

    return augment_func

def bulk_augment_dataset(
    src_root="cleaned_data",
    dst_root="augmented_data",
    copies_per_image=2,
    rotate_deg=15,
    do_rotate=True,
    do_brightness_contrast=True,
    do_gamma=True
):
    """
    Create 'augmented_data' from 'cleaned_data' by applying random
    transformations multiple times. For each image in src_root,
    we generate N copies in dst_root with random seeds.
    Overwrites if dst_root exists (partially).
    Returns distribution info (how many new images) and any error logs.
    """
    error_log = []
    total_new_images = 0

    if not os.path.isdir(src_root):
        raise ValueError(f"{src_root} does not exist to augment from.")

    # Copy entire structure of 'cleaned_data' => 'augmented_data' (only folder structure, not images)
    for subset in ["Training", "Testing"]:
        src_subset = os.path.join(src_root, subset)
        dst_subset = os.path.join(dst_root, subset)
        if os.path.isdir(src_subset):
            # ensure the folder structure
            os.makedirs(dst_subset, exist_ok=True)
            classes = [
                d for d in os.listdir(src_subset)
                if os.path.isdir(os.path.join(src_subset, d))
            ]
            for cls_ in classes:
                os.makedirs(os.path.join(dst_subset, cls_), exist_ok=True)

    # Build transform function
    augment_func = build_bulk_transform(
        rotate_deg=rotate_deg,
        do_rotate=do_rotate,
        do_brightness_contrast=do_brightness_contrast,
        do_gamma=do_gamma
    )

    # For each image in cleaned_data, create N copies in augmented_data
    for subset in ["Training", "Testing"]:
        src_subset = os.path.join(src_root, subset)
        dst_subset = os.path.join(dst_root, subset)
        if not os.path.isdir(src_subset):
            continue

        classes = [
            d for d in os.listdir(src_subset)
            if os.path.isdir(os.path.join(src_subset, d))
        ]

        for cls_ in classes:
            src_class_dir = os.path.join(src_subset, cls_)
            dst_class_dir = os.path.join(dst_subset, cls_)

            images = [
                f for f in os.listdir(src_class_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]

            for img_name in images:
                src_path = os.path.join(src_class_dir, img_name)
                # we do not copy the original image to augmented_data,
                # only new augmented copies
                try:
                    pil_img = Image.open(src_path).convert('L')
                except Exception as e:
                    error_log.append(f"[Error reading {src_path}]: {e}")
                    continue

                base_name, ext = os.path.splitext(img_name)
                # generate N augmented copies
                for i in range(copies_per_image):
                    try:
                        aug_pil = augment_func(pil_img)
                        out_name = f"{base_name}_aug{i}{ext}"
                        out_path = os.path.join(dst_class_dir, out_name)
                        aug_pil.save(out_path)
                        total_new_images += 1
                    except Exception as e:
                        error_log.append(f"[AugError {src_path} -> {out_path}]: {e}")

    # Return how many new images & any errors
    return total_new_images, error_log

