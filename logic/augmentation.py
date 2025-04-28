# File: logic/augmentation.py

import os
import random
import math
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from logic.duplicates import rotate_image_pil

def preview_augmentation(
    pil_image: Image.Image,
    rotate_deg: float = 15,
    do_rotate: bool = True,
    do_brightness_contrast: bool = True,
    do_gamma: bool = True,
    do_minmax: bool = False,
    do_flip: bool = False,
    do_perspective: bool = False,
    do_crop: bool = False,
    brightness_range=(0.85, 1.15),
    contrast_range=(0.85, 1.15),
    gamma_range=(0.85, 1.15)
):
    """
    Apply random transformations to a single PIL image for preview.
    ...
    (unchanged from your advanced version)
    """

    base_resize = T.Resize((224, 224), interpolation=InterpolationMode.BILINEAR)
    base_t = T.ToTensor()(base_resize(pil_image))
    temp_pil = T.ToPILImage()(base_t).convert('L')

    # Rotation
    if do_rotate:
        angle = random.uniform(-rotate_deg, rotate_deg)
        temp_pil = rotate_image_pil(temp_pil, angle=angle, resample=Image.BILINEAR)

    # Flip
    if do_flip:
        if random.random() < 0.5:
            temp_pil = temp_pil.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            temp_pil = temp_pil.transpose(Image.FLIP_TOP_BOTTOM)

    # Perspective
    if do_perspective:
        distort = random.uniform(0.1, 0.4)
        perspective = T.RandomPerspective(distortion_scale=distort, p=1.0)
        temp_pil = perspective(temp_pil)

    # Crop
    if do_crop:
        cropper = T.RandomResizedCrop(size=(224,224), scale=(0.8,1.0))
        temp_pil = cropper(temp_pil)

    # Brightness/Contrast
    if do_brightness_contrast:
        b_min, b_max = brightness_range
        c_min, c_max = contrast_range
        color_jit = T.ColorJitter(brightness=(b_min, b_max),
                                  contrast=(c_min, c_max))
        temp_pil = color_jit(temp_pil)

    # Gamma
    if do_gamma:
        g_min, g_max = gamma_range
        g = random.uniform(g_min, g_max)
        temp_pil = F.adjust_gamma(temp_pil, gamma=g, gain=1)

    out_tens = T.ToTensor()(temp_pil)

    if do_minmax:
        mn, mx = out_tens.min(), out_tens.max()
        if (mx - mn) > 1e-5:
            out_tens = (out_tens - mn) / (mx - mn)

    return out_tens


def build_bulk_transform(
    rotate_deg=15,
    do_rotate=True,
    do_brightness_contrast=True,
    do_gamma=True,
    do_flip=False,
    do_perspective=False,
    do_crop=False,
    brightness_range=(0.85, 1.15),
    contrast_range=(0.85, 1.15),
    gamma_range=(0.85, 1.15)
):
    """
    ...
    (unchanged advanced version)
    """
    def augment_func(pil_img: Image.Image):
        temp = pil_img

        # rotate
        if do_rotate:
            angle = random.uniform(-rotate_deg, rotate_deg)
            temp = rotate_image_pil(temp, angle=angle)

        # flip
        if do_flip:
            if random.random() < 0.5:
                temp = temp.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                temp = temp.transpose(Image.FLIP_TOP_BOTTOM)

        # perspective
        if do_perspective:
            distort = random.uniform(0.1, 0.4)
            perspective = T.RandomPerspective(distortion_scale=distort, p=1.0)
            temp = perspective(temp)

        # crop
        if do_crop:
            cropper = T.RandomResizedCrop(size=temp.size, scale=(0.8,1.0))
            temp = cropper(temp)

        # brightness & contrast
        if do_brightness_contrast:
            b_min, b_max = brightness_range
            c_min, c_max = contrast_range
            color_jit = T.ColorJitter(brightness=(b_min,b_max),
                                      contrast=(c_min,c_max))
            temp = color_jit(temp)

        # gamma
        if do_gamma:
            g_min, g_max = gamma_range
            g = random.uniform(g_min, g_max)
            temp = F.adjust_gamma(temp, gamma=g, gain=1)

        return temp

    return augment_func


def _balance_class_counts(class_counts):
    """
    Return how many augmentations each class needs to match the largest class.
    """
    max_count = max(class_counts.values())
    needed = {}
    for c, cnt in class_counts.items():
        needed[c] = max_count - cnt if cnt < max_count else 0
    return needed


def bulk_augment_dataset(
    src_root="cleaned_data",
    dst_root="augmented_data",
    copies_per_image=2,
    rotate_deg=15,
    do_rotate=True,
    do_brightness_contrast=True,
    do_gamma=True,
    balance_classes=True,  # <-- FORCED True
    do_flip=False,
    do_perspective=False,
    do_crop=False,
    brightness_range=(0.85,1.15),
    contrast_range=(0.85,1.15),
    gamma_range=(0.85,1.15),
    progress_callback=None
):
    """
    We now always do balance_classes=True, ignoring user input.
    ...
    """
    error_log = []
    total_new_images = 0

    if not os.path.isdir(src_root):
        raise ValueError(f"{src_root} does not exist.")

    # 1) Make folder structure
    for subset in ["Training", "Testing"]:
        src_subset = os.path.join(src_root, subset)
        dst_subset = os.path.join(dst_root, subset)
        if os.path.isdir(src_subset):
            os.makedirs(dst_subset, exist_ok=True)
            classes = [
                d for d in os.listdir(src_subset)
                if os.path.isdir(os.path.join(src_subset, d))
            ]
            for cls_ in classes:
                os.makedirs(os.path.join(dst_subset, cls_), exist_ok=True)

    # 2) Build random transform
    augment_func = build_bulk_transform(
        rotate_deg=rotate_deg,
        do_rotate=do_rotate,
        do_brightness_contrast=do_brightness_contrast,
        do_gamma=do_gamma,
        do_flip=do_flip,
        do_perspective=do_perspective,
        do_crop=do_crop,
        brightness_range=brightness_range,
        contrast_range=contrast_range,
        gamma_range=gamma_range
    )

    # 3) Always do balancing => figure out how many we must produce per class
    class_counts = {}
    for subset in ["Training", "Testing"]:
        src_subset = os.path.join(src_root, subset)
        if not os.path.isdir(src_subset):
            continue
        classes = [
            d for d in os.listdir(src_subset)
            if os.path.isdir(os.path.join(src_subset, d))
        ]
        for cls_ in classes:
            src_class_dir = os.path.join(src_subset, cls_)
            images = [
                f for f in os.listdir(src_class_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            count_here = len(images)
            key = f"{subset}/{cls_}"
            class_counts[key] = class_counts.get(key, 0) + count_here

    needed_dict = {}
    if class_counts:
        needed_dict = _balance_class_counts(class_counts)

    # 4) Actually do augmentation
    all_image_paths = []
    for subset in ["Training", "Testing"]:
        src_subset = os.path.join(src_root, subset)
        if not os.path.isdir(src_subset):
            continue
        classes = [
            d for d in os.listdir(src_subset)
            if os.path.isdir(os.path.join(src_subset, d))
        ]
        for cls_ in classes:
            src_class_dir = os.path.join(src_subset, cls_)
            images = [
                f for f in os.listdir(src_class_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            for img_name in images:
                full_path = os.path.join(src_class_dir, img_name)
                all_image_paths.append((subset, cls_, full_path))

    total_images = len(all_image_paths)
    for idx, (subset, cls_, full_path) in enumerate(all_image_paths):
        if progress_callback:
            pval = int((idx / total_images) * 100)
            msg = f"Processing {idx+1}/{total_images} ({subset}/{cls_})"
            progress_callback(pval, msg)

        try:
            pil_img = Image.open(full_path).convert('L')
        except Exception as e:
            error_log.append(f"[Error reading {full_path}]: {e}")
            continue

        class_key = f"{subset}/{cls_}"
        extra_needed = needed_dict.get(class_key, 0)

        copies_for_this_image = copies_per_image
        base_count = class_counts[class_key]
        if base_count > 0 and extra_needed > 0:
            # spread that across all images
            per_image = math.ceil(extra_needed / base_count)
            copies_for_this_image += per_image

        subset_dst = os.path.join(dst_root, subset)
        dst_class_dir = os.path.join(subset_dst, cls_)

        base_name, ext = os.path.splitext(os.path.basename(full_path))
        for i in range(copies_for_this_image):
            try:
                aug_pil = augment_func(pil_img)
                out_name = f"{base_name}_aug{i}{ext}"
                out_path = os.path.join(dst_class_dir, out_name)
                aug_pil.save(out_path)
                total_new_images += 1
            except Exception as e:
                error_log.append(f"[AugError {full_path} -> {out_path}]: {e}")

    if progress_callback:
        progress_callback(100, "Augmentation completed.")

    return total_new_images, error_log
