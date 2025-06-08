import os
import math
import random
import shutil
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
    Preview a single random augmentation on one image
    (used by the Preview tab).
    """
    base_resize = T.Resize((224, 224), interpolation=InterpolationMode.BILINEAR)
    base_t = T.ToTensor()(base_resize(pil_image))
    temp_pil = T.ToPILImage()(base_t).convert('L')

    if do_rotate:
        angle = random.uniform(-rotate_deg, rotate_deg)
        temp_pil = rotate_image_pil(temp_pil, angle=angle, resample=Image.BILINEAR)

    if do_flip:
        if random.random() < 0.5:
            temp_pil = temp_pil.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            temp_pil = temp_pil.transpose(Image.FLIP_TOP_BOTTOM)

    if do_perspective:
        distort = random.uniform(0.1, 0.4)
        perspective = T.RandomPerspective(distortion_scale=distort, p=1.0)
        temp_pil = perspective(temp_pil)

    if do_crop:
        cropper = T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0))
        temp_pil = cropper(temp_pil)

    if do_brightness_contrast:
        b_min, b_max = brightness_range
        c_min, c_max = contrast_range
        color_jit = T.ColorJitter(
            brightness=(b_min, b_max),
            contrast=(c_min, c_max)
        )
        temp_pil = color_jit(temp_pil)

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


def bulk_augment_dataset(
    src_root: str,
    dst_root: str,
    final_target: int,
    rotate_deg=15,
    do_rotate=True,
    do_brightness_contrast=True,
    do_gamma=True,
    do_flip=False,
    do_perspective=False,
    do_crop=False,
    brightness_range=(0.85, 1.15),
    contrast_range=(0.85, 1.15),
    gamma_range=(0.85, 1.15),
    progress_callback=None
):
    """
    Creates an augmented dataset in `dst_root`, ensuring for each
    class (under the "Training" or other subsets) we have at least `final_target` images.

    Steps:
      1) Copy all original images from src_root -> dst_root.
      2) Count how many each class has. If already >= final_target, skip augmentation.
      3) If < final_target, do a round-robin approach using the source images.
      4) Stop once you have exactly `final_target` for that class.
    """
    if not os.path.isdir(src_root):
        raise ValueError(f"Source folder '{src_root}' does not exist.")

    error_log = []
    total_new_images = 0

    subsets = []
    for name in os.listdir(src_root):
        s_path = os.path.join(src_root, name)
        if os.path.isdir(s_path):
            subsets.append(name)

    class_info = {}
    total_ops = 0

    for subset in subsets:
        src_subset = os.path.join(src_root, subset)
        dst_subset = os.path.join(dst_root, subset)
        os.makedirs(dst_subset, exist_ok=True)

        classes = [
            d for d in os.listdir(src_subset)
            if os.path.isdir(os.path.join(src_subset, d))
        ]
        for cls_ in classes:
            src_cls_dir = os.path.join(src_subset, cls_)
            dst_cls_dir = os.path.join(dst_subset, cls_)
            os.makedirs(dst_cls_dir, exist_ok=True)

            src_images = [
                f for f in os.listdir(src_cls_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            base_count = len(src_images)

            needed = max(0, final_target - base_count)

            class_key = f"{subset}/{cls_}"
            class_info[class_key] = {
                "src_images": src_images,
                "base_count": base_count,
                "needed": needed
            }
            total_ops += base_count  
            total_ops += needed    

    if total_ops == 0:
        if progress_callback:
            progress_callback(100, "No images found or no classes in source.")
        return 0, []

    done_ops = 0

    def update_progress(msg):
        nonlocal done_ops, total_ops, progress_callback
        done_ops += 1
        pct = int((done_ops / total_ops) * 100)
        if progress_callback:
            progress_callback(pct, msg)

    for class_key, info in class_info.items():
        subset_name, cls_name = class_key.split("/", 1)
        src_subset = os.path.join(src_root, subset_name, cls_name)
        dst_subset = os.path.join(dst_root, subset_name, cls_name)

        for img_name in info["src_images"]:
            src_path = os.path.join(src_subset, img_name)
            dst_path = os.path.join(dst_subset, img_name)
            try:
                shutil.copy2(src_path, dst_path)
                update_progress(f"Copy {class_key}")
            except Exception as e:
                err_msg = f"[CopyError] {src_path} => {str(e)}"
                error_log.append(err_msg)

    def build_augment_func():
        def augment_func(pil_img: Image.Image):
            temp = pil_img
            if do_rotate:
                angle = random.uniform(-rotate_deg, rotate_deg)
                temp = rotate_image_pil(temp, angle=angle, resample=Image.BILINEAR)

            if do_flip:
                if random.random() < 0.5:
                    temp = temp.transpose(Image.FLIP_LEFT_RIGHT)
                else:
                    temp = temp.transpose(Image.FLIP_TOP_BOTTOM)

            if do_perspective:
                distort = random.uniform(0.1, 0.4)
                persp = T.RandomPerspective(distortion_scale=distort, p=1.0)
                temp = persp(temp)

            if do_crop:
                w, h = temp.size
                cropper = T.RandomResizedCrop(size=(h, w), scale=(0.8, 1.0))
                temp = cropper(temp)

            if do_brightness_contrast:
                b_min, b_max = brightness_range
                c_min, c_max = contrast_range
                color_jit = T.ColorJitter(
                    brightness=(b_min, b_max),
                    contrast=(c_min, c_max)
                )
                temp = color_jit(temp)

            if do_gamma:
                g_min, g_max = gamma_range
                g_val = random.uniform(g_min, g_max)
                temp = F.adjust_gamma(temp, gamma=g_val, gain=1)

            return temp
        return augment_func

    augment_func = build_augment_func()

    for class_key, info in class_info.items():
        needed = info["needed"]
        if needed <= 0:
            continue

        subset_name, cls_name = class_key.split("/", 1)
        dst_subset = os.path.join(dst_root, subset_name, cls_name)
        src_subset = os.path.join(src_root, subset_name, cls_name)
        base_count = info["base_count"]
        src_images = info["src_images"]

        idx = 0
        for i in range(needed):
            src_name = src_images[idx]
            idx = (idx + 1) % base_count  
            src_path = os.path.join(src_subset, src_name)
            try:
                with Image.open(src_path).convert('L') as pil_img:
                    aug_pil = augment_func(pil_img)
                    base_name, ext = os.path.splitext(src_name)
                    out_name = f"{base_name}_aug{i}{ext}"
                    out_path = os.path.join(dst_subset, out_name)
                    aug_pil.save(out_path)
                    total_new_images += 1
                    update_progress(f"Augment {class_key}")
            except Exception as e:
                err_msg = f"[AugError] {src_path} => {str(e)}"
                error_log.append(err_msg)

    if progress_callback:
        progress_callback(100, "Augmentation done.")
    return total_new_images, error_log
