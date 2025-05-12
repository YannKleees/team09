import os
import time
import shutil
import numpy as np
from PIL import Image, ImageFilter
import cv2  # for CLAHE-based histogram eq

# IMPORT the new remove_duplicates_recursively function
from logic.duplicates import remove_duplicates_recursively

def _adaptive_histogram_equalize(pil_img):
    """
    Applies OpenCV's CLAHE to a PIL grayscale image.
    """
    arr = np.array(pil_img, dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq_arr = clahe.apply(arr)
    return Image.fromarray(eq_arr)

def _get_pil_interpolation(mode_str):
    """
    Convert a string like "BILINEAR"|"BICUBIC"|"LANCZOS"
    into the corresponding PIL interpolation constant.
    Defaults to Image.BILINEAR if unknown.
    """
    mode_str = mode_str.upper()
    if mode_str == "BICUBIC":
        return Image.BICUBIC
    elif mode_str == "LANCZOS":
        return Image.LANCZOS
    else:
        return Image.BILINEAR

def preprocess_entire_dataset(
    root_dir,
    dst_root,
    resize_val=128,
    delete_duplicates=False,
    apply_hist_eq=False,
    filter_mode="none",
    interpolation="BILINEAR",
    progress_fn=None
):
    """
    Copy images from 'root_dir' (with subfolders e.g. Training/Testing) into 'dst_root'.
    Then optionally:
      - remove duplicates (MD5 hash check) [RECURSIVE]
      - apply CLAHE-based histogram eq
      - apply blur/sharpen filters
      - resize with chosen interpolation

    progress_fn: optional callback(current_idx, total_count).

    Returns:
        dist_info (dict): distribution info { subset: {class: count} }
        error_log (list): any errors
        total_processed (int): how many images were processed
        dups_removed (int): how many duplicates removed
        elapsed_sec (float): time taken in seconds
    """
    t0 = time.time()
    error_log = []
    dist_info = {}
    total_processed = 0
    dups_removed = 0

    if not os.path.isdir(root_dir):
        raise ValueError(f"Source '{root_dir}' does not exist.")

    subsets = ["Training", "Testing"]
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root, ignore_errors=True)
    os.makedirs(dst_root, exist_ok=True)

    # Step 1: Copy structure
    for subset in subsets:
        src_subset = os.path.join(root_dir, subset)
        dst_subset = os.path.join(dst_root, subset)
        if os.path.isdir(src_subset):
            os.makedirs(dst_subset, exist_ok=True)

            classes = [
                d for d in os.listdir(src_subset)
                if os.path.isdir(os.path.join(src_subset, d))
            ]
            for cls_ in classes:
                os.makedirs(os.path.join(dst_subset, cls_), exist_ok=True)

            images_copied = 0
            for cls_ in classes:
                src_class_dir = os.path.join(src_subset, cls_)
                dst_class_dir = os.path.join(dst_subset, cls_)
                images = [
                    f for f in os.listdir(src_class_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
                for img_name in images:
                    src_path = os.path.join(src_class_dir, img_name)
                    dst_path = os.path.join(dst_class_dir, img_name)
                    try:
                        shutil.copy2(src_path, dst_path)
                        images_copied += 1
                    except Exception as e:
                        error_log.append(f"CopyError: {src_path} => {e}")

    # Step 2: Remove duplicates *recursively* if requested
    if delete_duplicates:
        duplicates = remove_duplicates_recursively(dst_root)
        dups_removed = len(duplicates)
        for dup_file in duplicates:
            try:
                os.remove(dup_file)
            except OSError:
                pass

    # Step 3: Process (hist eq, filter, resize) for all images
    all_image_paths = []
    for subset in subsets:
        subset_path = os.path.join(dst_root, subset)
        if os.path.isdir(subset_path):
            classes = [
                d for d in os.listdir(subset_path)
                if os.path.isdir(os.path.join(subset_path, d))
            ]
            for cls_ in classes:
                cpath = os.path.join(subset_path, cls_)
                files = [
                    f for f in os.listdir(cpath)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
                for fname in files:
                    fullpath = os.path.join(cpath, fname)
                    all_image_paths.append(fullpath)

    interp_mode = _get_pil_interpolation(interpolation)
    n_images = len(all_image_paths)

    def apply_filter(pil_im, mode):
        if mode == "blur":
            return pil_im.filter(ImageFilter.BLUR)
        elif mode == "sharpen":
            return pil_im.filter(ImageFilter.SHARPEN)
        return pil_im

    idx = 0
    for ipath in all_image_paths:
        idx += 1
        if progress_fn:
            progress_fn(idx, n_images)

        try:
            with Image.open(ipath).convert('L') as im:
                # hist eq if requested
                if apply_hist_eq:
                    im = _adaptive_histogram_equalize(im)
                # blur/sharpen if requested
                im = apply_filter(im, filter_mode.lower())
                # resize
                im = im.resize((resize_val, resize_val), interp_mode)
                # save
                im.save(ipath)
                total_processed += 1

        except Exception as e:
            error_log.append(f"ProcessingError: {ipath} => {e}")

    # Step 4: Summarize distribution
    for subset in subsets:
        subset_path = os.path.join(dst_root, subset)
        if os.path.isdir(subset_path):
            cls_dict = {}
            for cls_ in os.listdir(subset_path):
                cpath = os.path.join(subset_path, cls_)
                if os.path.isdir(cpath):
                    n = sum(
                        f.lower().endswith(('.png', '.jpg', '.jpeg'))
                        for f in os.listdir(cpath)
                    )
                    cls_dict[cls_] = n
            dist_info[subset] = cls_dict

    elapsed_sec = time.time() - t0
    return dist_info, error_log, total_processed, dups_removed, elapsed_sec
