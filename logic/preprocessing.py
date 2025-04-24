# File: logic/preprocessing.py

import os
import time
import shutil
import numpy as np
from PIL import Image

from logic.duplicates import remove_duplicates_across_dataset
# (REMOVED the "from logic.preprocessing import preprocess_entire_dataset" line to avoid circular import)


def preprocess_entire_dataset(
    root_dir="data/archive",
    dst_root="cleaned_data",
    resize_val=128,
    norm_method=None,
    delete_duplicates=False
):
    """
    Copy images from 'root_dir' (with subfolders e.g. Training/Testing) into 'dst_root',
    optionally remove duplicates, resize, and normalize.

    Returns:
        dist_info (dict): distribution info { subset: {class: count} }
        error_log (list): any errors
        total_processed (int): how many images we processed
        dups_removed (int): how many duplicates removed
        elapsed_sec (float): time taken in seconds
    """
    t0 = time.time()
    error_log = []
    dist_info = {}
    total_processed = 0
    dups_removed = 0

    if not os.path.isdir(root_dir):
        raise ValueError(f"{root_dir} does not exist.")

    # 1) Copy structure
    # We'll look for 'Training' and 'Testing' subfolders. If your dataset has different naming,
    # adjust accordingly.
    subsets = ["Training", "Testing"]
    os.makedirs(dst_root, exist_ok=True)

    for subset in subsets:
        src_subset = os.path.join(root_dir, subset)
        dst_subset = os.path.join(dst_root, subset)
        if os.path.isdir(src_subset):
            shutil.rmtree(dst_subset, ignore_errors=True)  # clear any old data
            os.makedirs(dst_subset, exist_ok=True)

            classes = [
                d for d in os.listdir(src_subset)
                if os.path.isdir(os.path.join(src_subset, d))
            ]
            for cls_ in classes:
                os.makedirs(os.path.join(dst_subset, cls_), exist_ok=True)

            # Copy images
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
                    except Exception as e:
                        error_log.append(f"CopyError: {src_path} => {e}")

    # 2) Optionally remove duplicates
    if delete_duplicates:
        dups_removed = remove_duplicates_across_dataset(dst_root)

    # 3) Resize & normalize
    total_processed, err_list = _resize_and_normalize_images(dst_root, resize_val, norm_method)
    error_log.extend(err_list)

    # 4) Summarize distribution
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
    return dist_info, error_log, total_processed, len(dups_removed), elapsed_sec


def _resize_and_normalize_images(folder, resize_val, norm_method):
    """
    Internal helper to loop over all images and apply resizing + optional normalization.
    """
    errors = []
    processed_count = 0

    for root, dirs, files in os.walk(folder):
        for fname in files:
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                fpath = os.path.join(root, fname)
                try:
                    with Image.open(fpath).convert('L') as im:
                        # 1) Resize
                        im = im.resize((resize_val, resize_val), Image.BILINEAR)

                        # 2) Normalize if requested
                        if norm_method == "zscore":
                            arr = np.array(im, dtype=np.float32)
                            mean_val = arr.mean()
                            std_val = arr.std()
                            if std_val > 1e-5:
                                arr = (arr - mean_val) / std_val
                            # Convert back to [0..255] for saving
                            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
                            im = Image.fromarray(arr.astype(np.uint8))
                        elif norm_method == "minmax":
                            arr = np.array(im, dtype=np.float32)
                            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
                            im = Image.fromarray(arr.astype(np.uint8))
                        # else norm_method=None => do nothing

                        # Finally save back
                        im.save(fpath)
                        processed_count += 1

                except Exception as e:
                    errors.append(f"ProcessingError: {fpath} => {e}")

    return processed_count, errors


# Below is an example “extended” function that also does histogram eq:
def preprocess_entire_dataset_with_eq(
    root_dir="data/archive",
    dst_root="cleaned_data",
    resize_val=128,
    norm_method=None,
    delete_duplicates=False,
    patient_split=False,
    apply_hist_eq=False
):
    """
    A variant that optionally does histogram eq before resizing,
    and can do "patient-level" splitting if needed.

    This is used by `PreprocessingWorker` in the new tab if you want it.
    """
    # For simplicity, we’ll just do the same structure, then
    # apply eq if `apply_hist_eq` is True inside _resize_and_normalize_images_with_eq().
    t0 = time.time()
    error_log = []
    dist_info = {}
    total_processed = 0

    if not os.path.isdir(root_dir):
        raise ValueError(f"{root_dir} does not exist.")

    # If you’re using “Validation” / “Test” subfolders, adapt here:
    subsets = ["Validation", "Test"]
    os.makedirs(dst_root, exist_ok=True)

    for subset in subsets:
        src_subset = os.path.join(root_dir, subset)
        dst_subset = os.path.join(dst_root, subset)
        if os.path.isdir(src_subset):
            shutil.rmtree(dst_subset, ignore_errors=True)
            os.makedirs(dst_subset, exist_ok=True)

            classes = [
                d for d in os.listdir(src_subset)
                if os.path.isdir(os.path.join(src_subset, d))
            ]
            for cls_ in classes:
                os.makedirs(os.path.join(dst_subset, cls_), exist_ok=True)

            # Copy images
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
                    except Exception as e:
                        error_log.append(f"CopyError: {src_path} => {e}")

    # optionally remove duplicates
    if delete_duplicates:
        from logic.duplicates import remove_duplicates_across_dataset
        duplicates = remove_duplicates_across_dataset(dst_root)
        # If you want them actually removed, you'd do:
        for dup_file in duplicates:
            try:
                os.remove(dup_file)
            except OSError:
                pass

    # Do the resizing & eq
    total_processed, err_list = _resize_and_normalize_images_with_eq(
        dst_root, resize_val, norm_method, apply_hist_eq
    )
    error_log.extend(err_list)

    # Summarize distribution
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

    # If patient_split is True, you’d incorporate your custom logic here...

    elapsed_sec = time.time() - t0
    return dist_info, error_log, total_processed


def _resize_and_normalize_images_with_eq(folder, resize_val, norm_method, apply_hist_eq):
    """
    Similar to _resize_and_normalize_images, but applies histogram eq if requested.
    """
    import numpy as np
    from PIL import Image, ImageOps

    errors = []
    processed_count = 0

    for root, dirs, files in os.walk(folder):
        for fname in files:
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                fpath = os.path.join(root, fname)
                try:
                    with Image.open(fpath).convert('L') as im:
                        if apply_hist_eq:
                            im = ImageOps.equalize(im)

                        # 1) Resize
                        im = im.resize((resize_val, resize_val), Image.BILINEAR)

                        # 2) Normalize if requested
                        if norm_method == "zscore":
                            arr = np.array(im, dtype=np.float32)
                            mean_val = arr.mean()
                            std_val = arr.std()
                            if std_val > 1e-5:
                                arr = (arr - mean_val) / std_val
                            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
                            im = Image.fromarray(arr.astype(np.uint8))
                        elif norm_method == "minmax":
                            arr = np.array(im, dtype=np.float32)
                            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
                            im = Image.fromarray(arr.astype(np.uint8))
                        # else None => do nothing

                        im.save(fpath)
                        processed_count += 1

                except Exception as e:
                    errors.append(f"ProcessingError: {fpath} => {e}")

    return processed_count, errors
