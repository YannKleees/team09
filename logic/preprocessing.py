import os
import shutil
from PIL import Image
from torchvision.transforms import ToPILImage

from logic.duplicates import remove_duplicates_across_dataset
from src.transforms_config import create_transform

def copy_dataset_structure(src_root, dst_root):
    """
    Copy 'Training' and 'Testing' subfolders (and all images)
    from src_root -> dst_root, preserving folder structure.
    """
    for subset in ["Training", "Testing"]:
        src_subset = os.path.join(src_root, subset)
        dst_subset = os.path.join(dst_root, subset)
        if os.path.isdir(src_subset):
            shutil.copytree(src_subset, dst_subset, dirs_exist_ok=True)

def remove_duplicates_in_cleaned(dst_root):
    """
    Remove duplicates from the 'Training' and 'Testing' subfolders in dst_root.
    """
    for subset in ["Training", "Testing"]:
        subpath = os.path.join(dst_root, subset)
        if os.path.isdir(subpath):
            dups = remove_duplicates_across_dataset(subpath)
            for dup in dups:
                try:
                    os.remove(dup)
                except:
                    pass

def resize_and_normalize_in_place(dst_root, resize_val, norm_method, error_log):
    """
    Resize + normalize in place all images under 
    dst_root/Training and dst_root/Testing.
    Also record any error messages in error_log list.
    """
    transform = create_transform(
        image_size=(resize_val, resize_val),
        normalization=norm_method,
        augment=False  # no augmentation in the final cleaned set
    )

    total_processed = 0
    for subset in ["Training", "Testing"]:
        subset_path = os.path.join(dst_root, subset)
        if not os.path.isdir(subset_path):
            continue
        classes = [
            d for d in os.listdir(subset_path)
            if os.path.isdir(os.path.join(subset_path, d))
        ]
        for cls in classes:
            cls_path = os.path.join(subset_path, cls)
            images = [
                f for f in os.listdir(cls_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            for img_name in images:
                img_path = os.path.join(cls_path, img_name)
                try:
                    pil_img = Image.open(img_path).convert('L')
                    tens = transform(pil_img)
                    out_pil = ToPILImage()(tens).convert('L')
                    out_pil.save(img_path)  # overwrite in place
                    total_processed += 1
                except Exception as e:
                    msg = f"[Error: {img_path}] {e}"
                    error_log.append(msg)

    return total_processed

def count_images_in_folders(root_folder):
    """
    Return a dict {subset: {class_name: count, ...}, 'total': overall_count}
    for quick distribution logging.
    """
    result = {"Training": {}, "Testing": {}, "total": 0}
    total_count = 0
    for subset in ["Training", "Testing"]:
        subpath = os.path.join(root_folder, subset)
        if not os.path.isdir(subpath):
            result[subset] = {}
            continue
        subset_count = 0
        classes = [
            d for d in os.listdir(subpath)
            if os.path.isdir(os.path.join(subpath, d))
        ]
        for c in classes:
            cpath = os.path.join(subpath, c)
            n = sum(
                f.lower().endswith(('.png', '.jpg', '.jpeg'))
                for f in os.listdir(cpath)
            )
            subset_count += n
            result[subset][c] = n
        total_count += subset_count
    result["total"] = total_count
    return result

def preprocess_entire_dataset(
    root_dir="data/archive",
    dst_root="cleaned_data",
    resize_val=128,
    norm_method="zscore",
    delete_duplicates=False
):
    """
    Higher-level function to:
      1) Copy 'Training' & 'Testing' from root_dir -> dst_root (so raw data is untouched).
      2) (Optional) Remove duplicates inside the new dst_root.
      3) Resize + normalize everything in place.
      4) Return distribution info and error logs for convenience.
    """
    error_log = []

    # 1) Copy from data/archive -> cleaned_data
    if not os.path.isdir(root_dir):
        raise ValueError(f"{root_dir} is not a valid directory containing 'Training'/'Testing'.")

    if not os.path.exists(dst_root):
        os.makedirs(dst_root, exist_ok=True)
    # if it already exists, partial overwrite is done
    copy_dataset_structure(root_dir, dst_root)

    # 2) Possibly remove duplicates
    if delete_duplicates:
        remove_duplicates_in_cleaned(dst_root)

    # 3) Resize + normalize
    total_processed = resize_and_normalize_in_place(dst_root, resize_val, norm_method, error_log)

    # 4) Distribution
    dist_info = count_images_in_folders(dst_root)
    return dist_info, error_log, total_processed
