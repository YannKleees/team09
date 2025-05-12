import os
import hashlib
from PIL import Image


def rotate_image_pil(pil_img, angle=0, resample=Image.BILINEAR):
    if angle == 0:
        return pil_img
    return pil_img.rotate(angle, resample=resample)


def compute_file_hash(file_path: str) -> str:
    """Compute MD5 hash of a file's content."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def remove_duplicates_in_folder(folder_path: str) -> list:
    """
    OLD function: remove duplicates only if they are in the SAME folder.
    (We keep it in case other code calls it.)
    """
    file_hashes = {}
    duplicates = []
    if not os.path.isdir(folder_path):
        return duplicates

    for fname in os.listdir(folder_path):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(folder_path, fname)
            file_hash = compute_file_hash(full_path)
            if file_hash in file_hashes:
                duplicates.append(full_path)
            else:
                file_hashes[file_hash] = full_path
    return duplicates


def remove_duplicates_across_dataset(root_dir: str) -> list:
    """
    OLD function: check duplicates for each immediate subfolder in root_dir.
    But it does NOT go deeper into sub-subfolders.
    """
    all_duplicates = []
    if not os.path.isdir(root_dir):
        return all_duplicates

    for subdir in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, subdir)
        if os.path.isdir(class_dir):
            dups = remove_duplicates_in_folder(class_dir)
            all_duplicates.extend(dups)

    return all_duplicates


def remove_duplicates_recursively(root_dir: str) -> list:
    """
    NEW function: Recursively find duplicates in ALL subfolders under `root_dir`.
    We build one dictionary keyed by MD5 hash so duplicates are recognized
    across sub-subfolders as well.
    Returns a list of file paths that are duplicates (excluding the first occurrence).
    """
    if not os.path.isdir(root_dir):
        return []

    # 1) Collect all .png/.jpg/.jpeg under root_dir (any depth).
    all_files = []
    for dirpath, dirs, files in os.walk(root_dir):
        for fname in files:
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(dirpath, fname)
                all_files.append(full_path)

    # 2) Compute hashes and detect duplicates
    hash_dict = {}
    duplicates = []
    for fpath in all_files:
        hval = compute_file_hash(fpath)
        if hval in hash_dict:
            duplicates.append(fpath)
        else:
            hash_dict[hval] = fpath

    return duplicates
