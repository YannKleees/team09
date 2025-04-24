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
    Identify duplicate images in a folder by hashing.
    Returns a list of file paths that are duplicates.
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
    Recursively check duplicates for each subfolder (class folder) in root_dir.
    Returns a list of full paths to duplicate files.
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
