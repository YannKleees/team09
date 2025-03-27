import os
import hashlib
from PIL import Image

def remove_duplicates_in_folder(folder_path):
    file_hashes = {}
    duplicates = []
    if not os.path.isdir(folder_path):
        return duplicates

    for fname in os.listdir(folder_path):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(folder_path, fname)
            with open(full_path, 'rb') as f:
                filehash = hashlib.md5(f.read()).hexdigest()
            if filehash in file_hashes:
                duplicates.append(full_path)
            else:
                file_hashes[filehash] = full_path
    return duplicates

def remove_duplicates_across_dataset(root_dir):
    all_duplicates = []
    if not os.path.isdir(root_dir):
        return []
    for subdir in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, subdir)
        if os.path.isdir(class_dir):
            dups = remove_duplicates_in_folder(class_dir)
            all_duplicates.extend(dups)
    return all_duplicates

def rotate_image_pil(pil_img, angle=0, resample=Image.BICUBIC):
    if angle == 0:
        return pil_img
    return pil_img.rotate(angle, resample=resample)
