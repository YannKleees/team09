# logic/advanced_eda.py

import os
import random
import numpy as np
from PIL import Image
import cv2
from PIL import ImageOps

def gather_random_images(root_folder, subfolders, max_samples=200):
    """
    Collect up to 'max_samples' images from each specified subfolder,
    including top-level (directly under subset) AND sub-subfolders.

    Example: subfolders = ["Training", "Testing"].
    We do:
       - Look in data/archive/Training -> gather images in that folder
       - Also gather images in subfolders (e.g. "ClassA", "ClassB") if they exist
       - Do the same for data/archive/Testing
    Return a random subset of these images up to 'max_samples' total.
    """
    imgs = []
    for subset in subfolders:
        subset_path = os.path.join(root_folder, subset)
        if not os.path.isdir(subset_path):
            continue

        top_level_images = [
            os.path.join(subset_path, f)
            for f in os.listdir(subset_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        imgs.extend(top_level_images)

        for d in os.listdir(subset_path):
            dpath = os.path.join(subset_path, d)
            if os.path.isdir(dpath):
                sub_imgs = [
                    os.path.join(dpath, f)
                    for f in os.listdir(dpath)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
                imgs.extend(sub_imgs)

    random.shuffle(imgs)
    return imgs[:max_samples]

def gather_all_images_anywhere(root_folder, subfolders):
    """
    Similar to gather_random_images, but returns ALL images (no random limit).
    Great for building a 'gallery' of all images in Training/Testing.

    NOTE: Potentially large if your dataset is big. Use with caution or impose a limit.
    """
    imgs = []
    for subset in subfolders:
        subset_path = os.path.join(root_folder, subset)
        if not os.path.isdir(subset_path):
            continue

        for f in os.listdir(subset_path):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                imgs.append(os.path.join(subset_path, f))

        for d in os.listdir(subset_path):
            dpath = os.path.join(subset_path, d)
            if os.path.isdir(dpath):
                for fname in os.listdir(dpath):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        imgs.append(os.path.join(dpath, fname))
    return imgs


def collect_resolutions(image_paths):
    """
    Return lists of widths, heights, plus a count of read errors.
    """
    widths = []
    heights = []
    error_count = 0
    for path in image_paths:
        try:
            with Image.open(path) as im:
                w, h = im.size
                widths.append(w)
                heights.append(h)
        except:
            error_count += 1
    return widths, heights, error_count

def detect_poor_quality(image_paths):
    """
    Return a list of poor-quality image paths (basic checks):
      - extremely small dimension (<50px)
      - very low variance (<5.0)
    """
    poor_quality = []
    for path in image_paths:
        try:
            with Image.open(path) as im:
                w, h = im.size
                if w < 50 or h < 50:
                    poor_quality.append(path)
                    continue
                arr = np.array(im.convert('L'), dtype=np.float32)
                if arr.size > 0 and arr.std() < 5.0:
                    poor_quality.append(path)
        except:
            pass
    return poor_quality

def histogram_equalize(pil_img):
    """
    Now uses Adaptive Histogram Equalization (CLAHE) instead of global eq.
    """
    arr = np.array(pil_img, dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq_arr = clahe.apply(arr)
    out_pil = Image.fromarray(eq_arr)
    return out_pil

def detect_extended_outliers(
    image_paths,
    brightness_low=15.0,
    brightness_high=240.0,
    dimension_threshold=50,
    var_threshold=5.0
):
    """
    Extended outlier detection with dimension, variance, brightness checks.
    """
    extended_outliers = []
    for path in image_paths:
        try:
            with Image.open(path) as im:
                w, h = im.size
                arr = np.array(im.convert('L'), dtype=np.float32)
                if arr.size == 0:
                    extended_outliers.append(path)
                    continue

                mean_val = arr.mean()
                std_val = arr.std()

                if w < dimension_threshold or h < dimension_threshold:
                    extended_outliers.append(path)
                    continue

                if std_val < var_threshold:
                    extended_outliers.append(path)
                    continue

                if mean_val < brightness_low or mean_val > brightness_high:
                    extended_outliers.append(path)
                    continue

        except:
            extended_outliers.append(path)
    return extended_outliers

def gather_subset_resolutions(root_folder, subsets):
    """
    For each subset in subsets (e.g. ["Training","Testing"]),
    gather all images and return (subset_to_widths, subset_to_heights).
    """
    subset_to_widths = {}
    subset_to_heights = {}

    for sub in subsets:
        sub_path = os.path.join(root_folder, sub)
        if not os.path.isdir(sub_path):
            subset_to_widths[sub] = []
            subset_to_heights[sub] = []
            continue

        w_list = []
        h_list = []

        for fname in os.listdir(sub_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                fullp = os.path.join(sub_path, fname)
                try:
                    with Image.open(fullp) as im:
                        w, h = im.size
                        w_list.append(w)
                        h_list.append(h)
                except:
                    pass

        class_list = [
            d for d in os.listdir(sub_path)
            if os.path.isdir(os.path.join(sub_path, d))
        ]
        for c in class_list:
            cpath = os.path.join(sub_path, c)
            for fname in os.listdir(cpath):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    fullp = os.path.join(cpath, fname)
                    try:
                        with Image.open(fullp) as im:
                            w, h = im.size
                            w_list.append(w)
                            h_list.append(h)
                    except:
                        pass

        subset_to_widths[sub] = w_list
        subset_to_heights[sub] = h_list

    return subset_to_widths, subset_to_heights
