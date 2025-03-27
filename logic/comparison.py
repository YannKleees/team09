import os
from PIL import Image

def count_images_in_classes(folder: str) -> dict:
    """
    Return a dict {class_name: number_of_images}
    for each subfolder in 'folder'.
    """
    if not os.path.isdir(folder):
        return {}
    result = {}
    for cls in os.listdir(folder):
        cpath = os.path.join(folder, cls)
        if os.path.isdir(cpath):
            n = sum(
                f.lower().endswith(('.png','.jpg','.jpeg'))
                for f in os.listdir(cpath)
            )
            result[cls] = n
    return result

def find_image_in_subfolders(root_dir: str, image_name: str) -> str:
    """
    Search each subfolder of root_dir for image_name.
    Return the full path if found, else None.
    """
    if not os.path.isdir(root_dir):
        return None

    for subdir in os.listdir(root_dir):
        cpath = os.path.join(root_dir, subdir)
        if os.path.isdir(cpath):
            candidate = os.path.join(cpath, image_name)
            if os.path.isfile(candidate):
                return candidate
    return None

def load_image_as_grayscale(path: str) -> Image.Image:
    """
    Open an image at path as grayscale PIL image. 
    Returns None if fails.
    """
    try:
        return Image.open(path).convert('L')
    except:
        return None
