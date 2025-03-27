import os
import hashlib

def compute_file_hash(file_path: str) -> str:
    """Compute MD5 hash of a file's content."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def remove_duplicates_in_folder(folder_path: str) -> list:
    """
    Identify duplicate images in a single folder by hashing.
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

def list_and_remove_duplicates(project_dir: str, label_folders: list):
    """
    Example usage from your snippet. Not used in the new pipeline,
    but kept here for reference.
    """
    import os

    total_removed = 0
    hash_dict = {}

    for data_type in ['Training', 'Testing']:
        for label in label_folders:
            folder_path = os.path.join(project_dir, 'data', 'raw', data_type, label)
            if not os.path.isdir(folder_path):
                continue
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(".jpg"):
                        full_file = os.path.join(root, file)
                        file_hash = compute_file_hash(full_file)
                        if file_hash not in hash_dict:
                            hash_dict[file_hash] = [full_file]
                        else:
                            hash_dict[file_hash].append(full_file)

    for hval, file_paths in hash_dict.items():
        if len(file_paths) > 1:
            for dup_path in file_paths[1:]:
                print(f"Removing duplicate (hash: {hval}) => {dup_path}")
                os.remove(dup_path)
                total_removed += 1

    print(f"Number of duplicates removed: {total_removed}")
