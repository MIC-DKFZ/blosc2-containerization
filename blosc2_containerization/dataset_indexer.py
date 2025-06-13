import os
from tqdm import tqdm
from tqdmp import tqdmp
import json
from pathlib import Path
from bloscio import Blosc2IO
import argparse


def get_image_filepaths(load_dir, save_dir, data_dir):
    print("Indexing Blosc2 files...", flush=True)
    image_filepaths = get_leaf_parts_of_b2nd_files(load_dir / data_dir)
    print("Indexed Blosc2 files.", flush=True)

    print("Saving image_filepaths.json...", flush=True)
    with open(save_dir / "image_filepaths.json", "w", encoding="utf-8") as f: 
        json.dump(image_filepaths, f, indent=4)
    print("Saved image_filepaths.json.", flush=True)


def get_leaf_parts_of_b2nd_files(directory):
    """
    Given a directory containing a hierarchy of folders with millions of files,
    this function retrieves all leaf parts of all .b2nd files and returns them.

    Args:
        directory (str): The root directory to search.

    Returns:
        list: A list of paths to the leaf parts of all .b2nd files.
    """
    leaf_parts = []

    for root, _, files in tqdm(os.walk(directory), miniters=20000, disable=True):
        for file in files:
            if file.endswith(".b2nd"):
                leaf_parts.append(os.path.join(root, file))

    return leaf_parts


def validate_filepaths(load_dir, save_dir, processes):
    print("Loading image_filepaths.json...", flush=True)
    with open(save_dir / "image_filepaths.json", "r", encoding="utf-8") as f: 
        image_filepaths = json.load(f)
    print("Loaded image_filepaths.json.", flush=True)
        
    print("Validating images...", flush=True)
    valid_ids = {}
    image_validity = tqdmp(validate_image, image_filepaths, processes)
    print("Validated images.", flush=True)

    print("Formatting image paths...", flush=True)
    for path, valid in tqdm(zip(image_filepaths, image_validity)):
        if valid:
            image_id = Path(path).stem
            rel_path = path.replace(load_dir, "")[:-5]
            valid_ids[image_id] = rel_path
    print("Formatted image paths", flush=True)

    print("Saving dataset_index.json...", flush=True)
    with open(save_dir / "dataset_index.json", "w", encoding="utf-8") as f: 
        json.dump(valid_ids, f, indent=4)
    print("Saved dataset_index.json.", flush=True)
        
    print(f"image_filepaths: {len(image_filepaths)}", flush=True)
    print(f"valid_ids: {len(valid_ids)}", flush=True)
    print(f"diff: {len(image_filepaths) - len(valid_ids)}", flush=True)


def validate_image(path):
    try:
        img, meta = Blosc2IO.load(path)
        img = img[0, 0, 0, 0]
        valid = True
    except Exception as e:
        valid = False
    return valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", required=True, type=str, help="Path to the input dir")
    parser.add_argument('-o', "--output", required=True, type=str, help="Path to the output dir")
    parser.add_argument('-s', "--dataset_sub_dir", required=False, type=str, default="nnsslPlans_noresample", help="Name of dataset subdir. Default: 'nnsslPlans_noresample'")
    parser.add_argument('-p', "--processes", required=False, type=int, default=50, help="Number of processes.")
    args = parser.parse_args()

    load_dir = Path(args.input)
    save_dir = Path(args.output)
    dataset_sub_dir = args.dataset_sub_dir
    processes = args.processes

    get_image_filepaths(load_dir, save_dir, dataset_sub_dir)
    validate_filepaths(load_dir, save_dir, processes)