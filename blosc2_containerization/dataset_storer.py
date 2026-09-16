from blosc2_containerization.image_container import ContainerWriter
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from blosc2_containerization.bloscio import Blosc2IO
import argparse


def store_dataset(json_filepath, save_dir, num_threads, patch_size, indices=None):
    json_filepath = Path(json_filepath)
    load_dir = json_filepath.parent
    save_dir = Path(save_dir)

    print("Initializing container writer...")
    container_writer = ContainerWriter(save_dir, patch_size, np.float32, num_threads)
    container_writer.load_storage()

    with open(json_filepath, 'r') as openfile:
        images = json.load(openfile)

    print("Retrieving sub-containers...")
    sub_containers = container_writer.get_sub_containers()
    num_sub_containers = len(sub_containers)

    if indices is not None:
        sub_containers = [sub_containers[int(i)] for i in indices]
        print(f"Processing the following sub-containers ({len(sub_containers)}/{num_sub_containers}): {[sub_container.id for sub_container in sub_containers]}")

    print("Initializing sub-containers...")
    for sub_container in sub_containers:
        sub_container.init_array(overwrite=True)
    print("Initialized sub-containers.")

    print("Filtering out complete sub-containers...")
    unfinished_sub_containers = []
    for i in range(len(sub_containers)):
        if not sub_containers[i].is_container_stored:
            unfinished_sub_containers.append(sub_containers[i])
    sub_containers = unfinished_sub_containers
    print("Filtered out complete sub-containers.")

    if len(sub_containers) == 0:
        print("All sub containers are already finished.")
        return

    image_ids = []
    for sub_container in sub_containers:
        image_ids.extend(sub_container.image_ids)
    images = {image_id: images[image_id] for image_id in image_ids}

    print("Storing images in sub-containers...")

    for image_id, image_path in tqdm(images.items()):
        store_image(image_id, image_path, load_dir, container_writer)

    print("Finished.")


def store_image(image_id, image_path, load_dir, container_writer):
    root_path = "/omics/groups/OE0441/e230-thrp-data/mic_rocket/nnssl_preprocessed/Dataset805_Rocket_v4"
    root_path = Path(root_path)
    load_filepath = load_dir / (image_path + ".b2nd")
    array = Blosc2IO.load(load_filepath, num_threads=1)[0][...]
    container_writer.store_image(image_id, array)


def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", required=True, type=str, help="Path to a json file containing a list of dict(image_id: image_path) pairs.")
    parser.add_argument('-o', "--output", required=True, type=str, help="Absolute output path to the directory used for saving the containerized dataset.")
    parser.add_argument('-t', "--threads", required=False, type=int, default=24, help="Number of threads.")
    parser.add_argument('-p', '--patch_size', default=(1, 160, 160, 160), required=False, type=int, nargs=4, help="The image patch size.")
    parser.add_argument("--indices", required=False, type=str, default=None, nargs="+", help="Sub-Container indices that should be processed")
    args = parser.parse_args()

    store_dataset(args.input, args.output, args.threads, args.patch_size, indices=args.indices)
    