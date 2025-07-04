from blosc2_containerization.image_container import ContainerWriter
import shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm
from tqdmp import tqdmp
import json
from blosc2_containerization.bloscio import Blosc2IO
import gc
import argparse


def register_dataset(json_filepath, save_dir, num_images, num_threads, patch_size):
    json_filepath = Path(json_filepath)
    load_dir = json_filepath.parent
    save_dir = Path(save_dir)

    # shutil.rmtree(save_dir, ignore_errors=True)

    container_writer = ContainerWriter(save_dir, patch_size, np.float32)

    with open(json_filepath, 'r') as openfile:
        images = json.load(openfile)

    image_ids = list(images.keys())
    if num_images is not None:
        image_ids = image_ids[:num_images]
    images = {image_id: images[image_id] for image_id in image_ids}

    print("Registering images...")
    image_shapes = tqdmp(get_image_shape, list(images.values()), num_threads, load_dir=load_dir)

    for image_id, image_shape in tqdm(zip(list(images.keys()), image_shapes)):
        if image_shape is not None:
            container_writer.register_image(image_id, image_shape)

    gc.collect()

    print("Creating containers...")
    container_writer.create_containers()
    container_writer.save_storage()
    print("Finished.")

def get_image_shape(filepath, load_dir):
    load_filepath = load_dir / (filepath + ".b2nd")
    try:
        image_shape = Blosc2IO.load(load_filepath)[0].shape
    except Exception as e:
        print(f"Could not open image {image_id}.")
        print(e)
        image_shape = None
    return image_shape


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", required=True, type=str, help="Path to a json file containing a list of dict(image_id: image_path) pairs.")
    parser.add_argument('-o', "--output", required=True, type=str, help="Absolute output path to the directory used for saving the containerized dataset.")
    parser.add_argument('-n', "--num_images", required=False, type=int, default=None, help="Number of images. None if all images should be registered.")
    parser.add_argument('-t', "--threads", required=False, type=int, default=24, help="Number of threads.")
    parser.add_argument('-p', '--patch_size', default=(1, 160, 160, 160), required=False, type=int, nargs=4, help="The image patch size.")
    args = parser.parse_args()

    register_dataset(args.input, args.output, args.num_images, args.threads, args.patch_size)
