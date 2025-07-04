from blosc2_containerization.image_container import ContainerWriter
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from blosc2_containerization.bloscio import Blosc2IO
# import threading
# import queue
import argparse


def store_dataset(json_filepath, save_dir, num_threads, patch_size, num_loader_threads=1, indices=None):
    json_filepath = Path(json_filepath)
    load_dir = json_filepath.parent
    save_dir = Path(save_dir)

    container_writer = ContainerWriter(save_dir, patch_size, np.float32, 1)
    container_writer.load_storage()

    with open(json_filepath, 'r') as openfile:
        images = json.load(openfile)

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
    # load_queue = queue.Queue(maxsize=8)

    # loader_threads = []
    # for i in range(num_loader_threads):
    #     t = threading.Thread(target=image_loader, args=(i, image_ids, images, load_dir, load_queue, num_threads, num_loader_threads))
    #     t.start()
    #     loader_threads.append(t)

    # saver_thread = threading.Thread(target=image_saver, args=(container_writer, load_queue, num_loader_threads, len(image_ids)))
    # saver_thread.start()

    # for t in loader_threads:
    #     t.join()
    # saver_thread.join()

    for image_id, image_path in tqdm(images.items()):
        store_image(image_id, image_path, load_dir, container_writer)

    print("Finished.")


def store_image(image_id, image_path, load_dir, container_writer):
    load_filepath = load_dir / (image_path + ".b2nd")
    array = Blosc2IO.load(load_filepath, num_threads=1)[0][...]
    container_writer.store_image(image_id, array)


# def image_loader(worker_id, image_ids, images, load_dir, load_queue, num_threads, num_loader_threads):
#     for i in range(worker_id, len(image_ids), num_loader_threads):
#         image_id = image_ids[i]
#         load_filepath = load_dir / (images[image_id] + ".b2nd")
#         array = Blosc2IO.load(load_filepath, num_threads=num_threads)[0][...]
#         load_queue.put((image_id, array))
#     load_queue.put(None)  # Each loader adds a sentinel

# def image_saver(container_writer, load_queue, num_loader_threads, total):
#     pbar = tqdm(total=total, desc="Store images")
#     finished_loaders = 0
#     while True:
#         item = load_queue.get()
#         if item is None:
#             finished_loaders += 1
#             if finished_loaders == num_loader_threads:
#                 break
#             continue
#         image_id, array = item
#         container_writer.store_image(image_id, array)
#         pbar.update(1)
#     pbar.close()


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
    