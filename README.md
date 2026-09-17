# Blosc2 Containerization

Efficient management and storage of large-scale 3D medical imaging data using Blosc2 multi-image containers in Python.

## Introduction

Medical imaging research and clinical workflows often require handling thousands of volumetric images, resulting in millions of individual files and significant filesystem overhead. The **Blosc2 Containerization** project provides a scalable, fast, and storage-efficient solution by grouping multiple images with shared dimensions into compressed Blosc2 containers. This reduces filesystem clutter, accelerates I/O, and is especially beneficial on high-performance compute clusters or when working with very large datasets.

The package provides:

* **ContainerWriter**: registers, packs, and writes image arrays into shared containers.
* **ContainerReader**: retrieves entire images or arbitrary patches using image IDs and bounding boxes.
* **Blosc2IO** (`blosc2_containerization.bloscio`): low-level save/load of Blosc2 NDArrays with recommended chunk/block sizing.
* **container_checker** (`blosc2_containerization.container_checker`): verifies container status (`FINISHED` / `UNFINISHED` / `NOT_EXISTING` / `PYTHON_ERROR` / `SEG_FAULT`) for resume support.
* **Dataset CLI tools** (`dataset_indexer`, `dataset_registerer`, `dataset_storer`): index, register, and store a directory of preprocessed `.b2nd` images. A cluster-specific LSF driver is provided in `scripts/dataset_storer_lsf.py` (not part of the installable package).

## Installation

```bash
pip install blosc2_containerization
```

Until the package is published, install from a source checkout:

```bash
pip install /path/to/blosc2_containerization
```

Requirements:

* Python 3.10 – 3.12
* Dependencies: `numpy`, `blosc2>=3.0.0`, `tqdm`, `tqdmp` (the latter two are only exercised by the dataset CLI tools; `tqdmp` provides the parallel progress bar).

## Usage Examples

### Writing Images to Blosc2 Containers

```python
import numpy as np
from blosc2_containerization import ContainerWriter

# Set up storage directory and patch size
storage_dir = "/path/to/image_store"
patch_size = (1, 192, 192, 192)

# Prepare images
images = {
    "img_1": {"shape": (1, 64, 128, 128)},
    "img_2": {"shape": (1, 128, 128, 128)},
    "img_3": {"shape": (1, 127, 128, 128)},
    # ...
}
for image_id in images:
    images[image_id]["array"] = np.random.random(images[image_id]["shape"])

# Initialize the writer
container_writer = ContainerWriter(storage_dir, patch_size, dtype=np.float32)

# Register all images
for image_id, image_dict in images.items():
    container_writer.register_image(image_id, image_dict["shape"])

# Create container files
container_writer.create_containers()

# Store image data
for image_id in images:
    container_writer.store_image(image_id, images[image_id]["array"])

# Save metadata
container_writer.save_storage()
```

### Reading Images and Patches

```python
import numpy as np
from blosc2_containerization import ContainerReader

# Initialize the reader
container_reader = ContainerReader("/path/to/image_store")

# List all image IDs (container-sorted)
image_ids = container_reader.get_image_ids()

# Access region using the ContainerRegion interface. The ContainerRegion is a wrapper for a memory-mapped array of the image region within the container.
region = container_reader.open_image(image_id)
partial = region[:, :32, :, :64]  # Read a sub-volume

# Load a full patch from an image by bounding box (C, D, H, W)
image_id = "img_1"
bbox = ((0, 1), (0, 64), (0, 128), (0, 128))
patch = container_reader.load_patch(image_id, bbox)
print("Patch shape:", patch.shape)
```

### Grouping semantics

* Images are grouped into **super-containers by (C, H, W)** — channel count, height, width — and stacked along the depth (D) axis inside each super-container.
* Images sharing (C, H, W) are packed into the same sub-container up to `container_size` (default 100) images per sub-container, which reduces the number of on-disk files.
* The channel count must be uniform within a group: registering a mismatched channel count places the image in its own super-container (it can never share one with a differently-channeled image), and the write path validates shape/dtype/contiguity and raises a descriptive `ValueError` rather than truncating data.
* The default container dtype is `float32` (`ContainerWriter(dtype=...)` to change it); stored arrays must match it.

### Resume and container status

* Each container records an `is_container_stored` flag. `ContainerWriter.load_storage()` + `create_containers()`/`store_image()` can be re-run over a partially stored dataset: finished sub-containers are detected and skipped, unfinished ones are reopened.
* `check_image_container(path)` from `blosc2_containerization.container_checker` returns one of `Status.FINISHED`, `Status.UNFINISHED`, `Status.NOT_EXISTING`, `Status.PYTHON_ERROR`, or `Status.SEG_FAULT`.

### Dataset CLI tools

The package ships three dataset-level entry points (run as modules or scripts, each with `-i/--input` and `-o/--output`):

* `dataset_indexer` — indexes all `.b2nd` files under an input directory and validates them.
* `dataset_registerer` — registers the indexed images into a `ContainerWriter` and creates the containers.
* `dataset_storer` — stores image arrays into the registered containers (resumable).

A cluster-specific LSF job-submission driver is provided in `scripts/dataset_storer_lsf.py` outside the package; it is not installed and is intended for internal LSF clusters only.

## Features

* Pack thousands of 3D images into a small number of highly compressed container files
* Efficient random-access patch reading (no need to load full image into memory)
* Fully compatible with numpy arrays and medical imaging pipelines
* Metadata-based: quick lookups, resume, and sharing

## License

MIT License

## Acknowledgments

<p align="left">
  <img src="https://github.com/MIC-DKFZ/vidata/raw/main/imgs/Logos/HI_Logo.png" width="150"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/MIC-DKFZ/vidata/raw/main/imgs/Logos/DKFZ_Logo.png" width="500">
</p>

This repository is developed and maintained by the Applied Computer Vision Lab (ACVL)
of [Helmholtz Imaging](https://www.helmholtz-imaging.de/) and the
[Division of Medical Image Computing](https://www.dkfz.de/en/medical-image-computing) at DKFZ.
