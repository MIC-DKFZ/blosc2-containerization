# Blosc2 Containerization

[![License Apache Software License 2.0](https://img.shields.io/pypi/l/blosc2_containerization.svg?color=green)](https://github.com/Karol-G/blosc2_containerization/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/blosc2_containerization.svg?color=green)](https://pypi.org/project/blosc2_containerization)
[![Python Version](https://img.shields.io/pypi/pyversions/blosc2_containerization.svg?color=green)](https://python.org)

Efficient management and storage of large-scale 3D medical imaging data using Blosc2 multi-image containers in Python.

## Introduction

Medical imaging research and clinical workflows often require handling thousands of volumetric images, resulting in millions of individual files and significant filesystem overhead. The **Blosc2 Containerization** project provides a scalable, fast, and storage-efficient solution by grouping multiple images with shared spatial dimensions into compressed Blosc2 containers. This reduces filesystem clutter, accelerates I/O, and is especially beneficial on high-performance compute clusters or when working with very large datasets.

This library introduces two core classes:

* **ContainerWriter**: For registering, packing, and writing image arrays into shared containers.
* **ContainerReader**: For retrieving entire images or arbitrary patches using image IDs and bounding boxes.

The approach automatically groups images by shared (H, W) dimensions, so you can efficiently access or store full images or subregions on demand.

## Installation

You can install `blosc2_containerization` via:

```bash
git clone git@git.dkfz.de:mic/internal/mic-rocket/blosc2-containerization.git blosc2_containerization
cd blosc2_containerization
pip install -e .
```

## Usage Examples

### Writing Images to Blosc2 Containers

```python
import numpy as np
from container_writer import ContainerWriter

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
from container_reader import ContainerReader

# Initialize the reader
container_reader = ContainerReader("/path/to/image_store")

# List all image IDs (container-sorted)
image_ids = container_reader.get_image_ids()

# Load a full patch from an image by bounding box (C, D, H, W)
image_id = "img_1"
bbox = ((0, 1), (0, 64), (0, 128), (0, 128))
patch = container_reader.load_patch(image_id, bbox)
print("Patch shape:", patch.shape)

# Access region using the ContainerRegion interface
region = container_reader.open_image(image_id)
partial = region[:, :32, :, :64]  # Read a sub-volume
```

## Features

* Pack thousands of 3D images into a small number of highly compressed container files
* Efficient random-access patch reading (no need to load full image into memory)
* Fully compatible with numpy arrays and medical imaging pipelines
* Metadata-based: quick lookups, resume, and sharing

## License

Apache Software License 2.0
