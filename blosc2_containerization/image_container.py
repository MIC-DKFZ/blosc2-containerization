from typing import List, Tuple, Union, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
from collections import defaultdict
import blosc2
from blosc2_containerization.bloscio import Blosc2IO
import json
import os
from blosc2_containerization.container_checker import check_image_container, Status


@dataclass
class Image:
    """
    Represents metadata for a 3D medical image managed by Blosc2 container classes.

    Attributes:
        id (str): Unique identifier for the image.
        shape (Tuple[int, int, int, int]): Shape of the image in (C, D, H, W) format.
        super_container_id (Optional[str]): ID of the super container grouping images with the same (C, H, W) dimensions.
        sub_container_id (Optional[str]): ID of the sub container where this image is stored.
        sub_container_index (Optional[int]): Index of the sub container in the super container.
        bbox (Optional[Tuple[int, int]]): Bounding box in the Blosc2 array indicating where the image is stored.
    """
    id: str
    shape: Tuple[int, int, int, int]  # (C, D, H, W)
    super_container_id: Optional[str] = None
    sub_container_id: Optional[str] = None
    sub_container_index: Optional[int] = None
    bbox: Optional[Tuple[int, int]] = None


class SubContainer:
    """
    Represents a Blosc2 container that stores multiple 3D medical images sharing the same spatial dimensions.

    Attributes:
        id (str): Unique identifier for the sub-container.
        sub_index (int): Index of the sub-container within its super-container.
        super_id (str): ID of the associated super-container.
        storage_dir (str): Path to the directory where this container is stored.
        shape (Optional[Tuple[int, int, int, int]]): Shape of the container array (C, D, H, W).
        image_ids (List[str]): List of image IDs stored in this sub-container.
        patch_size: Patch size for optimal Blosc2 storage (unused directly in SubContainer).
        block_size: Block size parameter for Blosc2 chunking.
        chunk_size: Chunk size parameter for Blosc2 chunking.
        num_threads (int): Number of threads to use for compression/decompression.
        dtype (str): Data type of stored arrays.
        path (Optional[str]): Path to the actual Blosc2 container file.
        stored_images (dict): Tracks which images are already stored.
        is_container_stored (bool): Flag indicating whether all images are stored in the container.
        array (Optional[blosc2.NDArray]): Blosc2 array object for this container.
    """

    def __init__(
        self,
        id: str,
        sub_index: int,
        super_id: str,
        storage_dir: Union[str, Path],
        shape: Optional[Tuple[int, int, int, int]] = None,
        image_ids: List[str] = None,
        patch_size=None,
        block_size=None,
        chunk_size=None,
        num_threads: int = 1,
        dtype=None,
        path=None
    ):
        """
        Initializes a SubContainer object.

        Args:
            id (str): Unique identifier for the sub-container.
            sub_index (int): Index of the sub-container within the super-container.
            super_id (str): ID of the super-container grouping containers with the same C, H and W.
            storage_dir (Union[str, Path]): Path to the directory for storing this container.
            shape (Optional[Tuple[int, int, int, int]]): Shape (C, D, H, W) of the container.
            image_ids (List[str], optional): List of image IDs initially assigned to this container.
            patch_size (optional): Patch size parameter for Blosc2 optimization.
            block_size (optional): Block size parameter for Blosc2 chunking.
            chunk_size (optional): Chunk size parameter for Blosc2 chunking.
            num_threads (int): Number of threads to use for Blosc2 operations.
            dtype (optional): Numpy dtype or string for stored images.
            path (optional): Path to the container file.
        """
        self.id = id
        self.sub_index = sub_index
        self.super_id = super_id
        self.storage_dir = str(storage_dir)
        self.shape = shape
        self.image_ids = image_ids if image_ids is not None else []
        self.patch_size = patch_size
        self.block_size = block_size
        self.chunk_size = chunk_size
        self.num_threads = num_threads
        self.dtype = np.dtype(dtype).str
        self.path = path
        self.stored_images = {image_id: False for image_id in self.image_ids}
        self.is_container_stored = False
        self.array = None

    def init_array(self, overwrite: bool = False):
        """
        Initializes or opens the Blosc2 array for this container.

        Args:
            overwrite (bool): If True, will forcibly re-create the array even if it exists.

        Raises:
            RuntimeError: If container status is invalid or unrecoverable.
        """
        blosc2.set_nthreads(self.num_threads)
        dparams = {'nthreads': self.num_threads}

        if self.array is None:
            status = check_image_container(self.path)

            if status == Status.FINISHED:
                self.array = blosc2.open(self.path, mode="w", dparams=dparams)
                self.is_container_stored = True
            elif status == Status.UNFINISHED and not overwrite:
                self.array = blosc2.open(self.path, mode="w", dparams=dparams)
                self.is_container_stored = False
            elif (status == Status.PYTHON_ERROR or status == Status.SEG_FAULT) and not overwrite:
                raise RuntimeError(f"Could not initialize container. Received container status {status}.")
            else:
                del self.array
                self.array = None
                if self.path is not None and Path(self.path).is_file():
                    os.remove(self.path)

                self.block_size, self.chunk_size = Blosc2IO.comp_blosc2_params(self.shape[1:], self.patch_size[1:])
                self.block_size = tuple(int(x) for x in self.block_size)
                self.chunk_size = tuple(int(x) for x in self.chunk_size)
                self.path = str(Path(self.storage_dir) / (self.id + ".b2nd"))
                self.array = blosc2.empty(
                    shape=self.shape,
                    dtype=np.dtype(self.dtype),
                    urlpath=self.path,
                    mode="w",
                    chunks=(1, *self.chunk_size),
                    blocks=(1, *self.block_size),
                    dparams=dparams
                )
                self.array.vlmeta[b"is_container_stored"] = False
                self.is_container_stored = False

    def append_image(self, image: 'Image'):
        """
        Appends an image's metadata to this sub-container, updates shape and image tracking.

        Args:
            image (Image): The image object to append.

        Side effects:
            Updates internal shape, image_ids, stored_images, and sets container info in image.
        """
        self.image_ids.append(image.id)
        self.stored_images[image.id] = False

        if self.shape is None:
            self.shape = image.shape
        else:
            if image.shape[0] != self.shape[0]:
                raise ValueError(
                    f"Cannot append image '{image.id}': channel count "
                    f"{image.shape[0]} does not match container channel count {self.shape[0]}."
                )
            self.shape = (self.shape[0], self.shape[1] + image.shape[1], self.shape[2], self.shape[3])
        
        image.super_container_id = self.super_id
        image.sub_container_id = self.id
        image.sub_container_index = self.sub_index
        bbox_c = (0, self.shape[0])
        bbox_d = (self.shape[1] - image.shape[1], self.shape[1])
        bbox_h = (0, self.shape[2])
        bbox_w = (0, self.shape[3])            
        bbox = (bbox_c, bbox_d, bbox_h, bbox_w)
        image.bbox = bbox

    def store_image(self, image: 'Image', array: np.ndarray):
        """
        Stores an image array into the sub-container at the location specified by the image's bbox.

        Args:
            image (Image): Image metadata specifying location.
            array (np.ndarray): Numpy array with image data to store.

        Side effects:
            Updates container and image storage status.
        """
        container_region = ContainerRegion(self.array, image.bbox)
        container_region[...] = array
        self.stored_images[image.id] = True
        self.is_container_stored = all(list(self.stored_images.values()))
        self.array.vlmeta[b"is_container_stored"] = self.is_container_stored

    def to_dict(self) -> dict:
        """
        Returns a dictionary representation of the sub-container metadata (excluding non-serializable attributes).

        Returns:
            dict: Serializable dictionary of sub-container attributes.
        """
        state_dict = dict(self.__dict__)
        del state_dict["array"]
        del state_dict["stored_images"]
        del state_dict["is_container_stored"]
        return state_dict


class ContainerRegion:
    """
    Provides bounding-box-based access to a region within a Blosc2 NDArray.

    This allows slicing and assignment within a specific region of the container,
    as defined by the bounding box (bbox). Indexing is mapped relative to the
    global Blosc2 array.

    Attributes:
        array (blosc2.NDArray): The Blosc2 NDArray representing the storage container.
        bbox (tuple[tuple[int, int], ...]): Bounding box for region (one (start, end) per dimension).
        offset (tuple[int, ...]): Start indices for the region in each dimension.
        limits (tuple[int, ...]): Size of the region in each dimension.
    """

    def __init__(self, array: 'blosc2.NDArray', bbox: tuple[tuple[int, int], ...]):
        """
        Initializes a ContainerRegion for a specific region within a Blosc2 array.

        Args:
            array (blosc2.NDArray): The Blosc2 NDArray to index into.
            bbox (tuple[tuple[int, int], ...]): Bounding box, one (start, end) tuple per dimension.

        Raises:
            ValueError: If the bbox dimensionality does not match the array.
        """
        self.array = array
        self.bbox = bbox
        self.offset = tuple(start for start, _ in bbox)
        self.limits = tuple(end - start for start, end in bbox)

        if len(bbox) != self.array.ndim:
            raise ValueError(f"BBox must match array dimensions. Got {len(bbox)} vs {self.array.ndim}")

    def __getitem__(self, key):
        """
        Gets a slice or element within the region, using region-relative indexing.

        Args:
            key (int, slice, tuple): Index or indices, relative to the region.

        Returns:
            np.ndarray: The selected data from the region.

        Raises:
            IndexError: If indices are out of bounds for the region.
            TypeError: If key is not int, slice, or tuple.
        """
        if not isinstance(key, tuple):
            key = (key,)

        key = self._expand_ellipsis(key)
        key = key + (slice(None),) * (self.ndim - len(key))

        bounded_key = tuple(self._bound_index(k, off, lim) for k, off, lim in zip(key, self.offset, self.limits))
        return self.array[bounded_key]

    def __setitem__(self, key, value):
        """
        Sets data within the region, using region-relative indexing.

        Args:
            key (int, slice, tuple): Index or indices, relative to the region.
            value (np.ndarray): Value to assign at the specified indices.

        Raises:
            IndexError: If indices are out of bounds for the region.
            TypeError: If key is not int, slice, or tuple.
        """
        if not isinstance(key, tuple):
            key = (key,)

        key = self._expand_ellipsis(key)
        key = key + (slice(None),) * (self.ndim - len(key))

        bounded_key = tuple(self._bound_index(k, off, lim) for k, off, lim in zip(key, self.offset, self.limits))

        assert all([value.shape[i] <= self.shape[i] for i in range(len(self.shape))])
        assert value.dtype == self.array.dtype
        assert value.flags['C_CONTIGUOUS']
        assert all([s >= 0 for s in value.strides])

        self.array[bounded_key] = value

    def _bound_index(self, idx, offset, limit):
        """
        Maps a region-relative index/slice to a global index/slice within the container.

        Args:
            idx (int or slice): Index or slice, region-relative.
            offset (int): Starting offset for this dimension.
            limit (int): Length of the region in this dimension.

        Returns:
            int or slice: Translated global index or slice.

        Raises:
            IndexError: If index or slice is out of bounds for the region.
            TypeError: If idx is not int or slice.
        """
        if isinstance(idx, int):
            if idx < 0 or idx >= limit:
                raise IndexError("Index out of bbox bounds")
            return idx + offset
        elif isinstance(idx, slice):
            start = 0 if idx.start is None else idx.start
            stop = limit if idx.stop is None else idx.stop
            if start < 0 or stop > limit:
                raise IndexError("Slice out of bbox bounds")
            return slice(start + offset, stop + offset, idx.step)
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")

    def _expand_ellipsis(self, key):
        """
        Expands an ellipsis in a key to fill missing dimensions.

        Args:
            key (tuple): Key tuple (may contain Ellipsis).

        Returns:
            tuple: Key tuple with ellipsis expanded.
        """
        if Ellipsis not in key:
            return key
        idx = key.index(Ellipsis)
        n_missing = self.ndim - (len(key) - 1)
        return key[:idx] + (slice(None),) * n_missing + key[idx+1:]

    @property
    def shape(self) -> tuple:
        """
        Returns the shape of the region.

        Returns:
            tuple: Shape of the region.
        """
        return self.limits

    @property
    def ndim(self) -> int:
        """
        Returns the number of dimensions of the region.

        Returns:
            int: Number of dimensions.
        """
        return len(self.bbox)

    def __repr__(self) -> str:
        """
        Returns a string representation of the ContainerRegion.

        Returns:
            str: String representation.
        """
        return f"{self.__class__.__name__}(shape={self.shape}, bbox={self.bbox})"


class ContainerWriter:
    """
    Manages efficient storage of multiple 3D medical images using Blosc2 containers,
    grouping images with shared channel count and spatial dimensions.

    This class enables efficient disk storage of multiple 3D medical images by grouping them into shared
    Blosc2 containers based on their channel count and spatial dimensions. Specifically, images with the
    same number of channels (C), height (H) and width (W) are packed into the same container along the
    depth (D) axis, allowing for flexible
    depths while reducing the number of files and improving storage efficiency—especially important in
    large-scale compute environments with inode limitations.

    Usage follows a staged approach:
    1. **Register images**: Each image must be registered with a unique identifier and its shape 
        `(C, D, H, W)`. The actual array data is not required at this stage.
    2. **Create containers**: Once all images are registered, containers are created based on their shared
        (C, H, W) dimensions.
    3. **Store image data**: After container creation, individual image arrays can be stored in the appropriate
        location within each container.
    4. **Save metadata**: All container and image mapping metadata is serialized for later loading and reuse.

    Example:
        ```python
        container_writer = ContainerWriter("/path/to/image_store")

        container_writer.register_image(Image("confining_backlash_085", (1, 64, 128, 128)))
        container_writer.register_image(Image("electoral_redistribution_125", (1, 64, 128, 128)))
        container_writer.register_image(Image("smart_bellflower_749", (1, 127, 178, 128)))

        container_writer.create_containers()

        container_writer.store_image("confining_backlash_085", np.random.random((1, 64, 128, 128)))
        container_writer.store_image("electoral_redistribution_125", np.random.random((1, 64, 128, 128)))
        container_writer.store_image("smart_bellflower_749", np.random.random((1, 127, 178, 128)))

        container_writer.save_storage()
        ```

    Attributes:
        storage_dir (Path): Directory for storing container files and metadata.
        patch_size: Patch size used for Blosc2 storage optimization.
        h_axis (int): Index of the height axis in image shapes (default 2).
        w_axis (int): Index of the width axis in image shapes (default 3).
        d_axis (int): Index of the depth axis in image shapes (default 1).
        images (dict): Mapping of image_id to Image dataclasses.
        containers (defaultdict): Mapping of super_container_id to subcontainers.
        dtype: Data type for all images.
        num_threads (int): Number of Blosc2 threads.
        container_size (int): Maximum number of images per sub-container.
    """

    def __init__(
        self,
        storage_dir: Union[str, Path],
        patch_size,
        dtype=np.float32,
        num_threads: int = 12,
        container_size: int = 100
    ):
        """
        Initializes a ContainerWriter instance for efficient image storage using Blosc2 containers.

        Args:
            storage_dir (Union[str, Path]): Directory for Blosc2 container files and metadata.
            patch_size: Patch size for optimizing Blosc2 chunking (typically tuple of ints).
            dtype: Numpy data type for images (default: np.float32).
            num_threads (int): Number of Blosc2 threads to use (default: 12).
            container_size (int): Max number of images per sub-container (default: 100).
        """
        self.storage_dir = Path(storage_dir)
        self.patch_size = patch_size
        Path(self.storage_dir).mkdir(parents=True, exist_ok=True)
        self.h_axis, self.w_axis, self.d_axis = 2, 3, 1
        self.images = {}
        self.containers = defaultdict(dict)
        self.dtype = dtype
        self.num_threads = num_threads
        self.container_size = container_size

    def register_image(self, image_id: str, shape: Tuple[int, int, int, int]):
        """
        Registers an image for storage, grouping it by its (C, H, W) dimensions.

        Args:
            image_id (str): Unique string identifier for the image.
            shape (Tuple[int, int, int, int]): Shape of the image (C, D, H, W).

        Raises:
            RuntimeError: If containers already created, image ID is duplicate, or shape is invalid.
        """
        if image_id in self.images:
            raise RuntimeError("Duplicate image id found in images.")
        if len(shape) != 4:
            raise RuntimeError("Expected shape to be (C, D, H, W), but image dimensions do not match.")
        
        image = Image(image_id, shape)
        self.images[image_id] = image
        super_container_id = (image.shape[0], image.shape[self.h_axis], image.shape[self.w_axis])
        sub_container = self._get_sub_container(super_container_id)
        sub_container.append_image(image)

    def create_containers(self):
        """
        Creates Blosc2 containers on disk for all registered images.

        Allocates optimal storage layout by (C, H, W) group and creates the necessary sub-containers.
        """
        for super_container in self.containers.values():
            for sub_container in super_container.values():
                sub_container.init_array()
        num_sub_containers = sum([len(super_container) for super_container in self.containers.values()])
        print(
            f"Registered images: {len(self.images)}, "
            f"created containers: {num_sub_containers}, "
            f"inode reduction: {round((1 - (num_sub_containers / len(self.images))) * 100, 2)}%"
        )

    def store_image(self, id: str, array: np.ndarray):
        """
        Stores the array for a registered image in its assigned Blosc2 container.

        Args:
            id (str): ID of the registered image.
            array (np.ndarray): Image array of shape matching registered metadata.

        Raises:
            RuntimeError: If the image was not registered, or shape/dtype mismatch.
        """
        if id not in self.images:
            raise RuntimeError("Image has not been registered before.")
        image = self.images[id]
        if not np.array_equal(array.shape, image.shape):
            raise RuntimeError(f"Array shape of image ({id}) does not match registered image shape")
        sub_container = self.containers[str(image.super_container_id)][str(image.sub_container_index)]
        sub_container.init_array()
        if array.dtype != sub_container.array.dtype:
            raise RuntimeError(f"Array dtype of image ({id}) is different to container dtype.")
        sub_container.store_image(image, array)

    def get_image_ids(self, container_sorted: bool = True) -> list:
        """
        Returns a list of all registered image IDs.

        Args:
            container_sorted (bool): If True, order by container layout; else, unordered.

        Returns:
            list: List of image IDs.
        """
        if not container_sorted:
            return list(self.images.keys())
        else:
            images_ids = []
            for super_container in self.containers.values():
                for sub_container in super_container.values():
                    images_ids.extend(sub_container.image_ids)
            return images_ids

    def get_sub_containers(self) -> list:
        """
        Returns a list of all sub-containers.

        Returns:
            list: List of SubContainer instances.
        """
        sub_containers = []
        for super_container in self.containers.values():
            sub_containers.extend(list(super_container.values()))
        return sub_containers

    def load_storage(self):
        """
        Loads container and image metadata from disk.

        Side Effects:
            Populates self.containers and self.images from metadata file.
        """
        with open(self.storage_dir / "storage_metadata.json", "r", encoding="utf-8") as f: 
            storage = json.load(f)
        self.containers = {
            str(super_container_id): {
                str(sub_container_id): SubContainer(**sub_container) 
                for sub_container_id, sub_container in super_container.items()
            } for super_container_id, super_container in storage["containers"].items()
        }
        self.containers = self._to_defaultdict(self.containers)
        self.images = {
            image["id"]: Image(**image) for image in storage["images"].values()
        }

    def save_storage(self):
        """
        Saves all container and image metadata to disk as a JSON file.

        Side Effects:
            Writes a "storage_metadata.json" file to self.storage_dir.
        """
        storage = {
            "containers": {
                super_container_id: {
                    sub_container_id: sub_container.to_dict()
                    for sub_container_id, sub_container in super_container.items()
                } for super_container_id, super_container in self.containers.items()
            },
            "images": {
                image_id: asdict(image) for image_id, image in self.images.items()
            }
        }
        with open(self.storage_dir / "storage_metadata.json", "w", encoding="utf-8") as f: 
            json.dump(storage, f, indent=4)

    def _get_sub_container(self, super_container_id):
        """
        Retrieves or creates a sub-container for a given super-container (C, H, W) group.

        Args:
            super_container_id: Key representing the (C, H, W) group for the container.

        Returns:
            SubContainer: The assigned or newly created sub-container.
        """
        sub_container_index = len(self.containers[str(super_container_id)]) - 1

        create_container = False
        if sub_container_index == -1:
            sub_container_index = 0
            create_container = True
        elif (self.containers[str(super_container_id)][str(sub_container_index)].path is not None) or \
                (len(self.containers[str(super_container_id)][str(sub_container_index)].image_ids) >= self.container_size):
            sub_container_index += 1
            create_container = True

        if create_container:
            sub_container_id = f"container{super_container_id}_sub{str(sub_container_index).zfill(5)}"
            sub_container = SubContainer(
                sub_container_id, sub_container_index, str(super_container_id),
                self.storage_dir, patch_size=self.patch_size, num_threads=self.num_threads, dtype=self.dtype
            )
            self.containers[str(super_container_id)][str(sub_container_index)] = sub_container
        else:
            sub_container = self.containers[str(super_container_id)][str(sub_container_index)]

        return sub_container
    
    def _to_defaultdict(self, d):
        if not isinstance(d, dict):
            return d
        return defaultdict(dict, {k: self._to_defaultdict(v) for k, v in d.items()})
    

class ContainerReader:
    """
    Provides access to image data stored in Blosc2 containers using metadata written by ContainerWriter.

    Allows reading of full images or arbitrary patches from 3D medical images stored in
    shared container files, using region and bounding-box indexing.

    Usage:
        1. Instantiate the loader with the directory containing saved containers and metadata.
        2. Use `open_image()` to load a ContainerRegion representing the image or 
        use `load_patch()` to extract specific image patches using their ID and a image bounding box.

    Example:
        ```python
        container_reader = ContainerReader("/path/to/image_store")
        patch1 = container_reader.open_image("electoral_redistribution_125")[:, 32:64, :128, :17]
        patch2 = container_reader.load_patch("electoral_redistribution_125", bbox=((0, 1), (32, 64), (0, 128), (0, 17)))
        ```

    Attributes:
        storage_dir (Path): Path to the directory containing containers and metadata.
        containers (dict): Mapping of super-container IDs to subcontainers.
        images (dict): Mapping of image IDs to Image dataclasses.
    """

    def __init__(self, storage_dir: Union[str, Path]):
        """
        Initializes a ContainerReader for reading image data from Blosc2 containers.

        Args:
            storage_dir (Union[str, Path]): Path to the directory where container files
                and associated metadata ("storage_metadata.json") are stored.

        Side Effects:
            Loads metadata and sets up containers/images mappings for fast lookup.
        """
        self.storage_dir = Path(storage_dir)
        with open(self.storage_dir / "storage_metadata.json", "r", encoding="utf-8") as f: 
            storage = json.load(f)
        self.containers = {}
        for super_container_id, super_container in storage["containers"].items():
            self.containers[str(super_container_id)] = {}
            for sub_container_id, sub_container in super_container.items():
                sub_container["storage_dir"] = str(storage_dir)
                sub_container["path"] = f"{str(storage_dir)}/{sub_container['id']}.b2nd"
                self.containers[str(super_container_id)][str(sub_container_id)] = SubContainer(**sub_container)

        self.images = {
            image["id"]: Image(**image) for image in storage["images"].values()
        }

    def get_image_ids(self, container_sorted: bool = True) -> list:
        """
        Returns a list of all image IDs present in the storage.

        Args:
            container_sorted (bool): If True, IDs are returned in container order; otherwise, unordered.

        Returns:
            list: List of image IDs.
        """
        if not container_sorted:
            return list(self.images.keys())
        else:
            images_ids = []
            for super_container in self.containers.values():
                for sub_container in super_container.values():
                    images_ids.extend(sub_container.image_ids)
            return images_ids

    def open_image(self, id: str) -> 'ContainerRegion':
        """
        Opens a region for a specific image, enabling region-based slicing.

        Args:
            id (str): Image ID to open.

        Returns:
            ContainerRegion: ContainerRegion object representing the region for the image.

        Raises:
            RuntimeError: If the image ID is not present in metadata.
        """
        if id not in self.images:
            raise RuntimeError(f"Image ID '{id}' not found in metadata.")
        
        image = self.images[id]
        container = self.containers[str(image.super_container_id)][str(image.sub_container_index)]
        array = Blosc2IO.load(container.path)[0]
        container_region = ContainerRegion(array, image.bbox)
        return container_region

    def load_patch(
        self,
        id: str,
        bbox: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]
    ) -> np.ndarray:
        """
        Loads a patch (subregion) from a stored image.

        Args:
            id (str): Image ID to load patch from.
            bbox (Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]):
                Bounding box to load, in (C, D, H, W) dimensions, relative to the image.

        Returns:
            np.ndarray: Patch extracted from the stored container.

        Raises:
            RuntimeError: If the image ID is not present in metadata.
        """
        if id not in self.images:
            raise RuntimeError(f"Image ID '{id}' not found in metadata.")
        
        container_region = self.open_image(id)
        image_bbox_slices = tuple([slice(bbox_dim[0], bbox_dim[1]) for bbox_dim in bbox])
        image_patch = container_region[image_bbox_slices]
        return image_patch


if __name__ == "__main__":
    # Example usage of the ContainerWriter and ContainerReader classes
    import numpy as np
    import shutil

    # Set up the storage directory
    storage_dir = "/home/k539i/Documents/datasets/original/image_store"

    # Clear previous test data
    shutil.rmtree(storage_dir, ignore_errors=True)

    # Define a list of example image metadata entries
    images = {}

    images["confining_backlash_085"] = {"shape":       (1, 64, 128, 128)}
    images["electoral_redistribution_125"] = {"shape": (1, 64, 128, 128)}
    images["irrational_gazelle_637"] = {"shape":       (1, 64, 128, 128)}
    images["materialistic_hardship_929"] = {"shape":   (1, 128, 128, 128)}
    images["renewed_ammonia_718"] = {"shape":          (1, 127, 128, 128)}

    images["smart_bellflower_749"] = {"shape":         (1, 127, 178, 128)}

    images["subjective_irrigation_395"] = {"shape":    (1, 64, 128, 256)}

    for i, image_id in enumerate(images.keys()):
        array = np.random.random(images[image_id]["shape"])
        # array = np.ones(images[image_id]["shape"]) * i
        images[image_id]["array"] = array

    # Initialize saver and register images
    container_writer = ContainerWriter(storage_dir, (1, 192, 192, 192), images["confining_backlash_085"]["array"].dtype)

    for image_id, image_dict in images.items():
        container_writer.register_image(image_id, image_dict["shape"])

    image_ids = container_writer.get_image_ids()

    # Create container files and store image arrays
    container_writer.create_containers()
    for image_id in image_ids:
        container_writer.store_image(image_id, images[image_id]["array"])

    # Save container metadata
    container_writer.save_storage()

    # Load container metadata and array data
    container_reader = ContainerReader(storage_dir)

    # Load a patch from a specific image
    image_id = "electoral_redistribution_125"
    image_bbox = ((0, 1), (32, 64), (0, 128), (0, 17))
    image_bbox_slices = tuple([slice(bbox_dim[0], bbox_dim[1]) for bbox_dim in image_bbox])
    image_patch0 = images[image_id]["array"][image_bbox_slices]
    image_patch1 = container_reader.load_patch(image_id, image_bbox)
    print("Image patch shape: ", image_patch1.shape)
    image = container_reader.open_image(image_id)
    image_patch2 = image[image_bbox_slices]
    print("Image patch shape: ", image_patch2.shape)
    print("Equal (load_patch): ", np.array_equal(image_patch0, image_patch1)) # load_patch
    print("Equal (open_image): ", np.array_equal(image_patch0, image_patch2)) # open_image
    print("Equal: ", np.array_equal(image_patch1, image_patch2))