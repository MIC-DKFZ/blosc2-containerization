"""dataset_* helper functions.

The parallel call paths (register_dataset, validate_filepaths) import tqdmp
lazily and tqdmp is not installed in this env, so only import-level coverage
and the non-parallel helpers are tested (PROJECT.md addendum #8).
"""
import json

import numpy as np

from blosc2_containerization.bloscio import Blosc2IO
from blosc2_containerization.dataset_indexer import (
    get_image_filepaths,
    get_leaf_parts_of_b2nd_files,
    validate_image,
)
from blosc2_containerization.dataset_registerer import get_image_shape
from blosc2_containerization.dataset_storer import split_list, store_dataset, store_image
from blosc2_containerization.image_container import ContainerReader, ContainerWriter


class TestIndexer:
    def test_get_leaf_parts_of_b2nd_files(self, tmp_path):
        (tmp_path / "a" / "b").mkdir(parents=True)
        (tmp_path / "a" / "b" / "x.b2nd").write_bytes(b"\0" * 8)
        (tmp_path / "a" / "y.b2nd").write_bytes(b"\0" * 8)
        (tmp_path / "a" / "notes.txt").write_text("hi")
        found = get_leaf_parts_of_b2nd_files(tmp_path)
        assert len(found) == 2
        assert all(p.endswith(".b2nd") for p in found)

    def test_get_image_filepaths_writes_json(self, tmp_path):
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "x.b2nd").write_bytes(b"\0" * 8)
        out = tmp_path / "out"
        out.mkdir()
        get_image_filepaths(tmp_path, out, "data")
        with open(out / "image_filepaths.json") as f:
            data = json.load(f)
        assert len(data) == 1

    def test_validate_image_valid(self, tmp_path):
        p = str(tmp_path / "v.b2nd")
        Blosc2IO.save(np.random.default_rng(0).random((1, 8, 8, 8)).astype(np.float32), p)
        assert validate_image(p) is True

    def test_validate_image_invalid(self, tmp_path):
        p = str(tmp_path / "bad.b2nd")
        with open(p, "wb") as f:
            f.write(b"\0" * 32)
        assert validate_image(p) is False


class TestRegisterer:
    def test_get_image_shape(self, tmp_path):
        p = str(tmp_path / "img.b2nd")
        Blosc2IO.save(np.random.default_rng(0).random((2, 8, 8, 8)).astype(np.float32), p)
        # the helper appends ".b2nd" to the path it is given
        shape = get_image_shape("img", tmp_path)
        assert tuple(shape) == (2, 8, 8, 8)

    def test_get_image_shape_missing(self, tmp_path):
        assert get_image_shape("nope", tmp_path) is None


class TestStorer:
    def test_split_list_even(self):
        assert split_list([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]

    def test_split_list_uneven(self):
        assert split_list([1, 2, 3, 4, 5], 2) == [[1, 2, 3], [4, 5]]

    def test_split_list_single(self):
        assert split_list([1, 2], 1) == [[1, 2]]

    def test_store_image(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        a = np.random.default_rng(0).random((1, 8, 8, 8)).astype(np.float32)
        Blosc2IO.save(a, str(src / "img.b2nd"))
        store_dir = tmp_path / "store"
        w = ContainerWriter(store_dir, (1, 8, 8, 8), num_threads=2)
        w.register_image("img", (1, 8, 8, 8))
        w.create_containers()
        # store_image appends ".b2nd" to image_path
        store_image("img", "img", src, w)
        w.save_storage()
        r = ContainerReader(store_dir)
        assert np.array_equal(r.open_image("img")[...], a)



class TestStoreDataset:
    """store_dataset: the non-parallel data path of the storer CLI.

    register_dataset (which needs tqdmp) is not exercised here; the helper
    builds the state it would leave behind, then store_dataset runs over it.
    """

    def _dataset(self, tmp_path, n=3, container_size=2):
        src = tmp_path / "src"
        src.mkdir()
        store = tmp_path / "store"
        rng = np.random.default_rng(0)
        manifest, arrs = {}, {}
        w = ContainerWriter(store, (1, 8, 8, 8), num_threads=2, container_size=container_size)
        for i in range(n):
            name = f"img{i}"
            a = rng.random((1, 4, 8, 8)).astype(np.float32)
            Blosc2IO.save(a, str(src / f"{name}.b2nd"))
            manifest[name] = name
            arrs[name] = a
            w.register_image(name, a.shape)
        w.create_containers()
        w.save_storage()
        manifest_path = src / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        return str(manifest_path), store, arrs

    def test_full(self, tmp_path):
        manifest_path, store, arrs = self._dataset(tmp_path)
        store_dataset(manifest_path, str(store), 2, (1, 8, 8, 8))
        r = ContainerReader(store)
        for k, a in arrs.items():
            assert np.array_equal(r.open_image(k)[...], a), k

    def test_indices_only(self, tmp_path):
        # container_size=2 with 3 images -> sub00000(img0, img1), sub00001(img2)
        manifest_path, store, arrs = self._dataset(tmp_path, container_size=2)
        store_dataset(manifest_path, str(store), 2, (1, 8, 8, 8), indices=["1"])
        r = ContainerReader(store)
        assert np.array_equal(r.open_image("img2")[...], arrs["img2"])
        assert not np.any(r.open_image("img0")[...])  # untouched -> zeros

    def test_all_finished_early_return(self, tmp_path):
        manifest_path, store, arrs = self._dataset(tmp_path)
        store_dataset(manifest_path, str(store), 2, (1, 8, 8, 8))
        # second run: every sub-container is FINISHED, so all are filtered
        # out and the function returns early without disturbing the data
        store_dataset(manifest_path, str(store), 2, (1, 8, 8, 8))
        r = ContainerReader(store)
        for k, a in arrs.items():
            assert np.array_equal(r.open_image(k)[...], a), k