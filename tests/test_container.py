"""ContainerWriter / ContainerReader / SubContainer: round-trip, resume,
patch loading, and the register/create/store error paths.

Locks in:
- F-1: grouping key is (C, H, W); same H,W with different C -> different
  super containers, both storable and readable at their own shapes.
- F-2: SubContainer.append_image raises ValueError on channel mismatch.
- F-13: ContainerWriter.register_image raises RuntimeError for duplicate id,
  non-4D shape, and register-after-create (any sub-container path set).
"""
import os

import numpy as np
import pytest

from blosc2_containerization.image_container import (
    ContainerReader,
    ContainerWriter,
    Image,
    SubContainer,
)

PATCH = (1, 16, 16, 16)


def make_arrays(spec, dtype=np.float32, seed=0):
    rng = np.random.default_rng(seed)
    return {k: rng.random(s).astype(dtype) for k, s in spec.items()}


def build(tmp_path, spec, dtype=np.float32, seed=0, **kw):
    w = ContainerWriter(tmp_path, PATCH, dtype=dtype, num_threads=2, **kw)
    arrs = make_arrays(spec, dtype=dtype, seed=seed)
    for k, s in spec.items():
        w.register_image(k, s)
    w.create_containers()
    return w, arrs


class TestRoundTrip:
    def test_two_images_same_group(self, tmp_path):
        spec = {"a": (1, 16, 16, 16), "b": (1, 32, 16, 16)}
        w, arrs = build(tmp_path, spec)
        for k, a in arrs.items():
            w.store_image(k, a)
        w.save_storage()
        r = ContainerReader(tmp_path)
        for k, a in arrs.items():
            assert np.array_equal(r.open_image(k)[...], a), k

    def test_load_patch_full_and_corner(self, tmp_path):
        spec = {"img": (1, 16, 16, 16)}
        w, arrs = build(tmp_path, spec)
        w.store_image("img", arrs["img"])
        w.save_storage()
        r = ContainerReader(tmp_path)
        full = r.load_patch("img", ((0, 1), (0, 16), (0, 16), (0, 16)))
        assert np.array_equal(full, arrs["img"])
        corner = r.load_patch("img", ((0, 1), (0, 1), (0, 1), (0, 1)))
        assert corner.shape == (1, 1, 1, 1)
        assert np.array_equal(corner, arrs["img"][:1, :1, :1, :1])
        edge = r.load_patch("img", ((0, 1), (0, 4), (12, 16), (0, 8)))
        assert np.array_equal(edge, arrs["img"][:, 0:4, 12:16, 0:8])
        # empty region
        empty = r.load_patch("img", ((0, 1), (0, 0), (0, 16), (0, 16)))
        assert empty.shape == (1, 0, 16, 16)
        assert empty.size == 0

    def test_open_image_region_slice(self, tmp_path):
        spec = {"img": (2, 8, 8, 8)}
        w, arrs = build(tmp_path, spec)
        w.store_image("img", arrs["img"])
        w.save_storage()
        r = ContainerReader(tmp_path)
        region = r.open_image("img")
        got = region[:, 0:4, :, :]
        assert np.array_equal(got, arrs["img"][:, 0:4, :, :])

    def test_unknown_id_raises(self, tmp_path):
        w, _ = build(tmp_path, {"img": (1, 8, 16, 16)})
        w.save_storage()
        r = ContainerReader(tmp_path)
        with pytest.raises(RuntimeError, match="not found"):
            r.open_image("nope")
        with pytest.raises(RuntimeError, match="not found"):
            r.load_patch("nope", ((0, 1), (0, 1), (0, 1), (0, 1)))


class TestResume:
    def test_store_save_reload_store(self, tmp_path):
        spec = {f"i{i}": (1, 8, 16, 16) for i in range(4)}
        arrs = make_arrays(spec)
        w1 = ContainerWriter(tmp_path, PATCH, num_threads=2, container_size=4)
        for k, s in spec.items():
            w1.register_image(k, s)
        w1.create_containers()
        for k in ("i0", "i1"):
            w1.store_image(k, arrs[k])
        w1.save_storage()

        w2 = ContainerWriter(tmp_path, PATCH, num_threads=2, container_size=4)
        w2.load_storage()
        for k in ("i2", "i3"):
            w2.store_image(k, arrs[k])
        w2.save_storage()

        r = ContainerReader(tmp_path)
        for k, a in arrs.items():
            assert np.array_equal(r.open_image(k)[...], a), k


class TestRegisterErrors:
    def test_duplicate_id_raises(self, tmp_path):
        w = ContainerWriter(tmp_path, PATCH, num_threads=2)
        w.register_image("x", (1, 8, 16, 16))
        with pytest.raises(RuntimeError, match="[Dd]uplicate"):
            w.register_image("x", (1, 8, 16, 16))

    def test_non_4d_shape_raises(self, tmp_path):
        w = ContainerWriter(tmp_path, PATCH, num_threads=2)
        with pytest.raises(RuntimeError, match="dimensions do not match"):
            w.register_image("x", (8, 16, 16))

    def test_register_after_create_raises(self, tmp_path):
        w = ContainerWriter(tmp_path, PATCH, num_threads=2)
        w.register_image("a", (1, 8, 16, 16))
        w.create_containers()
        with pytest.raises(RuntimeError, match="already been created"):
            w.register_image("b", (1, 8, 16, 16))


class TestStoreErrors:
    def test_store_unregistered_raises(self, tmp_path):
        w = ContainerWriter(tmp_path, PATCH, num_threads=2)
        w.register_image("a", (1, 8, 16, 16))
        w.create_containers()
        with pytest.raises(RuntimeError, match="not been registered"):
            w.store_image("nope", np.zeros((1, 8, 16, 16), np.float32))

    def test_store_shape_mismatch_raises(self, tmp_path):
        w, _ = build(tmp_path, {"a": (1, 8, 16, 16)})
        with pytest.raises(RuntimeError, match="shape"):
            w.store_image("a", np.zeros((1, 9, 16, 16), np.float32))

    def test_store_dtype_mismatch_raises(self, tmp_path):
        w, _ = build(tmp_path, {"a": (1, 8, 16, 16)}, dtype=np.float32)
        with pytest.raises(RuntimeError, match="dtype"):
            w.store_image("a", np.zeros((1, 8, 16, 16), np.float64))


class TestGrouping:
    def test_f1_different_c_different_super_container(self, tmp_path):
        w = ContainerWriter(tmp_path, PATCH, num_threads=2)
        w.register_image("c1", (1, 64, 128, 128))
        w.register_image("c3", (3, 64, 128, 128))
        s1 = w.images["c1"].super_container_id
        s3 = w.images["c3"].super_container_id
        assert s1 != s3
        assert "1" in s1 and "3" in s3

    def test_same_chw_same_group(self, tmp_path):
        w = ContainerWriter(tmp_path, PATCH, num_threads=2)
        w.register_image("a", (1, 8, 64, 64))
        w.register_image("b", (1, 16, 64, 64))
        assert w.images["a"].super_container_id == w.images["b"].super_container_id

    def test_different_hw_different_super_container(self, tmp_path):
        w = ContainerWriter(tmp_path, PATCH, num_threads=2)
        w.register_image("a", (1, 8, 64, 64))
        w.register_image("b", (1, 8, 128, 64))
        assert w.images["a"].super_container_id != w.images["b"].super_container_id


class TestSubContainer:
    def test_append_mismatched_channels_raises(self, tmp_path):
        sc = SubContainer(
            "c0", 0, "(1, 16, 16)", tmp_path, shape=(1, 16, 16, 16), dtype=np.float32
        )
        img = Image("img", (3, 16, 16, 16))
        with pytest.raises(ValueError, match="channel count"):
            sc.append_image(img)

    def test_append_extends_depth(self, tmp_path):
        sc = SubContainer(
            "c0", 0, "(1, 16, 16)", tmp_path, shape=(1, 16, 16, 16), dtype=np.float32
        )
        img = Image("img", (1, 8, 16, 16))
        sc.append_image(img)
        assert sc.shape == (1, 24, 16, 16)
        assert img.super_container_id == "(1, 16, 16)"
        assert img.sub_container_index == 0
        assert img.bbox == ((0, 1), (16, 24), (0, 16), (0, 16))

    def test_to_dict_excludes_runtime_state(self, tmp_path):
        sc = SubContainer(
            "c0", 0, "(1,16,16)", tmp_path, shape=(1, 16, 16, 16), image_ids=["a"], dtype=np.float32
        )
        d = sc.to_dict()
        assert "array" not in d
        assert "stored_images" not in d
        assert "is_container_stored" not in d
        assert d["id"] == "c0"
        assert d["image_ids"] == ["a"]


class TestWriterHelpers:
    def test_get_image_ids(self, tmp_path):
        w = ContainerWriter(tmp_path, PATCH, num_threads=2)
        w.register_image("a1", (1, 8, 64, 64))
        w.register_image("b1", (2, 8, 64, 64))
        w.register_image("a2", (1, 16, 64, 64))
        unsorted = w.get_image_ids(container_sorted=False)
        assert set(unsorted) == {"a1", "b1", "a2"}
        s = w.get_image_ids(container_sorted=True)
        assert set(s) == {"a1", "b1", "a2"}
        # container ordering keeps the a-group together
        assert s.index("a1") < s.index("a2")

    def test_get_sub_containers(self, tmp_path):
        w, _ = build(tmp_path, {"a": (1, 8, 16, 16), "b": (1, 16, 16, 16)})
        subs = w.get_sub_containers()
        assert len(subs) == 1

    def test_container_size_rollover(self, tmp_path):
        w = ContainerWriter(tmp_path, PATCH, num_threads=2, container_size=2)
        for i in range(4):
            w.register_image(f"i{i}", (1, 8, 16, 16))
        subs = w.get_sub_containers()
        assert len(subs) == 2
        assert all(len(s.image_ids) <= 2 for s in subs)

    def test_load_storage_roundtrip_metadata(self, tmp_path):
        w, _ = build(tmp_path, {"a": (1, 8, 16, 16), "b": (1, 16, 16, 16)})
        w.save_storage()
        w2 = ContainerWriter(tmp_path, PATCH, num_threads=2)
        w2.load_storage()
        assert list(w2.images["a"].shape) == [1, 8, 16, 16]
        assert w2.get_sub_containers() is not None


class TestInitArray:
    def _stored_dir(self, tmp_path):
        w = ContainerWriter(tmp_path, PATCH, num_threads=2)
        w.register_image("a", (1, 8, 16, 16))
        w.register_image("b", (1, 16, 16, 16))
        w.create_containers()
        rng = np.random.default_rng(0)
        a = rng.random((1, 8, 16, 16)).astype(np.float32)
        b = rng.random((1, 16, 16, 16)).astype(np.float32)
        w.store_image("a", a)
        w.store_image("b", b)
        w.save_storage()
        return a, b

    def test_finished_reopen_keeps_data(self, tmp_path):
        a, b = self._stored_dir(tmp_path)
        w = ContainerWriter(tmp_path, PATCH, num_threads=2)
        w.load_storage()
        sc = w.get_sub_containers()[0]
        sc.init_array()  # status FINISHED -> reopen in place, mark stored
        assert sc.is_container_stored
        r = ContainerReader(tmp_path)
        assert np.array_equal(r.open_image("a")[...], a)
        assert np.array_equal(r.open_image("b")[...], b)

    def test_corrupt_raises_without_overwrite(self, tmp_path):
        self._stored_dir(tmp_path)
        w = ContainerWriter(tmp_path, PATCH, num_threads=2)
        w.load_storage()
        sc = w.get_sub_containers()[0]
        os.truncate(sc.path, os.path.getsize(sc.path) // 2)
        with pytest.raises(RuntimeError, match="Could not initialize container"):
            sc.init_array()

    def test_corrupt_overwrite_recreates(self, tmp_path):
        self._stored_dir(tmp_path)
        w = ContainerWriter(tmp_path, PATCH, num_threads=2)
        w.load_storage()
        sc = w.get_sub_containers()[0]
        os.truncate(sc.path, os.path.getsize(sc.path) // 2)
        sc.init_array(overwrite=True)  # corrupt file removed, container recreated
        assert sc.array is not None
        assert not sc.is_container_stored
        assert os.path.isfile(sc.path)


class TestReaderImageIds:
    def test_sorted_and_unsorted(self, tmp_path):
        w = ContainerWriter(tmp_path, PATCH, num_threads=2)
        w.register_image("a1", (1, 8, 16, 16))
        w.register_image("b1", (2, 8, 16, 16))
        w.register_image("a2", (1, 16, 16, 16))
        w.create_containers()
        rng = np.random.default_rng(0)
        for k in w.images:
            w.store_image(k, rng.random(tuple(w.images[k].shape)).astype(np.float32))
        w.save_storage()
        r = ContainerReader(tmp_path)
        s = r.get_image_ids(container_sorted=True)
        assert set(s) == {"a1", "a2", "b1"}
        assert s.index("a1") < s.index("a2")  # container order keeps groups together
        u = r.get_image_ids(container_sorted=False)
        assert set(u) == {"a1", "a2", "b1"}
