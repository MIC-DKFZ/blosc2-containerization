"""Shape scenarios (priority): channel counts, D vs patch, spatial dims,
dtype, non-contiguous arrays.
"""
import numpy as np
import pytest

from blosc2_containerization.image_container import ContainerReader, ContainerWriter

PATCH = (1, 16, 16, 16)


def rt(tmp_path, spec, dtype=np.float32, seed=0, **kw):
    w = ContainerWriter(tmp_path, PATCH, dtype=dtype, num_threads=2, **kw)
    rng = np.random.default_rng(seed)
    arrs = {k: rng.random(s).astype(dtype) for k, s in spec.items()}
    for k, s in spec.items():
        w.register_image(k, s)
    w.create_containers()
    for k, a in arrs.items():
        w.store_image(k, a)
    w.save_storage()
    r = ContainerReader(tmp_path)
    return arrs, r


class TestChannelCounts:
    def test_c1(self, tmp_path):
        arrs, r = rt(tmp_path, {"x": (1, 16, 16, 16)})
        assert np.array_equal(r.open_image("x")[...], arrs["x"])

    def test_c3(self, tmp_path):
        arrs, r = rt(tmp_path, {"x": (3, 16, 16, 16)})
        assert np.array_equal(r.open_image("x")[...], arrs["x"])

    def test_c256(self, tmp_path):
        arrs, r = rt(tmp_path, {"x": (256, 4, 8, 8)})
        assert np.array_equal(r.open_image("x")[...], arrs["x"])

    def test_mismatched_c_is_separate_not_corrupt(self, tmp_path):
        # F-1: (C,H,W) grouping means C=1 and C=3 images land in different
        # super containers; both round-trip at their own shape without corrupting
        # each other.
        w = ContainerWriter(tmp_path, PATCH, num_threads=2)
        rng = np.random.default_rng(0)
        a1 = rng.random((1, 64, 128, 128)).astype(np.float32)
        a3 = rng.random((3, 64, 128, 128)).astype(np.float32)
        w.register_image("im1", a1.shape)
        w.register_image("im3", a3.shape)
        w.create_containers()
        w.store_image("im1", a1)
        w.store_image("im3", a3)
        w.save_storage()
        r = ContainerReader(tmp_path)
        assert np.array_equal(r.open_image("im1")[...], a1)
        assert np.array_equal(r.open_image("im3")[...], a3)
        assert w.images["im1"].super_container_id != w.images["im3"].super_container_id


class TestDepthVsPatch:
    def test_single_element_1x1x1(self, tmp_path):
        arrs, r = rt(tmp_path, {"x": (1, 1, 1, 1)})
        assert arrs["x"].shape == (1, 1, 1, 1)
        assert np.array_equal(r.open_image("x")[...], arrs["x"])

    def test_small(self, tmp_path):
        arrs, r = rt(tmp_path, {"x": (1, 4, 8, 8)})
        assert np.array_equal(r.open_image("x")[...], arrs["x"])

    def test_exactly_fills_patch(self, tmp_path):
        arrs, r = rt(tmp_path, {"x": (1, 16, 16, 16)})
        assert np.array_equal(r.open_image("x")[...], arrs["x"])

    def test_smaller_than_patch(self, tmp_path):
        arrs, r = rt(tmp_path, {"x": (1, 8, 16, 16)})
        assert np.array_equal(r.open_image("x")[...], arrs["x"])

    def test_spanning_multiple_patches(self, tmp_path):
        arrs, r = rt(tmp_path, {"x": (1, 48, 16, 16)})
        assert np.array_equal(r.open_image("x")[...], arrs["x"])


class TestSpatialDims:
    def test_different_hw_different_super_container(self, tmp_path):
        w = ContainerWriter(tmp_path, PATCH, num_threads=2)
        w.register_image("a", (1, 8, 64, 64))
        w.register_image("b", (1, 8, 128, 64))
        assert w.images["a"].super_container_id != w.images["b"].super_container_id

    def test_matching_hw_same_super_container(self, tmp_path):
        w = ContainerWriter(tmp_path, PATCH, num_threads=2)
        w.register_image("a", (1, 8, 64, 64))
        w.register_image("b", (1, 16, 64, 64))
        assert w.images["a"].super_container_id == w.images["b"].super_container_id


class TestDtype:
    def test_float32_default(self, tmp_path):
        arrs, r = rt(tmp_path, {"x": (1, 8, 16, 16)}, dtype=np.float32)
        got = r.open_image("x")[...]
        assert got.dtype == np.float32
        assert np.array_equal(got, arrs["x"])

    def test_float64(self, tmp_path):
        arrs, r = rt(tmp_path, {"x": (1, 8, 16, 16)}, dtype=np.float64)
        got = r.open_image("x")[...]
        assert got.dtype == np.float64
        assert np.array_equal(got, arrs["x"])

    def test_int32(self, tmp_path):
        w = ContainerWriter(tmp_path, PATCH, dtype=np.int32, num_threads=2)
        a = np.arange(1 * 8 * 16 * 16, dtype=np.int32).reshape(1, 8, 16, 16)
        w.register_image("x", (1, 8, 16, 16))
        w.create_containers()
        w.store_image("x", a)
        w.save_storage()
        r = ContainerReader(tmp_path)
        got = r.open_image("x")[...]
        assert got.dtype == np.int32
        assert np.array_equal(got, a)


class TestNonContiguous:
    def test_store_transposed_view_raises(self, tmp_path):
        w = ContainerWriter(tmp_path, PATCH, num_threads=2)
        a = np.random.default_rng(0).random((1, 16, 16, 16)).astype(np.float32)
        w.register_image("x", a.shape)
        w.create_containers()
        view = a[:, ::-1, :, ::-1]  # same shape, not C-contiguous
        assert not view.flags["C_CONTIGUOUS"]
        with pytest.raises(ValueError):
            w.store_image("x", view)

    def test_store_strided_slice_raises_shape_mismatch(self, tmp_path):
        # arr[::2] on the depth axis halves D, so it no longer matches the
        # registered shape; the guard must raise, not silently store wrong data.
        w = ContainerWriter(tmp_path, PATCH, num_threads=2)
        a = np.random.default_rng(0).random((1, 16, 16, 16)).astype(np.float32)
        w.register_image("x", a.shape)
        w.create_containers()
        half = a[:, ::2, :, :]
        assert not half.flags["C_CONTIGUOUS"]
        with pytest.raises(RuntimeError, match="shape"):
            w.store_image("x", half)
