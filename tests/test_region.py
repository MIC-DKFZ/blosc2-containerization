"""ContainerRegion: bbox indexing, validation, and the F-3 ValueError contract.

The F-3 guarantee: region writes validate their input and raise ValueError
(not AssertionError) for bad values, and the validation must survive the
interpreter running with -O (which strips assert statements).
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from blosc2_containerization.image_container import ContainerRegion

CODE_DIR = Path(__file__).resolve().parent.parent


def full_bbox(shape):
    return tuple((0, s) for s in shape)


@pytest.fixture()
def arr():
    # shape (C, D, H, W)
    return np.zeros((2, 4, 5, 6), dtype=np.float32)


@pytest.fixture()
def region(arr):
    return ContainerRegion(arr, full_bbox(arr.shape))


class TestConstruction:
    def test_shape_offset_limits(self, region):
        assert region.ndim == 4
        assert region.shape == (2, 4, 5, 6)
        assert region.offset == (0, 0, 0, 0)
        assert region.limits == (2, 4, 5, 6)

    def test_repr(self, region):
        r = repr(region)
        assert "ContainerRegion" in r
        assert "bbox=" in r

    def test_bbox_dim_mismatch_raises(self, arr):
        with pytest.raises(ValueError, match="BBox must match array dimensions"):
            ContainerRegion(arr, ((0, 2), (0, 4)))


class TestGetItem:
    def test_full_slice(self, region):
        assert region[...].shape == (2, 4, 5, 6)

    def test_int_index(self, arr, region):
        arr[...] = np.arange(2 * 4 * 5 * 6).reshape(2, 4, 5, 6).astype(np.float32)
        assert region[0].shape == (4, 5, 6)
        assert region[0][0, 0, 0] == 0.0

    def test_slice_bounds(self, region):
        assert region[:, 0:2, 1:3, 2:4].shape == (2, 2, 2, 2)

    def test_out_of_bounds_int_raises(self, region):
        with pytest.raises(IndexError, match="out of bbox bounds"):
            region[7]

    def test_out_of_bounds_slice_raises(self, region):
        with pytest.raises(IndexError, match="out of bbox bounds"):
            region[:, 0:99, :, :]

    def test_bad_index_type_raises(self, region):
        with pytest.raises(TypeError, match="Unsupported index type"):
            region["nope"]


class TestSetItemValidation:
    def test_value_shape_too_big_raises(self, region):
        with pytest.raises(ValueError, match="does not fit region shape"):
            region[...] = np.zeros((3, 4, 5, 6), dtype=np.float32)

    def test_dtype_mismatch_raises(self, region):
        with pytest.raises(ValueError, match="does not match container dtype"):
            region[...] = np.zeros((2, 4, 5, 6), dtype=np.float64)

    def test_non_contiguous_raises(self, region):
        base = np.zeros((2, 4, 5, 6), dtype=np.float32)
        view = base[:, ::-1, :, ::-1]  # same shape, not C-contiguous
        assert not view.flags["C_CONTIGUOUS"]
        with pytest.raises(ValueError, match="C-contiguous"):
            region[...] = view

    def test_negative_strides_raises(self, region):
        base = np.zeros((2, 4, 5, 6), dtype=np.float32)
        view = base[:, ::-1, :, :]
        assert any(s < 0 for s in view.strides)
        with pytest.raises(ValueError):
            region[...] = view

    def test_valid_write(self, arr, region):
        region[...] = np.ones((2, 4, 5, 6), dtype=np.float32)
        assert np.array_equal(arr, np.ones((2, 4, 5, 6), dtype=np.float32))

    def test_partial_write(self, arr, region):
        region[:, 0:2, :, :] = np.full((2, 2, 5, 6), 7.0, dtype=np.float32)
        assert np.all(arr[:, 0:2, :, :] == 7.0)
        assert np.all(arr[:, 2:, :, :] == 0.0)


def test_setitem_valueerror_survives_optimize_flag():
    """F-3: validation must raise ValueError under `python -O` too."""
    snippet = (
        "import sys\n"
        f"sys.path.insert(0, {str(CODE_DIR)!r})\n"
        "import numpy as np\n"
        "from blosc2_containerization.image_container import ContainerRegion\n"
        "arr = np.zeros((2, 4, 5, 6), dtype=np.float32)\n"
        "r = ContainerRegion(arr, ((0, 2), (0, 4), (0, 5), (0, 6)))\n"
        "try:\n"
        "    r[...] = np.zeros((3, 4, 5, 6), dtype=np.float32)\n"
        "except ValueError as e:\n"
        "    print('VALUEERROR:', e)\n"
        "    sys.exit(0)\n"
        "print('NO_RAISE')\n"
        "sys.exit(1)\n"
    )
    res = subprocess.run([sys.executable, "-O", "-c", snippet],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "VALUEERROR" in res.stdout
    assert "NO_RAISE" not in res.stdout


def test_strides_guard_reachable_with_size_one_axes():
    # numpy treats an all-size-1 array as C-contiguous even when every stride
    # is negative, which is the one input that passes the contiguity check and
    # reaches the dedicated negative-strides guard.
    arr = np.zeros((1, 1, 1, 1), dtype=np.float32)
    region = ContainerRegion(arr, ((0, 1),) * 4)
    v = arr[::-1, ::-1, ::-1, ::-1]
    assert v.flags["C_CONTIGUOUS"]
    assert any(s < 0 for s in v.strides)
    with pytest.raises(ValueError, match="non-negative strides"):
        region[...] = v
