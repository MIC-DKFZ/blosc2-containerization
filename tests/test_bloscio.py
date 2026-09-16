"""Blosc2IO: save/load, mode validation (F-14), compression params (F-11)."""
import numpy as np
import pytest

from blosc2_containerization.bloscio import Blosc2IO


class TestSaveLoad:
    def test_roundtrip_with_metadata(self, tmp_path):
        p = str(tmp_path / "a.b2nd")
        a = np.random.default_rng(0).random((4, 8, 8, 8)).astype(np.float32)
        Blosc2IO.save(a, p, metadata={"note": "x"})
        data, meta = Blosc2IO.load(p)
        assert np.array_equal(data[...], a)
        assert meta["note"] == "x"
        assert "b2nd" not in meta

    def test_save_wrong_extension_raises(self, tmp_path):
        a = np.zeros((2, 2, 2, 2), np.float32)
        with pytest.raises(RuntimeError, match="b2nd"):
            Blosc2IO.save(a, str(tmp_path / "a.dat"))

    def test_save_compression_params(self, tmp_path):
        import blosc2

        p = str(tmp_path / "c.b2nd")
        a = np.random.default_rng(1).random((2, 8, 8, 8)).astype(np.float32)
        Blosc2IO.save(
            a,
            p,
            chunks=(2, 4, 4, 4),
            blocks=(2, 2, 2, 2),
            clevel=3,
            codec=blosc2.Codec.LZ4,
        )
        data, _ = Blosc2IO.load(p)
        assert np.array_equal(data[...], a)

    @pytest.mark.parametrize("mode", ["r", "r+"])
    def test_load_modes(self, tmp_path, mode):
        p = str(tmp_path / "a.b2nd")
        a = np.random.default_rng(1).random((2, 4, 4, 4)).astype(np.float32)
        Blosc2IO.save(a, p)
        data, _ = Blosc2IO.load(p, mode=mode)
        assert np.array_equal(data[...], a)

    def test_load_bad_mode_raises(self, tmp_path):
        p = str(tmp_path / "a.b2nd")
        a = np.zeros((2, 2, 2, 2), np.float32)
        Blosc2IO.save(a, p)
        with pytest.raises(ValueError, match="mode must be"):
            Blosc2IO.load(p, mode="w")

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(Exception):
            Blosc2IO.load(str(tmp_path / "nope.b2nd"))


class TestCompBlosc2Params:
    def test_degenerate_channel_raises(self):
        # F-11: no shrinkable spatial axis left and block over L1 budget -> RuntimeError
        with pytest.raises(RuntimeError):
            Blosc2IO.comp_blosc2_params((8192, 1, 1, 1), (1, 1, 1))

    def test_sanity_ok(self):
        block, chunk = Blosc2IO.comp_blosc2_params((4096, 1, 1, 1), (1, 1, 1))
        assert isinstance(block, tuple) and len(block) == 4
        assert isinstance(chunk, tuple) and len(chunk) == 4

    def test_2d_image_size(self):
        block, chunk = Blosc2IO.comp_blosc2_params((64, 64), (16, 16))
        assert len(block) == 2 and len(chunk) == 2

    def test_3d_image_size(self):
        block, chunk = Blosc2IO.comp_blosc2_params((1, 64, 64), (16, 16))
        assert len(block) == 3 and len(chunk) == 3

    def test_2d_patch_size(self):
        block, chunk = Blosc2IO.comp_blosc2_params((1, 64, 64, 64), (16, 16))
        assert len(block) == 4 and len(chunk) == 4

    def test_bad_image_dim_raises(self):
        with pytest.raises(RuntimeError, match="4D"):
            Blosc2IO.comp_blosc2_params((64, 64, 64, 64, 64), (16, 16))

    def test_bad_patch_dim_raises(self):
        with pytest.raises(RuntimeError, match="2D or 3D"):
            Blosc2IO.comp_blosc2_params((1, 64, 64, 64), (16,))

    def test_typical_3d_image(self):
        block, chunk = Blosc2IO.comp_blosc2_params((1, 128, 128, 128), (1, 160, 160))
        # block must not exceed image dims
        assert all(b <= i for b, i in zip(block, (1, 128, 128, 128)))
