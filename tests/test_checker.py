"""container_checker: status detection across valid / unfinished / corrupt / garbage.

check_image_container shells out to a subprocess, so the FINISHED path takes a
few seconds; that is normal.
"""
import os

import numpy as np
import pytest

from blosc2_containerization.container_checker import (
    Status,
    check_image_container,
    check_image_container_process,
)
from blosc2_containerization.image_container import ContainerWriter


def _stored_container(tmp_path):
    """Create, store into, and return the path of a finished container file."""
    w = ContainerWriter(tmp_path, (1, 8, 8, 8), num_threads=2)
    a = np.random.default_rng(0).random((1, 8, 8, 8)).astype(np.float32)
    w.register_image("x", (1, 8, 8, 8))
    w.create_containers()
    w.store_image("x", a)
    w.save_storage()
    return [str(s.path) for s in w.get_sub_containers()][0]


class TestStatusEnum:
    def test_member_values(self):
        assert Status.NOT_EXISTING.value == 1
        assert Status.UNFINISHED.value == 2
        assert Status.FINISHED.value == 3
        assert Status.PYTHON_ERROR.value == 4
        assert Status.SEG_FAULT.value == 5


class TestCheckImageContainer:
    def test_none_path(self):
        assert check_image_container(None) == Status.NOT_EXISTING

    def test_missing_path(self, tmp_path):
        assert check_image_container(str(tmp_path / "nope.b2nd")) == Status.NOT_EXISTING

    def test_unfinished_container(self, tmp_path):
        w = ContainerWriter(tmp_path, (1, 8, 8, 8), num_threads=2)
        w.register_image("x", (1, 8, 8, 8))
        w.create_containers()
        path = [str(s.path) for s in w.get_sub_containers()][0]
        assert check_image_container(path) == Status.UNFINISHED

    def test_finished_container(self, tmp_path):
        path = _stored_container(tmp_path)
        assert check_image_container(path) == Status.FINISHED

    def test_truncated_not_finished(self, tmp_path):
        path = _stored_container(tmp_path)
        os.truncate(path, os.path.getsize(path) // 2)
        status = check_image_container(path)
        assert status in (Status.PYTHON_ERROR, Status.SEG_FAULT)

    def test_garbage_not_finished(self, tmp_path):
        p = str(tmp_path / "garbage.b2nd")
        with open(p, "wb") as f:
            f.write(os.urandom(512))
        status = check_image_container(p)
        assert status in (Status.PYTHON_ERROR, Status.SEG_FAULT)


class TestProcessFunction:
    def test_process_returns_finished_via_sysexit(self, tmp_path):
        path = _stored_container(tmp_path)
        with pytest.raises(SystemExit) as exc:
            check_image_container_process(path)
        assert exc.value.code == Status.FINISHED.value

    def test_process_unfinished_via_sysexit(self, tmp_path):
        w = ContainerWriter(tmp_path, (1, 8, 8, 8), num_threads=2)
        w.register_image("x", (1, 8, 8, 8))
        w.create_containers()
        path = [str(s.path) for s in w.get_sub_containers()][0]
        with pytest.raises(SystemExit) as exc:
            check_image_container_process(path)
        assert exc.value.code == Status.UNFINISHED.value

    def test_process_python_error_via_sysexit(self, tmp_path):
        p = str(tmp_path / "garbage.b2nd")
        with open(p, "wb") as f:
            f.write(os.urandom(512))
        with pytest.raises(SystemExit) as exc:
            check_image_container_process(p)
        assert exc.value.code == Status.PYTHON_ERROR.value
