"""Smoke tests for the package's __main__ entry points, run in-process via runpy.

These exercise the CLI dispatchers and usage demos that are never imported by
the library API. The parallel dataset CLI paths are not triggered here (they
require a real .b2nd dataset tree); import + dispatch coverage only.
"""
import json
import os
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

import blosc2_containerization
from blosc2_containerization.bloscio import Blosc2IO
from blosc2_containerization.image_container import ContainerReader, ContainerWriter

PKG = Path(blosc2_containerization.__file__).parent


def _run_main(script, argv):
    old = sys.argv
    sys.argv = argv
    try:
        runpy.run_path(str(PKG / script), run_name="__main__")
    finally:
        sys.argv = old


def _register_and_save(store, patch, names_shapes, num_threads=2):
    w = ContainerWriter(store, patch, num_threads=num_threads)
    rng = np.random.default_rng(0)
    arrs = {}
    for name, shape in names_shapes.items():
        a = rng.random(shape).astype(np.float32)
        arrs[name] = a
        w.register_image(name, shape)
    w.create_containers()
    for name, a in arrs.items():
        w.store_image(name, a)
    w.save_storage()
    return arrs


def test_bloscio_main_demo(tmp_path):
    old_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        _run_main("bloscio.py", ["bloscio.py"])
    finally:
        os.chdir(old_cwd)
    assert (tmp_path / "tmp.b2nd").is_file()


def test_image_container_main_demo(capsys):
    # self-contained demo: mkdtemp on entry, rmtree at the end
    _run_main("image_container.py", ["image_container.py"])
    out = capsys.readouterr().out
    for marker in ("Equal (load_patch):", "Equal (open_image):", "Equal:"):
        line = next((l for l in out.splitlines() if l.startswith(marker)), None)
        assert line is not None, f"missing {marker!r} in demo output"
        assert "True" in line


def test_checker_main_dispatch(tmp_path):
    _register_and_save(tmp_path, (1, 8, 8, 8), {"x": (1, 8, 8, 8)})
    w = ContainerWriter(tmp_path, (1, 8, 8, 8), num_threads=2)
    w.load_storage()
    path = str(w.get_sub_containers()[0].path)

    # without --process: dispatches to check_image_container, no exit
    _run_main("container_checker.py", ["container_checker.py", "-i", path])

    # with --process: in-process checker exits with the status value
    with pytest.raises(SystemExit) as exc:
        _run_main("container_checker.py", ["container_checker.py", "-i", path, "--process"])
    assert exc.value.code == 3  # Status.FINISHED.value


def test_storer_main(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    store = tmp_path / "store"
    names_shapes = {f"img{i}": (1, 4, 8, 8) for i in range(2)}
    rng = np.random.default_rng(0)
    manifest, arrs = {}, {}
    w = ContainerWriter(store, (1, 8, 8, 8), num_threads=2)
    for name, shape in names_shapes.items():
        a = rng.random(shape).astype(np.float32)
        Blosc2IO.save(a, str(src / f"{name}.b2nd"))
        manifest[name] = name
        arrs[name] = a
        w.register_image(name, shape)
    w.create_containers()
    w.save_storage()
    manifest_path = src / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    _run_main("dataset_storer.py", [
        "dataset_storer.py",
        "-i", str(manifest_path),
        "-o", str(store),
        "-t", "2",
        "-p", "1", "8", "8", "8",
    ])
    r = ContainerReader(store)
    for k, a in arrs.items():
        assert np.array_equal(r.open_image(k)[...], a), k
