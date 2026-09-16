"""Import safety: importing the package and every module must not crash.

tqdm is a hard dependency of the dataset_* modules and is installed; tqdmp is
NOT installed and those modules import it only lazily, inside the parallel
call paths, so a plain import must succeed (PROJECT.md addendum #8).
"""
import importlib

import pytest

MODULES = [
    "blosc2_containerization",
    "blosc2_containerization.image_container",
    "blosc2_containerization.bloscio",
    "blosc2_containerization.container_checker",
    "blosc2_containerization.dataset_indexer",
    "blosc2_containerization.dataset_registerer",
    "blosc2_containerization.dataset_storer",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    module = importlib.import_module(name)
    assert module is not None


def test_top_level_exports():
    import blosc2_containerization as pkg

    assert hasattr(pkg, "ContainerWriter")
    assert hasattr(pkg, "ContainerReader")
    assert pkg.__version__ == "0.1.0"
