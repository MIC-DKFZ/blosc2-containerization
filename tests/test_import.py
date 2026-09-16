"""Import safety: importing the package and every module must not crash.

tqdm and tqdmp are declared dependencies of the dataset_* modules and are
installed in this env; both are imported at module level, so a plain import
must succeed and the dataset modules must expose the tqdmp symbol.
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

def test_dataset_modules_expose_tqdmp_at_module_level():
    # tqdmp is a declared dependency imported at top level (not lazily)
    from blosc2_containerization import dataset_indexer, dataset_registerer

    assert hasattr(dataset_indexer, "tqdmp")
    assert hasattr(dataset_registerer, "tqdmp")
