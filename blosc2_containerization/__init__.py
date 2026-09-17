try:
    from blosc2_containerization._version import __version__
except ImportError:
    __version__ = "0.0.0"
from blosc2_containerization.image_container import ContainerWriter, ContainerReader