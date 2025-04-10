from importlib import metadata

from .base import Recorder

__version__ = metadata.version(__name__.replace("_", "-"))

__all__ = ["Recorder"]
