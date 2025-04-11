from importlib import metadata

from .audio import AudioRecorder
from .base import Recorder
from .video import VideoRecorder

__version__ = metadata.version(__name__.replace("_", "-"))

__all__ = ["Recorder", "VideoRecorder", "AudioRecorder"]
