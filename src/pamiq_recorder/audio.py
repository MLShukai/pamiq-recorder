"""Audio recording module using soundfile."""

from pathlib import Path
from typing import Literal, override

import numpy as np
import numpy.typing as npt
import soundfile as sf

from .base import Recorder

type SupportFormat = Literal["WAV", "FLAC", "OGG", "CAF", "MP3"]
type SupportSubtype = Literal[
    "FLOAT", "PCM_16", "OPUS", "VORBIS", "ALAC_16", "MPEG_LAYER_III"
]


class AudioRecorder(Recorder[npt.NDArray[np.float32]]):
    """Records audio data to a file using soundfile.

    Supports various audio formats like wav, flac, mp3, m4a, ogg and
    opusbased on file extension. Input data should be float32 arrays
    with values in the range [-1.0, 1.0].
    """

    def __init__(
        self,
        file_path: str | Path,
        sample_rate: int,
        channels: int,
    ) -> None:
        """Initialize an audio recorder.

        Args:
            file_path: Path to save the audio file. File extension determines format.
            sample_rate: Sample rate in Hz.
            channels: Number of audio channels.
        """
        self.file_path = Path(file_path)
        self.sample_rate = sample_rate
        self.channels = channels

        # Get format from file extension
        format_name, subtype = self._get_format_and_subtype_from_extension(
            self.file_path.suffix
        )

        # Initialize SoundFile for streaming writes

        self._writer = sf.SoundFile(
            str(self.file_path),
            mode="w",
            samplerate=sample_rate,
            channels=channels,
            format=format_name,
            subtype=subtype,
        )

    def _get_format_and_subtype_from_extension(
        self, extension: str
    ) -> tuple[SupportFormat, SupportSubtype]:
        # Remove the leading dot if present
        ext = extension.lower().lstrip(".")

        # Map extensions to format names
        # Note: not all formats support writing in libsndfile/soundfile
        match ext:
            case "wav":
                return "WAV", "PCM_16"
            case "flac":
                return "FLAC", "PCM_16"
            case "ogg":
                return "OGG", "VORBIS"
            case "opus":
                return "OGG", "OPUS"
            case "m4a" | "mov" | "alac":
                return "CAF", "ALAC_16"
            case "mp3":
                return "MP3", "MPEG_LAYER_III"
            case _:
                raise ValueError(
                    f"Audio format '{ext}' is not supported or recognized."
                )

    @override
    def write(self, data: npt.NDArray[np.float32]) -> None:
        """Write audio data to the file.

        Args:
            data: Audio data as numpy array with shape (samples, channels)
                 or (samples,) for mono audio. Values should be in range [-1.0, 1.0].

        Raises:
            ValueError: If data shape doesn't match expected dimensions.
            RuntimeError: If the recorder is already closed.
        """
        if not hasattr(self, "_writer") or self._writer.closed:
            raise RuntimeError("Recorder is already closed.")

        # Convert to float32 if needed
        audio_data = np.asarray(data, dtype=np.float32)

        # Validate dimensions
        if audio_data.ndim not in [1, 2]:
            raise ValueError(f"Expected 1D or 2D array, got {audio_data.ndim}D")

        # For mono data with shape (samples,), check if it matches the channels setting
        if audio_data.ndim == 1:
            if self.channels != 1:
                raise ValueError(
                    f"Expected {self.channels} channels, but got mono data with shape {audio_data.shape}"
                )
        else:  # audio_data.ndim == 2
            if audio_data.shape[1] != self.channels:
                raise ValueError(
                    f"Expected {self.channels} channels, got data with shape {audio_data.shape}"
                )

        # Write the audio data
        self._writer.write(audio_data)

    @override
    def close(self) -> None:
        """Close the audio writer and release resources."""
        if hasattr(self, "_writer") and not self._writer.closed:
            self._writer.close()
