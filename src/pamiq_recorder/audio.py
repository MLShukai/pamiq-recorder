"""Audio recording module using soundfile."""

from typing import ClassVar, Literal, override

import numpy as np
import numpy.typing as npt
import soundfile as sf

from .base import Recorder, StrPath

type FormatStr = Literal[
    "WAV",
    "AIFF",
    "AU",
    "RAW",
    "PAF",
    "SVX",
    "NIST",
    "VOC",
    "IRCAM",
    "W64",
    "MAT4",
    "MAT5",
    "PVF",
    "XI",
    "HTK",
    "SDS",
    "AVR",
    "WAVEX",
    "SD2",
    "FLAC",
    "CAF",
    "WVE",
    "OGG",
    "MPC2K",
    "RF64",
    "MP3",
]

type SubtypeStr = Literal[
    "PCM_S8",
    "PCM_16",
    "PCM_24",
    "PCM_32",
    "PCM_U8",
    "FLOAT",
    "DOUBLE",
    "ULAW",
    "ALAW",
    "IMA_ADPCM",
    "MS_ADPCM",
    "GSM610",
    "VOX_ADPCM",
    "NMS_ADPCM_16",
    "NMS_ADPCM_24",
    "NMS_ADPCM_32",
    "G721_32",
    "G723_24",
    "G723_40",
    "DWVW_12",
    "DWVW_16",
    "DWVW_24",
    "DWVW_N",
    "DPCM_8",
    "DPCM_16",
    "VORBIS",
    "OPUS",
    "ALAC_16",
    "ALAC_20",
    "ALAC_24",
    "ALAC_32",
    "MPEG_LAYER_I",
    "MPEG_LAYER_II",
    "MPEG_LAYER_III",
]


class AudioRecorder(Recorder[npt.NDArray[np.float32]]):
    """Records audio data to a file using soundfile.

    Supports various audio formats like wav, flac, mp3, m4a and etc
    based on file extension. Input data should be float32 arrays with
    values in the range [-1.0, 1.0].
    """

    DEFAULT_FILE_NAME_FORMAT: ClassVar[str] = Recorder.DEFAULT_FILE_NAME_FORMAT + ".ogg"

    @override
    def __init__(
        self,
        file_path: StrPath,
        sample_rate: int,
        channels: int,
        subtype: SubtypeStr | None = None,
    ) -> None:
        """Initialize an audio recorder.

        Args:
            file_path: Path to save the audio file. File extension determines format.
            sample_rate: Sample rate in Hz.
            channels: Number of audio channels.
            subtype: Sub type of audio file format. If None, default subtype is used.
        """
        super().__init__(file_path)
        self.sample_rate = sample_rate
        self.channels = channels

        # Get format from file extension
        format_name, default_subtype = self._get_format_and_subtype_from_extension(
            self.file_path.suffix
        )

        if subtype is None:
            subtype = default_subtype
        else:
            available_subtypes = sf.available_subtypes(format_name)
            if subtype not in available_subtypes:
                raise ValueError(
                    f"Specified subtype '{subtype}' is invalid for {format_name} format. "
                    f"Available subtypes are {list(available_subtypes.keys())}"
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
    ) -> tuple[FormatStr, SubtypeStr | None]:
        # Remove the leading dot if present
        ext = extension.lower().lstrip(".")

        # Map extensions to format names
        # Note: not all formats support writing in libsndfile/soundfile
        format: FormatStr
        subtype: SubtypeStr | None = None

        match ext:
            # Common formats
            case "wav":
                format = "WAV"
            case "flac":
                format = "FLAC"
            case "ogg":
                format = "OGG"
            case "opus":
                format, subtype = "OGG", "OPUS"
            case "caf" | "m4a" | "mov" | "alac":
                format = "CAF"
            case "mp3":
                format = "MP3"
            case "aiff" | "aif":
                format = "AIFF"
            case "au" | "snd":
                format = "AU"
            case "avr":
                format = "AVR"
            case "htk":
                format = "HTK"
            case "sf" | "ircam":
                format = "IRCAM"
            case "mat4":
                format = "MAT4"
            case "mat5" | "mat":
                format = "MAT5"
            case "nist":
                format = "NIST"
            case "paf":
                format = "PAF"
            case "pvf":
                format = "PVF"
            case "sd2":
                format = "SD2"
            case "sds":
                format = "SDS"
            case "iff" | "svx":
                format = "SVX"
            case "voc":
                format = "VOC"
            case "w64":
                format = "W64"
            case "wavex":
                format = "WAVEX"
            case "wve":
                format = "WVE"
            case "xi":
                format = "XI"
            case "rf64":
                format = "RF64"
            case "mpc" | "mpc2k":
                format = "MPC2K"
            case _:
                raise ValueError(
                    f"Audio format '{ext}' is not supported or recognized."
                )

        if subtype is None:
            # Get the default subtype for this format
            subtype = sf.default_subtype(format)

        return format, subtype

    @override
    def write(self, data: npt.NDArray[np.float32]) -> None:
        """Write audio data to the file.

        Args:
            data: Audio data as numpy array with shape (samples, channels).
                Values should be in range [-1.0, 1.0].

        Raises:
            ValueError: If data shape doesn't match expected dimensions.
            RuntimeError: If the recorder is already closed.
        """
        if self._writer.closed:
            raise RuntimeError("Recorder is already closed.")

        # Convert to float32 if needed
        audio_data = np.asarray(data, dtype=np.float32)

        # Validate dimensions - strictly require 2D array
        if audio_data.ndim != 2:
            raise ValueError(
                f"Expected 2D array with shape (samples, channels), got {audio_data.ndim}D array with shape {audio_data.shape}"
            )

        # Validate channel count
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
