"""Type stubs for the SoundFile class from the soundfile package."""

from os import SEEK_SET
from typing import (
    Any,
    Generator,
    Literal,
    Optional,
    Protocol,
    Union,
    overload,
)

import numpy.typing as npt

# Format constants
_FormatStr = Literal[
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

# Subtype constants
_SubtypeStr = Literal[
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

# Endian constants
_EndianStr = Literal["FILE", "LITTLE", "BIG", "CPU"]

# Mode constants
_ModeStr = Literal["r", "r+", "w", "w+", "x", "x+"]

# Dtype constants
_DtypeStr = Literal["float64", "float32", "int32", "int16"]

# Bitrate mode constants
_BitrateModeStr = Literal["CONSTANT", "AVERAGE", "VARIABLE"]

# For file-like objects with specific methods needed for SoundFile
class _FileLike(Protocol):
    def seek(self, offset: int, whence: int = ...) -> int: ...
    def tell(self) -> int: ...
    def write(self, data: bytes) -> int: ...
    def read(self, size: int = ...) -> bytes: ...

# Main SoundFile class
class SoundFile:
    """A sound file.

    This class provides methods for reading and writing sound files in various formats.
    """

    def __init__(
        self,
        file: Union[str, int, _FileLike],
        mode: _ModeStr = "r",
        samplerate: Optional[int] = None,
        channels: Optional[int] = None,
        subtype: Optional[_SubtypeStr] = None,
        endian: Optional[_EndianStr] = None,
        format: Optional[_FormatStr] = None,
        closefd: bool = True,
        compression_level: Optional[float] = None,
        bitrate_mode: Optional[_BitrateModeStr] = None,
    ) -> None:
        """Open a sound file.

        Args:
            file: The file to open. This can be a file name, a file descriptor or a Python file object.
            mode: Open mode. 'r' for reading, 'w' for writing, 'x' for exclusive creation.
                 Add '+' for reading and writing.
            samplerate: The sample rate of the file. Required for writing.
            channels: The number of channels of the file. Required for writing.
            subtype: The subtype of the sound file.
            endian: The endian-ness of the sound file.
            format: The major format of the sound file.
            closefd: Whether to close the file descriptor on close().
            compression_level: The compression level (0.0 to 1.0).
            bitrate_mode: The bitrate mode ('CONSTANT', 'AVERAGE', 'VARIABLE').
        """
        ...

    # Properties
    @property
    def name(self) -> str:
        """The file name of the sound file."""
        ...

    @property
    def mode(self) -> _ModeStr:
        """The open mode the sound file was opened with."""
        ...

    @property
    def samplerate(self) -> int:
        """The sample rate of the sound file."""
        ...

    @property
    def frames(self) -> int:
        """The number of frames in the sound file."""
        ...

    @property
    def channels(self) -> int:
        """The number of channels in the sound file."""
        ...

    @property
    def format(self) -> _FormatStr:
        """The major format of the sound file."""
        ...

    @property
    def subtype(self) -> _SubtypeStr:
        """The subtype of data in the sound file."""
        ...

    @property
    def endian(self) -> _EndianStr:
        """The endian-ness of the data in the sound file."""
        ...

    @property
    def format_info(self) -> str:
        """A description of the major format of the sound file."""
        ...

    @property
    def subtype_info(self) -> str:
        """A description of the subtype of the sound file."""
        ...

    @property
    def sections(self) -> int:
        """The number of sections of the sound file."""
        ...

    @property
    def closed(self) -> bool:
        """Whether the sound file is closed or not."""
        ...

    @property
    def extra_info(self) -> str:
        """Retrieve the log string generated when opening the file."""
        ...

    @property
    def compression_level(self) -> Optional[float]:
        """The compression level on 'write()'"""
        ...

    @property
    def bitrate_mode(self) -> Optional[_BitrateModeStr]:
        """The bitrate mode on 'write()'"""
        ...

    # Metadata properties (dynamic, set via __getattr__ and __setattr__)
    @property
    def title(self) -> str: ...
    @title.setter
    def title(self, value: str) -> None: ...
    @property
    def copyright(self) -> str: ...
    @copyright.setter
    def copyright(self, value: str) -> None: ...
    @property
    def software(self) -> str: ...
    @software.setter
    def software(self, value: str) -> None: ...
    @property
    def artist(self) -> str: ...
    @artist.setter
    def artist(self, value: str) -> None: ...
    @property
    def comment(self) -> str: ...
    @comment.setter
    def comment(self, value: str) -> None: ...
    @property
    def date(self) -> str: ...
    @date.setter
    def date(self, value: str) -> None: ...
    @property
    def album(self) -> str: ...
    @album.setter
    def album(self, value: str) -> None: ...
    @property
    def license(self) -> str: ...
    @license.setter
    def license(self, value: str) -> None: ...
    @property
    def tracknumber(self) -> str: ...
    @tracknumber.setter
    def tracknumber(self, value: str) -> None: ...
    @property
    def genre(self) -> str: ...
    @genre.setter
    def genre(self, value: str) -> None: ...

    # File operations methods
    def __del__(self) -> None:
        """Close the file when the object is deleted."""
        ...

    def __enter__(self) -> "SoundFile":
        """Enter the context manager."""
        ...

    def __exit__(self, *args: Any) -> None:
        """Exit the context manager and close the file."""
        ...

    def __len__(self) -> int:
        """Return the number of frames. Deprecated."""
        ...

    def __bool__(self) -> bool:
        """Return True (to replace __len__)."""
        ...

    def seekable(self) -> bool:
        """Return True if the file supports seeking."""
        ...

    def seek(self, frames: int, whence: int = SEEK_SET) -> int:
        """Set the read/write position.

        Args:
            frames: The frame index or offset to seek.
            whence: By default (SEEK_SET), frames are counted from the beginning.
                   SEEK_CUR seeks from the current position.
                   SEEK_END seeks from the end.

        Returns:
            The new absolute read/write position in frames.
        """
        ...

    def tell(self) -> int:
        """Return the current read/write position."""
        ...

    # Read methods
    @overload
    def read(
        self,
        frames: int = -1,
        dtype: _DtypeStr = "float64",
        always_2d: bool = False,
        fill_value: Optional[float] = None,
        out: None = None,
    ) -> npt.NDArray[Any]: ...
    @overload
    def read(
        self,
        frames: int = -1,
        dtype: _DtypeStr = "float64",
        always_2d: bool = False,
        fill_value: Optional[float] = None,
        out: npt.NDArray[Any] = ...,
    ) -> npt.NDArray[Any]: ...
    def buffer_read(
        self, frames: int = -1, dtype: Optional[_DtypeStr] = None
    ) -> memoryview:
        """Read from the file and return data as buffer object.

        Args:
            frames: The number of frames to read. If negative, the whole rest of the file.
            dtype: Audio data will be converted to the given data type.

        Returns:
            A buffer containing the read data.
        """
        ...

    def buffer_read_into(self, buffer: bytearray, dtype: _DtypeStr) -> int:
        """Read from the file into a given buffer object.

        Args:
            buffer: Audio frames from the file are written to this buffer.
            dtype: The data type of buffer.

        Returns:
            The number of frames that were read from the file.
        """
        ...

    # Write methods
    def write(self, data: npt.ArrayLike) -> None:
        """Write audio data from a NumPy array to the file.

        Args:
            data: The data to write. Usually two-dimensional (frames x channels),
                 but one-dimensional data can be used for mono files.
        """
        ...

    def buffer_write(
        self, data: Union[bytes, bytearray, memoryview], dtype: _DtypeStr
    ) -> None:
        """Write audio data from a buffer/bytes object to the file.

        Args:
            data: A buffer or bytes object containing the audio data to be written.
            dtype: The data type of the audio data stored in data.
        """
        ...

    # Block methods
    def blocks(
        self,
        blocksize: Optional[int] = None,
        overlap: int = 0,
        frames: int = -1,
        dtype: _DtypeStr = "float64",
        always_2d: bool = False,
        fill_value: Optional[float] = None,
        out: Optional[npt.NDArray[Any]] = None,
    ) -> Generator[npt.NDArray[Any], None, None]:
        """Return a generator for block-wise reading.

        Args:
            blocksize: The number of frames to read per block.
            overlap: The number of frames to rewind between each block.
            frames: The number of frames to read.
            dtype: The data type to convert the audio to.
            always_2d: Always return 2D arrays, even for mono files.
            fill_value: Value to fill remaining frames with if EOF is reached.
            out: Write data to this array instead of creating a new one.

        Yields:
            Blocks of audio data.
        """
        ...

    # File modification methods
    def truncate(self, frames: Optional[int] = None) -> None:
        """Truncate the file to a given number of frames.

        Args:
            frames: Only the data before frames is kept, the rest is deleted.
                   If not specified, the current read/write position is used.
        """
        ...

    def flush(self) -> None:
        """Write unwritten data to the file system."""
        ...

    def close(self) -> None:
        """Close the file. Can be called multiple times."""
        ...

    def copy_metadata(self) -> dict[str, str]:
        """Get all metadata present in this SoundFile.

        Returns:
            A dict with all metadata. Possible keys are: 'title', 'copyright',
            'software', 'artist', 'comment', 'date', 'album', 'license',
            'tracknumber' and 'genre'.
        """
        ...

# Module-level functions
def read(
    file: Union[str, int, _FileLike],
    frames: int = -1,
    start: int = 0,
    stop: Optional[int] = None,
    dtype: _DtypeStr = "float64",
    always_2d: bool = False,
    fill_value: Optional[float] = None,
    out: Optional[npt.NDArray[Any]] = None,
    samplerate: Optional[int] = None,
    channels: Optional[int] = None,
    format: Optional[_FormatStr] = None,
    subtype: Optional[_SubtypeStr] = None,
    endian: Optional[_EndianStr] = None,
    closefd: bool = True,
) -> tuple[npt.NDArray[Any], int]:
    """Read audio data from a sound file and return data as NumPy array.

    Returns:
        A tuple containing the audio data and sample rate.
    """
    ...

def write(
    file: Union[str, int, _FileLike],
    data: npt.ArrayLike,
    samplerate: int,
    subtype: Optional[_SubtypeStr] = None,
    endian: Optional[_EndianStr] = None,
    format: Optional[_FormatStr] = None,
    closefd: bool = True,
    compression_level: Optional[float] = None,
    bitrate_mode: Optional[_BitrateModeStr] = None,
) -> None:
    """Write data to a sound file."""
    ...

def blocks(
    file: Union[str, int, _FileLike],
    blocksize: Optional[int] = None,
    overlap: int = 0,
    frames: int = -1,
    start: int = 0,
    stop: Optional[int] = None,
    dtype: _DtypeStr = "float64",
    always_2d: bool = False,
    fill_value: Optional[float] = None,
    out: Optional[npt.NDArray[Any]] = None,
    samplerate: Optional[int] = None,
    channels: Optional[int] = None,
    format: Optional[_FormatStr] = None,
    subtype: Optional[_SubtypeStr] = None,
    endian: Optional[_EndianStr] = None,
    closefd: bool = True,
) -> Generator[npt.NDArray[Any], None, None]:
    """Return a generator for block-wise reading."""
    ...

def available_formats() -> dict[str, str]:
    """Return a dictionary of available major formats."""
    ...

def available_subtypes(format: Optional[_FormatStr] = None) -> dict[str, str]:
    """Return a dictionary of available subtypes."""
    ...

def check_format(
    format: _FormatStr,
    subtype: Optional[_SubtypeStr] = None,
    endian: Optional[_EndianStr] = None,
) -> bool:
    """Check if the combination of format/subtype/endian is valid."""
    ...

def default_subtype(format: _FormatStr) -> _SubtypeStr:
    """Return the default subtype for a given format."""
    ...

def info(file: Union[str, int, _FileLike], verbose: bool = False) -> Any:
    """Return an object with information about a sound file."""
    ...
