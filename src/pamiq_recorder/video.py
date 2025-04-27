"""Video recording module using OpenCV."""

from typing import ClassVar, Literal, override

import cv2
import numpy as np
import numpy.typing as npt

from .base import Recorder, StrPath


class VideoRecorder(Recorder[npt.NDArray[np.uint8]]):
    """Records video frames to a file using OpenCV.

    Supports grayscale, RGB, and RGBA input formats, automatically
    converting from RGB/RGBA to BGR/BGRA for OpenCV compatibility.
    """

    DEFAULT_FILE_NAME_FORMAT: ClassVar[str] = Recorder.DEFAULT_FILE_NAME_FORMAT + ".mp4"

    @override
    def __init__(
        self,
        file_path: StrPath,
        fps: float,
        height: int,
        width: int,
        channels: Literal[1, 3, 4],
    ) -> None:
        """Initialize a video recorder.

        Args:
            file_path: Path to save the video file. File extension determines format.
            fps: Frames per second.
            height: Frame height in pixels.
            width: Frame width in pixels.
            channels: Number of color channels (1 for grayscale, 3 for RGB, 4 for RGBA).

        Raises:
            ValueError: If channels is not 1, 3, or 4.
            RuntimeError: If the video writer fails to initialize.
        """
        super().__init__(file_path)
        self.fps = fps
        self.height = height
        self.width = width
        self.channels = channels

        # Validate channels
        if channels not in [1, 3, 4]:
            raise ValueError(
                f"Channels must be 1 (grayscale), 3 (RGB), or 4 (RGBA), got {channels}"
            )

        # Determine codec based on file extension
        fourcc = self._get_fourcc_from_extension(self.file_path.suffix)

        # Initialize VideoWriter with color flag based on channels
        self._writer = cv2.VideoWriter(
            str(self.file_path),
            fourcc,
            fps,
            (width, height),
            channels > 1,  # is_color flag
        )

        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for file: {file_path}")

    def _get_fourcc_from_extension(self, extension: str) -> int:
        """Get the appropriate FourCC code based on file extension.

        Args:
            extension: File extension including the dot (e.g., '.mp4').

        Returns:
            FourCC code as an integer.
        """
        # Remove the leading dot if present
        ext = extension.lower().lstrip(".")

        # Map extensions to FourCC codes
        match ext:
            case "mp4":
                fourcc_str = "mp4v"
            case "avi":
                fourcc_str = "XVID"
            case "mov":
                fourcc_str = "mp4v"
            case "mkv":
                fourcc_str = "X264"
            case _:
                raise ValueError(f"Video format '{ext}' is not supported.")

        return cv2.VideoWriter.fourcc(*fourcc_str)

    @override
    def write(self, data: npt.NDArray[np.uint8]) -> None:
        """Write a frame to the video.

        Args:
            data: Image data as numpy array with shape (height, width, channels).

        Raises:
            ValueError: If data shape doesn't match expected dimensions.
            RuntimeError: If the recorder is already closed.
        """
        if not self._writer.isOpened():
            raise RuntimeError("Recorder is already closed.")

        # Validate dimensions - strictly require 3D array
        if data.ndim != 3:
            raise ValueError(
                f"Expected 3D array with shape (height, width, channels), got {data.ndim}D array with shape {data.shape}"
            )

        # Extract dimensions
        actual_height, actual_width, actual_channels = data.shape

        # Validate channel count
        if actual_channels != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, got {actual_channels}"
            )

        # Validate height and width
        if actual_height != self.height or actual_width != self.width:
            raise ValueError(
                f"Expected shape ({self.height}, {self.width}, {self.channels}), "
                f"got {data.shape}"
            )

        # Convert to uint8
        frame = np.asarray(data, dtype=np.uint8)

        # Convert color format if needed
        match self.channels:
            case 3:  # RGB to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            case 4:  # RGBA to BGRA
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
            case 1:  # Grayscale
                pass
            case _:
                raise ValueError(f"Unexpected {self.channels} channels.")

        # Write the frame
        self._writer.write(frame)

    @override
    def close(self) -> None:
        """Close the video writer and release resources."""
        if hasattr(self, "_writer") and self._writer.isOpened():
            self._writer.release()
