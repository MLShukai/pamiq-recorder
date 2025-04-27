"""Tests for the video recorder module."""

from pathlib import Path

import numpy as np
import pytest

from pamiq_recorder.video import VideoRecorder


class TestVideoRecorder:
    """Test suite for the VideoRecorder class."""

    def test_default_file_name_format(self):
        assert Path(VideoRecorder.DEFAULT_FILE_NAME_FORMAT).suffix == ".mp4"

    @pytest.fixture
    def video_path(self, tmp_path: Path) -> Path:
        """Create a temporary path for the video file."""
        return tmp_path / "test_video.mp4"

    def test_init_and_file_creation(self, video_path: Path):
        """Test recorder initialization creates a file."""
        recorder = VideoRecorder(
            file_path=video_path, fps=30.0, height=480, width=640, channels=3
        )

        try:
            # Check that the video file exists
            assert video_path.exists(), "Video file was not created"

            # Check that recorder attributes are set correctly
            assert recorder.file_path == video_path
            assert recorder.fps == 30.0
            assert recorder.height == 480
            assert recorder.width == 640
            assert recorder.channels == 3
        finally:
            # Clean up resources
            recorder.close()

    def test_init_invalid_channels(self, video_path: Path):
        """Test initialization fails with invalid channel count."""
        # Attempt to use invalid channels value using type ignore to bypass static typing
        with pytest.raises(
            ValueError, match="Channels must be 1 .grayscale., 3 .RGB., or 4 .RGBA."
        ):
            VideoRecorder(
                file_path=video_path,
                fps=30.0,
                height=480,
                width=640,
                channels=2,  # type: ignore
            )

    def test_init_invalid_extension(self, tmp_path: Path):
        """Test initialization fails with unsupported file extension."""
        invalid_path = tmp_path / "test_video.xyz"

        with pytest.raises(ValueError, match="Video format 'xyz' is not supported"):
            VideoRecorder(
                file_path=invalid_path, fps=30.0, height=480, width=640, channels=3
            )

    def test_write_rgb_frame(self, video_path: Path):
        """Test writing an RGB frame."""
        recorder = VideoRecorder(
            file_path=video_path, fps=30.0, height=100, width=100, channels=3
        )

        try:
            # Create a simple RGB gradient frame
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            for i in range(100):
                for j in range(100):
                    frame[i, j, 0] = i * 255 // 100  # Red gradient
                    frame[i, j, 1] = j * 255 // 100  # Green gradient
                    frame[i, j, 2] = (i + j) * 255 // 200  # Blue gradient

            # Write the frame
            recorder.write(frame)

            # Check file size has increased
            file_size = video_path.stat().st_size
            assert file_size > 0, "Video file is empty after writing a frame"
        finally:
            # Clean up resources
            recorder.close()

    def test_write_grayscale_frame(self, tmp_path: Path):
        """Test writing a grayscale frame."""
        video_path = tmp_path / "grayscale.mp4"
        recorder = VideoRecorder(
            file_path=video_path, fps=30.0, height=100, width=100, channels=1
        )

        try:
            # Create a simple gradient frame - now as 3D array with shape (height, width, 1)
            frame_2d = np.zeros((100, 100), dtype=np.uint8)
            for i in range(100):
                for j in range(100):
                    frame_2d[i, j] = (i + j) * 255 // 200  # Diagonal gradient

            # Reshape to 3D with explicit channel dimension
            frame = frame_2d.reshape(100, 100, 1)

            # Write the frame
            recorder.write(frame)

            # Check file size has increased
            file_size = video_path.stat().st_size
            assert file_size > 0, "Video file is empty after writing a frame"
        finally:
            # Clean up resources
            recorder.close()

    def test_strict_dimension_check(self, video_path: Path):
        """Test that 2D and 4D arrays are rejected."""
        recorder = VideoRecorder(
            file_path=video_path, fps=30.0, height=100, width=100, channels=3
        )

        try:
            # Create a 2D array
            frame_2d = np.zeros((100, 100), dtype=np.uint8)

            # Should raise ValueError for 2D array
            with pytest.raises(ValueError, match="Expected 3D array with shape"):
                recorder.write(frame_2d)

            # Create a 4D array
            frame_4d = np.zeros((1, 100, 100, 3), dtype=np.uint8)

            # Should raise ValueError for 4D array
            with pytest.raises(ValueError, match="Expected 3D array with shape"):
                recorder.write(frame_4d)
        finally:
            recorder.close()

    def test_write_rgba_frame(self, tmp_path: Path):
        """Test writing an RGBA frame."""
        video_path = tmp_path / "rgba.mp4"
        recorder = VideoRecorder(
            file_path=video_path, fps=30.0, height=100, width=100, channels=4
        )

        try:
            # Create a simple RGBA gradient frame with transparency
            frame = np.zeros((100, 100, 4), dtype=np.uint8)
            for i in range(100):
                for j in range(100):
                    frame[i, j, 0] = i * 255 // 100  # Red gradient
                    frame[i, j, 1] = j * 255 // 100  # Green gradient
                    frame[i, j, 2] = (i + j) * 255 // 200  # Blue gradient
                    frame[i, j, 3] = 255 - (i * 255 // 100)  # Alpha gradient

            # Write the frame
            recorder.write(frame)

            # Check file size has increased
            file_size = video_path.stat().st_size
            assert file_size > 0, "Video file is empty after writing a frame"
        finally:
            # Clean up resources
            recorder.close()

    def test_write_invalid_dimensions(self, video_path: Path):
        """Test write method rejects frames with wrong dimensions."""
        recorder = VideoRecorder(
            file_path=video_path, fps=30.0, height=480, width=640, channels=3
        )

        try:
            # Wrong number of dimensions
            with pytest.raises(
                ValueError,
                match=r"Expected 3D array with shape \(height, width, channels\)",
            ):
                recorder.write(np.zeros((10, 10, 10, 10), dtype=np.uint8))

            # Wrong height/width
            with pytest.raises(ValueError, match="Expected shape"):
                recorder.write(np.zeros((240, 320, 3), dtype=np.uint8))

            # Wrong channels
            with pytest.raises(ValueError, match="Expected 3 channels"):
                recorder.write(np.zeros((480, 640, 4), dtype=np.uint8))
        finally:
            recorder.close()

    def test_multiple_frames(self, video_path: Path):
        """Test writing multiple frames creates a larger file."""
        recorder = VideoRecorder(
            file_path=video_path, fps=30.0, height=100, width=100, channels=3
        )

        try:
            # Create a simple frame
            frame = np.zeros((100, 100, 3), dtype=np.uint8)

            # Write the first frame
            recorder.write(frame)
            size_after_one_frame = video_path.stat().st_size

            # Write a few more frames
            for i in range(10):
                # Update the frame with different content each time
                frame.fill(i * 20)  # Different color for each frame
                recorder.write(frame)

            recorder.close()
            # Check that file size has increased with multiple frames
            size_after_multiple_frames = video_path.stat().st_size
            assert (
                size_after_multiple_frames > size_after_one_frame
            ), "Video file size did not increase after writing multiple frames"
        finally:
            recorder.close()

    def test_close_and_reopen(self, video_path: Path):
        """Test closing and reopening a video file."""
        # Create and close a recorder
        recorder = VideoRecorder(
            file_path=video_path, fps=30.0, height=100, width=100, channels=3
        )

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        recorder.write(frame)
        recorder.close()

        # Check that the file exists
        assert video_path.exists(), "Video file does not exist after closing"

        # Check that writing after close raises an error
        with pytest.raises(RuntimeError, match="Recorder is already closed"):
            recorder.write(frame)
