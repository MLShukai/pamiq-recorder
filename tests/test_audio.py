"""Tests for the audio recorder module."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pamiq_recorder.audio import AudioRecorder


class TestAudioRecorder:
    """Test suite for the AudioRecorder class."""

    @pytest.fixture
    def audio_path(self, tmp_path: Path) -> Path:
        """Create a temporary path for the audio file."""
        return tmp_path / "test_audio.wav"

    @pytest.fixture
    def sample_audio_data(self) -> tuple[np.ndarray, int]:
        """Generate sample audio data for testing.

        Returns:
            Tuple containing (audio_data, sample_rate)
        """
        # Create a simple stereo sine wave
        sample_rate = 48000
        duration = 0.1  # seconds
        t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)

        # Left channel: 440 Hz, Right channel: 880 Hz
        left = np.sin(2 * np.pi * 440 * t)
        right = np.sin(2 * np.pi * 880 * t)

        # Combine into stereo data with shape (samples, channels)
        data = np.column_stack((left, right)).astype(np.float32)

        return data, sample_rate

    def test_init_and_file_creation(self, audio_path: Path):
        """Test recorder initialization creates a file."""
        recorder = AudioRecorder(file_path=audio_path, sample_rate=44100, channels=2)

        try:
            # Check that the audio file exists
            assert audio_path.exists(), "Audio file was not created"

            # Check that recorder attributes are set correctly
            assert recorder.file_path == audio_path
            assert recorder.sample_rate == 44100
            assert recorder.channels == 2
        finally:
            # Clean up resources
            recorder.close()

    def test_init_invalid_extension(self, tmp_path: Path):
        """Test initialization fails with unsupported file extension."""
        invalid_path = tmp_path / "test_audio.xyz"

        with pytest.raises(ValueError, match="Audio format 'xyz' is not supported"):
            AudioRecorder(file_path=invalid_path, sample_rate=44100, channels=2)

    def test_write_mono_data(self, audio_path: Path):
        """Test writing mono audio data."""
        recorder = AudioRecorder(file_path=audio_path, sample_rate=44100, channels=1)

        try:
            # Create a simple sine wave (mono)
            duration = 0.1  # seconds
            t = np.linspace(0, duration, int(duration * 44100), endpoint=False)
            data = np.sin(2 * 3.14 * 440 * t).astype(np.float32)  # 440 Hz tone

            # Write the audio data
            recorder.write(data)
            recorder.close()

            # Check file size has increased
            file_size = audio_path.stat().st_size
            assert file_size > 0, "Audio file is empty after writing data"

            # Verify the content using soundfile
            audio, sample_rate = sf.read(str(audio_path))
            assert sample_rate == 44100
            assert len(audio) == len(data)
            # Compare a few samples (not exact due to encoding)
            assert np.allclose(audio[:10], data[:10], atol=1e-3)
        finally:
            recorder.close()

    def test_write_stereo_data(self, audio_path: Path):
        """Test writing stereo audio data."""
        recorder = AudioRecorder(file_path=audio_path, sample_rate=44100, channels=2)

        try:
            # Create a simple stereo sine wave
            duration = 0.1  # seconds
            t = np.linspace(0, duration, int(duration * 44100), endpoint=False)

            # Left channel: 440 Hz, Right channel: 880 Hz
            left = np.sin(2 * np.pi * 440 * t)
            right = np.sin(2 * np.pi * 880 * t)

            # Combine into stereo data with shape (samples, channels)
            data = np.column_stack((left, right)).astype(np.float32)

            # Write the audio data
            recorder.write(data)
            recorder.close()

            # Check file size has increased
            file_size = audio_path.stat().st_size
            assert file_size > 0, "Audio file is empty after writing data"

            # Verify the content using soundfile
            audio, sample_rate = sf.read(str(audio_path))
            assert sample_rate == 44100
            assert audio.shape == data.shape
            # Compare a few samples (not exact due to encoding)
            assert np.allclose(audio[:10], data[:10], atol=1e-3)
        finally:
            # Clean up resources if not already closed
            recorder.close()

    def test_write_invalid_dimensions(self, audio_path: Path):
        """Test write method rejects data with wrong dimensions."""
        recorder = AudioRecorder(file_path=audio_path, sample_rate=44100, channels=2)

        try:
            # Wrong number of dimensions
            with pytest.raises(ValueError, match="Expected 1D or 2D array"):
                recorder.write(np.zeros((10, 10, 10), dtype=np.float32))

            # Wrong channels
            with pytest.raises(ValueError, match="Expected 2 channels"):
                recorder.write(np.zeros((100, 3), dtype=np.float32))
        finally:
            recorder.close()

    def test_multiple_writes(self, audio_path: Path):
        """Test writing multiple chunks of audio data."""
        recorder = AudioRecorder(file_path=audio_path, sample_rate=44100, channels=2)

        try:
            # Create a simple stereo data chunk
            chunk = np.zeros((1000, 2), dtype=np.float32)
            chunk[:, 0] = 0.5  # Left channel
            chunk[:, 1] = -0.5  # Right channel

            # Write multiple chunks
            total_samples = 0
            for _ in range(5):
                recorder.write(chunk)
                total_samples += len(chunk)

            recorder.close()

            # Verify the total content length
            audio, _ = sf.read(str(audio_path))
            assert (
                len(audio) == total_samples
            ), "Audio file length doesn't match total written samples"
        finally:
            recorder.close()

    def test_close_and_reopen(self, audio_path: Path):
        """Test closing and reopening an audio file."""
        # Create and close a recorder
        recorder = AudioRecorder(file_path=audio_path, sample_rate=44100, channels=2)

        data = np.zeros((1000, 2), dtype=np.float32)
        recorder.write(data)
        recorder.close()

        # Check that the file exists
        assert audio_path.exists(), "Audio file does not exist after closing"

        # Check that writing after close raises an error
        with pytest.raises(RuntimeError, match="Recorder is already closed"):
            recorder.write(data)

    @pytest.mark.parametrize(
        "format_extension", ["wav", "flac", "ogg", "opus", "m4a", "mov", "alac"]
    )
    def test_audio_formats(self, tmp_path: Path, sample_audio_data, format_extension):
        """Test writing to different audio formats."""
        data, sample_rate = sample_audio_data
        audio_path = tmp_path / f"test_audio.{format_extension}"

        # Skip test if format is not supported by the current soundfile installation
        recorder = AudioRecorder(
            file_path=audio_path, sample_rate=sample_rate, channels=2
        )

        try:
            # Write the audio data
            recorder.write(data)
            recorder.close()

            # Check file exists and has content
            assert audio_path.exists(), f"{format_extension} file was not created"
            assert audio_path.stat().st_size > 0, f"{format_extension} file is empty"

        finally:
            recorder.close()
