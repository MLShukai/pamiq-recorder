"""Tests for the audio recorder module."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pamiq_recorder.audio import AudioRecorder


class TestAudioRecorder:
    """Test suite for the AudioRecorder class."""

    def test_default_file_name_format(self):
        assert Path(AudioRecorder.DEFAULT_FILE_NAME_FORMAT).suffix == ".ogg"

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

        # Check that the audio file exists
        assert audio_path.exists(), "Audio file was not created"

        # Check that recorder attributes are set correctly
        assert recorder.file_path == audio_path
        assert recorder.sample_rate == 44100
        assert recorder.channels == 2
        # Clean up resources
        recorder.close()

    def test_init_invalid_extension(self, tmp_path: Path):
        """Test initialization fails with unsupported file extension."""
        invalid_path = tmp_path / "test_audio.xyz"

        with pytest.raises(ValueError, match="Audio format 'xyz' is not supported"):
            AudioRecorder(file_path=invalid_path, sample_rate=44100, channels=2)

    def test_init_with_custom_subtype(self, audio_path: Path):
        """Test initialization with custom subtype."""
        recorder = AudioRecorder(
            file_path=audio_path, sample_rate=44100, channels=2, subtype="FLOAT"
        )
        # Check that the audio file exists
        assert audio_path.exists(), "Audio file was not created"

        recorder.close()

    def test_init_invalid_subtype(self, audio_path: Path):
        """Test initialization fails with invalid subtype."""
        with pytest.raises(
            ValueError, match="Specified subtype 'INVALID_SUBTYPE' is invalid"
        ):
            AudioRecorder(
                file_path=audio_path,
                sample_rate=44100,
                channels=2,
                subtype="INVALID_SUBTYPE",  # type: ignore
            )

    def test_write_mono_data(self, audio_path: Path):
        """Test writing mono audio data."""
        recorder = AudioRecorder(file_path=audio_path, sample_rate=44100, channels=1)

        # Create a simple sine wave (mono)
        duration = 0.1  # seconds
        t = np.linspace(0, duration, int(duration * 44100), endpoint=False)
        mono_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # 440 Hz tone

        # Reshape to 2D array (samples, channels)
        data = mono_data.reshape(-1, 1)

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
        assert np.allclose(audio[:10], data[:10, 0], atol=1e-3)

    def test_strict_dimension_check(self, audio_path: Path):
        """Test that 1D arrays are rejected."""
        recorder = AudioRecorder(file_path=audio_path, sample_rate=44100, channels=1)

        # Create a 1D array
        data = np.zeros(1000, dtype=np.float32)

        # Should raise ValueError for 1D array
        with pytest.raises(ValueError, match="Expected 2D array with shape"):
            recorder.write(data)

        # Create a 3D array
        data_3d = np.zeros((100, 1, 2), dtype=np.float32)

        # Should raise ValueError for 3D array
        with pytest.raises(ValueError, match="Expected 2D array with shape"):
            recorder.write(data_3d)
        recorder.close()

    def test_write_stereo_data(self, audio_path: Path):
        """Test writing stereo audio data."""
        recorder = AudioRecorder(file_path=audio_path, sample_rate=44100, channels=2)

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
        # Clean up resources if not already closed
        recorder.close()

    def test_write_invalid_dimensions(self, audio_path: Path):
        """Test write method rejects data with wrong dimensions."""
        recorder = AudioRecorder(file_path=audio_path, sample_rate=44100, channels=2)

        # Wrong number of dimensions
        with pytest.raises(
            ValueError, match=r"Expected 2D array with shape \(samples, channels\)"
        ):
            recorder.write(np.zeros((10, 10, 10), dtype=np.float32))

        # Wrong channels
        with pytest.raises(ValueError, match="Expected 2 channels"):
            recorder.write(np.zeros((100, 3), dtype=np.float32))
        recorder.close()

    def test_multiple_writes(self, audio_path: Path):
        """Test writing multiple chunks of audio data."""
        recorder = AudioRecorder(file_path=audio_path, sample_rate=44100, channels=2)

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
        "format_extension",
        [
            "wav",
            "flac",
            "ogg",
            "opus",
            "m4a",
            "mov",
            "alac",
            "mp3",
            "aiff",
            "au",
            "w64",
        ],
    )
    def test_audio_formats(self, tmp_path: Path, sample_audio_data, format_extension):
        """Test writing to different audio formats."""
        data, sample_rate = sample_audio_data
        audio_path = tmp_path / f"test_audio.{format_extension}"

        # Create recorder
        recorder = AudioRecorder(
            file_path=audio_path, sample_rate=sample_rate, channels=2
        )

        # Write the audio data
        recorder.write(data)
        recorder.close()

        # Check file exists and has content
        assert audio_path.exists(), f"{format_extension} file was not created"
        assert audio_path.stat().st_size > 0, f"{format_extension} file is empty"

        recorder.close()

    @pytest.mark.parametrize(
        "format_extension,subtype",
        [
            ("wav", "PCM_16"),
            ("wav", "PCM_24"),
            ("wav", "FLOAT"),
            ("flac", "PCM_16"),
            ("ogg", "VORBIS"),
        ],
    )
    def test_audio_subtypes(
        self, tmp_path: Path, sample_audio_data, format_extension, subtype
    ):
        """Test writing to different audio formats with specific subtypes."""
        data, sample_rate = sample_audio_data
        audio_path = tmp_path / f"test_audio.{format_extension}"

        # Create recorder with specific subtype
        recorder = AudioRecorder(
            file_path=audio_path, sample_rate=sample_rate, channels=2, subtype=subtype
        )

        # Write the audio data
        recorder.write(data)
        recorder.close()

        # Check file exists and has content
        assert (
            audio_path.exists()
        ), f"{format_extension} file with subtype {subtype} was not created"
        assert (
            audio_path.stat().st_size > 0
        ), f"{format_extension} file with subtype {subtype} is empty"

        recorder.close()
