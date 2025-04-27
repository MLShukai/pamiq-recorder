from datetime import datetime
from pathlib import Path, PurePath
from typing import override

import pytest
from pytest_mock import MockerFixture

from pamiq_recorder.base import Recorder, RecordingWrapper, StrPath


class RecorderImpl(Recorder[str]):
    """Test implementation of Recorder."""

    def __init__(self, file_path: StrPath) -> None:
        super().__init__(file_path)
        self.recorded_data: list[str] = []

    @override
    def write(self, data: str) -> None:
        """Record the data by appending to internal list."""
        self.recorded_data.append(data)


class TestRecorder:
    """Test suite for the Recorder abstract base class."""

    @pytest.mark.parametrize("method", ["write"])
    def test_abstractmethod(self, method):
        """Verify that required methods are correctly marked as abstract."""
        assert method in Recorder.__abstractmethods__

    @pytest.fixture
    def recorder(self, tmp_path):
        """Provide a concrete implementation of Recorder for testing."""
        return RecorderImpl(tmp_path / "recorder")

    @pytest.mark.parametrize(
        "file_path", ["string/path", PurePath("pure/path"), Path("path")]
    )
    def test_init(self, file_path):
        recorder = RecorderImpl(file_path)
        assert isinstance(recorder.file_path, Path)
        assert recorder.file_path == Path(file_path)

    def test_del(self, recorder: RecorderImpl, mocker: MockerFixture):
        """Ensure the destructor properly calls the close method."""
        spy_close = mocker.spy(recorder, "close")
        recorder.__del__()
        spy_close.assert_called_once_with()

    def test_enter(self, recorder: RecorderImpl):
        """Ensure __enter__ returns self."""
        result = recorder.__enter__()
        assert result is recorder

    def test_exit(self, recorder: RecorderImpl, mocker: MockerFixture):
        """Ensure __exit__ properly calls the close method."""
        spy_close = mocker.spy(recorder, "close")
        recorder.__exit__(None, None, None)
        spy_close.assert_called_once_with()

    def test_context_manager(self, recorder: RecorderImpl, mocker: MockerFixture):
        """Ensure recorder can be used as a context manager."""
        spy_close = mocker.spy(recorder, "close")

        with recorder as r:
            assert r is recorder  # __enter__ returns self

        spy_close.assert_called_once_with()  # __exit__ calls close


class TestRecordingWrapper:
    """Test suite for the RecordingWrapper class."""

    @pytest.fixture
    def save_dir(self, tmp_path: Path) -> Path:
        """Create a temporary directory for saving recorded files."""
        return tmp_path / "recordings"

    @pytest.fixture
    def recorder_wrapper(self, save_dir: Path) -> RecordingWrapper[str]:
        """Create a wrapper instance for testing."""
        return RecordingWrapper(RecorderImpl, save_dir, "%Y%m%d-%H%M%S")

    def test_init(self, save_dir: Path):
        """Test wrapper initialization creates the save directory."""
        wrapper = RecordingWrapper(RecorderImpl, save_dir, "test-%Y%m%d")

        # Check directory was created
        assert save_dir.exists()
        assert save_dir.is_dir()

        # Check wrapper attributes
        assert wrapper.recorder_cls is RecorderImpl
        assert wrapper.save_dir == save_dir
        assert wrapper.name_format == "test-%Y%m%d"

    def test_create_recorder(
        self, recorder_wrapper: RecordingWrapper[str], mocker: MockerFixture
    ):
        """Test creating a new recorder instance with timestamped filename."""
        # Mock datetime to get a fixed timestamp
        mock_now = mocker.patch("pamiq_recorder.base.datetime")
        mock_now.now.return_value.strftime.return_value = "20250101-120000"

        recorder = recorder_wrapper.create_recorder()

        # Verify correct recorder was created with correct path
        assert isinstance(recorder, RecorderImpl)
        assert recorder.file_path == recorder_wrapper.save_dir / "20250101-120000"

        # Verify datetime was called correctly
        mock_now.now.assert_called_once()
        mock_now.now.return_value.strftime.assert_called_once_with("%Y%m%d-%H%M%S")

    def test_wrap(self, recorder_wrapper: RecordingWrapper[str]):
        """Test wrapping records data and passes it through."""
        # Setup the wrapper manually to create a recorder
        recorder_wrapper.setup()

        # Get the recorder instance for verification
        recorder = recorder_wrapper.recorder
        assert isinstance(recorder, RecorderImpl)

        # Test the wrap method
        result = recorder_wrapper.wrap("test data")

        # Check data was recorded
        assert "test data" in recorder.recorded_data

        # Check original data was returned unchanged
        assert result == "test data"

        # Cleanup
        recorder_wrapper.teardown()

    def test_lifecycle(
        self, recorder_wrapper: RecordingWrapper[str], mocker: MockerFixture
    ):
        """Test full lifecycle: setup, pause, resume, teardown."""
        # Spy on the recorder's close method
        close_spy = mocker.spy(RecorderImpl, "close")

        # Setup
        recorder_wrapper.setup()
        recorder1 = recorder_wrapper.recorder
        assert isinstance(recorder1, RecorderImpl)

        # Record some data
        recorder_wrapper.wrap("data1")
        assert "data1" in recorder1.recorded_data

        # Pause
        recorder_wrapper.on_paused()
        assert close_spy.call_count == 1

        # Resume
        recorder_wrapper.on_resumed()
        recorder2 = recorder_wrapper.recorder
        assert isinstance(recorder2, RecorderImpl)
        assert recorder2 is not recorder1  # Should be a new instance

        # Record more data in new recorder
        recorder_wrapper.wrap("data2")
        assert "data2" in recorder2.recorded_data
        assert "data1" not in recorder2.recorded_data  # Separate instance

        # Teardown
        recorder_wrapper.teardown()
        assert close_spy.call_count == 2

    def test_to_wrapper_static_method(self, tmp_path: Path):
        """Test the Recorder.to_wrapper class method."""
        save_dir = tmp_path / "wrapper_test"

        # Use the class method to create a wrapper
        wrapper = RecorderImpl.to_wrapper(save_dir, "file-%Y%m%d")

        # Check the wrapper was created correctly
        assert isinstance(wrapper, RecordingWrapper)
        assert wrapper.recorder_cls is RecorderImpl
        assert wrapper.save_dir == save_dir
        assert wrapper.name_format == "file-%Y%m%d"
        assert save_dir.exists()

    def test_to_wrapper_with_args(self, tmp_path: Path):
        """Test to_wrapper with additional constructor arguments."""

        # Create a subclass that needs extra args
        class CustomRecorder(RecorderImpl):
            def __init__(
                self, file_path: StrPath, extra_arg: str, extra_kwarg: int = 0
            ):
                super().__init__(file_path)
                self.extra_arg = extra_arg
                self.extra_kwarg = extra_kwarg

        save_dir = tmp_path / "custom_test"
        wrapper = CustomRecorder.to_wrapper(
            save_dir,
            "custom-%Y%m%d",
            "extra_value",  # Extra positional arg
            extra_kwarg=42,  # Extra keyword arg
        )

        # Setup to create the recorder
        wrapper.setup()
        recorder = wrapper.recorder

        # Check extra args were passed correctly
        assert isinstance(recorder, CustomRecorder)
        assert recorder.extra_arg == "extra_value"
        assert recorder.extra_kwarg == 42

        # Cleanup
        wrapper.teardown()

    def test_default_file_name_format(self, tmp_path: Path):
        """Test the default file name format follows expected pattern."""
        # Create a wrapper with default format
        wrapper = RecorderImpl.to_wrapper(tmp_path)
        wrapper.setup()

        # Check that file_path has correct format
        file_name = wrapper.recorder.file_path.name
        # Should match pattern like: 2025-01-01_12-34-56,123456
        assert len(file_name) > 0

        # Attempt to parse with datetime to verify format is valid
        try:
            datetime.strptime(file_name, Recorder.DEFAULT_FILE_NAME_FORMAT)
            valid_format = True
        except ValueError:
            valid_format = False

        assert valid_format, f"File name {file_name} does not match expected format"

        # Cleanup
        wrapper.teardown()
