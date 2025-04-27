from pathlib import Path, PurePath
from typing import Any, override

import pytest
from pytest_mock import MockerFixture

from pamiq_recorder.base import Recorder


class RecorderImpl(Recorder[Any]):
    @override
    def write(self, data: Any):
        pass


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
