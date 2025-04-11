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
    def recorder(self):
        """Provide a concrete implementation of Recorder for testing."""
        return RecorderImpl()

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
