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
