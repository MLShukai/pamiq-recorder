from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path, PurePath
from typing import Any, ClassVar, Self, override

from pamiq_core.interaction.wrappers import Wrapper

type StrPath = PurePath | str


class Recorder[T](ABC):
    """Abstract base class for data recording functionality.

    This class defines the interface for recording data of type T.
    Concrete implementations should specify the recording mechanism and
    handle proper resource management.

    Attributes:
        file_path: Path to the output file where data will be recorded.
                  Input paths are converted to pathlib.Path objects.
        DEFAULT_FILE_NAME_FORMAT: Default datetime format string for automatic file naming.
            Format follows the Python datetime strftime() syntax.
    """

    DEFAULT_FILE_NAME_FORMAT: ClassVar[str] = "%Y-%m-%d_%H-%M-%S,%f"

    def __init__(self, file_path: StrPath) -> None:
        """Initialize the recorder with a file path.

        Args:
            file_path: Path to the output file. Can be a string or a Path-like object.
        """
        self.file_path = Path(file_path)

    @abstractmethod
    def write(self, data: T) -> None:
        """Write data to the recorder.

        Args:
            data: The data to be recorded.
        """
        ...

    def close(self) -> None:
        """Close the recorder and release any resources."""
        ...

    def __del__(self) -> None:
        """Destructor that ensures resources are properly released."""
        self.close()

    def __enter__(self) -> Self:
        """Enter the context manager protocol.

        Returns:
            The recorder instance itself.
        """
        return self

    def __exit__(self, *args: Any, **kwds: Any) -> None:
        """Exit the context manager protocol.

        This method ensures that resources are properly released by calling close().

        Args:
            *args: Exception details if an exception was raised.
            **kwds: Additional keyword arguments.
        """
        self.close()

    @classmethod
    def to_wrapper(
        cls,
        save_dir: StrPath,
        name_format: str = DEFAULT_FILE_NAME_FORMAT,
        *init_args: Any,
        **init_kwds: Any,
    ) -> RecordingWrapper[T]:
        """Create a RecordingWrapper around this recorder class.

        This is a convenience method to easily create a wrapper that records
        data while passing it through unchanged.

        Args:
            save_dir: Directory where recorded files will be saved.
            name_format: Format string for file names, defaults to DEFAULT_FILE_NAME_FORMAT.
            *init_args: Positional arguments to pass to the recorder constructor.
            **init_kwds: Keyword arguments to pass to the recorder constructor.

        Returns:
            A RecordingWrapper instance that will create recorders of this class.
        """
        return RecordingWrapper(cls, save_dir, name_format, *init_args, **init_kwds)


class RecordingWrapper[T](Wrapper[T, T]):
    """A wrapper that records data while passing it through unchanged.

    This wrapper creates a recorder on setup and automatically manages
    its lifecycle based on system events (paused/resumed).

    Each time the wrapper is set up or resumed, a new recorder is
    created with a timestamped filename in the specified directory.
    """

    recorder: Recorder[T]

    def __init__(
        self,
        recorder_cls: type[Recorder[T]],
        save_dir: StrPath,
        name_format: str,
        *recorder_init_args: Any,
        **recorder_init_kwds: Any,
    ) -> None:
        """Initialize the recording wrapper.

        Args:
            recorder_cls: The recorder class to instantiate.
            save_dir: Directory where recorded files will be saved.
            name_format: Format string for file names.
            *recorder_init_args: Positional arguments to pass to the recorder constructor.
            **recorder_init_kwds: Keyword arguments to pass to the recorder constructor.
        """
        super().__init__()
        self.recorder_cls = recorder_cls
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.name_format = name_format
        self._args = recorder_init_args
        self._kwds = recorder_init_kwds

    def create_recorder(self) -> Recorder[T]:
        """Create a new recorder instance with a timestamped filename.

        Creates a new recorder with the current timestamp as the filename,
        using the format specified during initialization.

        Returns:
            A new recorder instance ready to record data.
        """
        file_path = self.save_dir / datetime.now().strftime(self.name_format)
        return self.recorder_cls(file_path, *self._args, **self._kwds)

    @override
    def wrap(self, value: T) -> T:
        """Record the data and pass it through unchanged.

        Args:
            value: The data to record.

        Returns:
            The same data, unmodified.
        """
        self.recorder.write(value)
        return value

    @override
    def setup(self) -> None:
        """Set up the wrapper by creating a new recorder instance."""
        super().setup()
        self.recorder = self.create_recorder()

    @override
    def teardown(self) -> None:
        """Clean up resources by closing the recorder."""
        super().teardown()
        self.recorder.close()

    @override
    def on_paused(self) -> None:
        """Handle system pause event by closing the current recorder."""
        super().on_paused()
        self.recorder.close()

    @override
    def on_resumed(self) -> None:
        """Handle system resume event by creating a new recorder."""
        super().on_resumed()
        self.recorder = self.create_recorder()
