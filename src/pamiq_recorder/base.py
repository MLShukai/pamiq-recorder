from abc import ABC, abstractmethod
from typing import Any, Self


class Recorder[T](ABC):
    """Abstract base class for data recording functionality.

    This class defines the interface for recording data of type T.
    Concrete implementations should specify the recording mechanism and
    handle proper resource management.
    """

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
