from abc import ABC, abstractmethod


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
