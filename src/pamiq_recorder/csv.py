"""CSV recording module using the built-in csv module."""

import csv
import time
from collections.abc import Iterable
from typing import Any, ClassVar, override

from .base import Recorder, StrPath


class CsvRecorder(Recorder[Iterable[Any]]):
    """Records data as CSV rows to a file.

    Automatically adds a timestamp column to each row. The timestamp is
    recorded as Unix time (seconds since epoch).
    """

    DEFAULT_FILE_NAME_FORMAT: ClassVar[str] = Recorder.DEFAULT_FILE_NAME_FORMAT + ".csv"

    @override
    def __init__(
        self,
        file_path: StrPath,
        headers: Iterable[str],
        timestamp_header: str = "timestamp",
    ) -> None:
        """Initialize a CSV recorder.

        Args:
            file_path: Path to save the CSV file.
            headers: Column headers excluding the timestamp column.
            timestamp_header: Name of the timestamp column (default: "timestamp").

        Raises:
            ValueError: If timestamp_header conflicts with any user-provided header.
        """
        super().__init__(file_path)
        self.headers = list(headers)
        self.timestamp_header = timestamp_header

        # Check if timestamp header conflicts with user headers
        if timestamp_header in self.headers:
            raise ValueError(
                f"Timestamp header '{timestamp_header}' conflicts with a user-provided header"
            )

        # Check for duplicate column names in user headers
        header_set: set[str] = set()
        duplicates: list[str] = []

        for header in self.headers:
            if header in header_set:
                duplicates.append(header)
            else:
                header_set.add(header)

        if len(duplicates) > 0:
            raise ValueError(
                f"Duplicate column names found in headers: {', '.join(duplicates)}"
            )

        # Create or open the file and write the headers
        self._file = open(self.file_path, "w", newline="")
        self._writer = csv.writer(self._file)

        # Write the header row (timestamp + user headers)
        self._writer.writerow([timestamp_header] + self.headers)
        self._file.flush()

    @override
    def write(self, data: Iterable[Any]) -> None:
        """Write a row of data to the CSV file.

        Automatically prepends the current timestamp (Unix time) to the row.

        Args:
            data: Iterable of values corresponding to the headers provided at initialization.

        Raises:
            ValueError: If the number of values doesn't match the number of headers.
            RuntimeError: If the recorder is already closed.
        """
        if self._file.closed:
            raise RuntimeError("Recorder is already closed.")

        # Convert to list to get length and for writing
        data_list = list(data)

        # Verify data length matches headers length
        if len(data_list) != len(self.headers):
            raise ValueError(
                f"Expected {len(self.headers)} values, got {len(data_list)}"
            )

        # Get current timestamp
        timestamp = time.time()

        # Write the row with timestamp
        self._writer.writerow([timestamp] + data_list)

        # Ensure data is written to disk
        self._file.flush()

    @override
    def close(self) -> None:
        """Close the CSV file and release resources."""
        if hasattr(self, "_file") and not self._file.closed:
            self._file.close()
