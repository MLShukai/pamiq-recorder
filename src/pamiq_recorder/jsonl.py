"""JSON Lines recording module for saving timestamped JSON data."""

import json
import time
from typing import Any, ClassVar, override

from .base import Recorder, StrPath


class JsonlRecorder(Recorder[Any]):
    """Records data as JSON Lines (LDJSON) entries to a file.

    Each entry is written as a single line containing a JSON object with
    a timestamp and the provided data.

    Format:
        {"timestamp": 1619855734.123, "data": {...}}

    This format makes it easy to stream and process large datasets line by line.
    """

    DEFAULT_FILE_NAME_FORMAT: ClassVar[str] = (
        Recorder.DEFAULT_FILE_NAME_FORMAT + ".jsonl"
    )

    @override
    def __init__(
        self,
        file_path: StrPath,
        ensure_ascii: bool = True,
    ) -> None:
        """Initialize a JSON Lines recorder.

        Args:
            file_path: Path to save the JSON Lines file.
            ensure_ascii: If True, ensure all non-ASCII characters are escaped.
                If False, these characters will be output as-is.
        """
        super().__init__(file_path)
        self.ensure_ascii = ensure_ascii

        # Create or open the file
        self._file = open(self.file_path, "w", encoding="utf-8")

    @override
    def write(self, data: Any) -> None:
        """Write data as a JSON Lines entry to the file.

        Automatically adds the current timestamp to each entry. The entry
        is written as a single line with the format:
        {"timestamp": timestamp, "data": data}

        Args:
            data: Data to be recorded. Must be JSON serializable.

        Raises:
            TypeError: If data is not JSON serializable.
            RuntimeError: If the recorder is already closed.
        """
        if self._file.closed:
            raise RuntimeError("Recorder is already closed.")

        # Get current timestamp
        timestamp = time.time()

        # Create the entry with timestamp
        entry = {"timestamp": timestamp, "data": data}

        # Convert to JSON string
        json_str = json.dumps(entry, ensure_ascii=self.ensure_ascii)

        # Write the entry and a newline
        self._file.write(json_str + "\n")

    @override
    def close(self) -> None:
        """Close the JSON Lines file and release resources."""
        if hasattr(self, "_file") and not self._file.closed:
            self._file.close()
