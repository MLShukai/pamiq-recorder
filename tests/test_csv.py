"""Tests for the CSV recorder module."""

import csv
import time
from pathlib import Path

import pytest

from pamiq_recorder.csv import CsvRecorder


class TestCsvRecorder:
    """Test suite for the CsvRecorder class."""

    def test_default_file_name_format(self):
        assert Path(CsvRecorder.DEFAULT_FILE_NAME_FORMAT).suffix == ".csv"

    @pytest.fixture
    def csv_path(self, tmp_path: Path) -> Path:
        """Create a temporary path for the CSV file."""
        return tmp_path / "test_data.csv"

    def test_init_and_file_creation(self, csv_path: Path):
        """Test recorder initialization creates a file with correct headers."""
        headers = ["value1", "value2", "value3"]
        recorder = CsvRecorder(file_path=csv_path, headers=headers)

        try:
            # Check that the CSV file exists
            assert csv_path.exists(), "CSV file was not created"

            # Check file content for headers
            with open(csv_path, newline="") as f:
                reader = csv.reader(f)
                header_row = next(reader)
                assert header_row == ["timestamp"] + headers
        finally:
            recorder.close()

    def test_init_custom_timestamp_header(self, csv_path: Path):
        """Test initialization with custom timestamp header."""
        headers = ["col1", "col2"]
        timestamp_header = "unix_time"
        recorder = CsvRecorder(
            file_path=csv_path, headers=headers, timestamp_header=timestamp_header
        )

        try:
            # Check file content for headers
            with open(csv_path, newline="") as f:
                reader = csv.reader(f)
                header_row = next(reader)
                assert header_row == [timestamp_header] + headers
        finally:
            recorder.close()

    def test_init_conflicting_timestamp_header(self, csv_path: Path):
        """Test initialization fails when timestamp header conflicts with user
        headers."""
        headers = [
            "time",
            "value",
            "timestamp",
        ]  # Conflict with default timestamp_header

        with pytest.raises(
            ValueError,
            match="Timestamp header 'timestamp' conflicts with a user-provided header",
        ):
            CsvRecorder(file_path=csv_path, headers=headers)

    def test_init_duplicate_headers(self, csv_path: Path):
        """Test initialization fails when duplicate column names are
        provided."""
        headers = ["time", "value", "time", "another_value"]  # "time" appears twice

        with pytest.raises(
            ValueError, match="Duplicate column names found in headers: time"
        ):
            CsvRecorder(file_path=csv_path, headers=headers)

    def test_init_multiple_duplicate_headers(self, csv_path: Path):
        """Test initialization fails when multiple sets of duplicate headers
        are found."""
        headers = ["a", "b", "a", "c", "b", "d"]  # "a" and "b" both appear twice

        with pytest.raises(ValueError, match="Duplicate column names found in headers"):
            CsvRecorder(file_path=csv_path, headers=headers)

    def test_write_data(self, csv_path: Path):
        """Test writing data rows."""
        headers = ["name", "value", "flag"]
        recorder = CsvRecorder(file_path=csv_path, headers=headers)

        try:
            # Write a few rows of data
            recorder.write(["item1", 42, True])
            recorder.write(["item2", -1.5, False])
            recorder.close()

            # Check file content
            with open(csv_path, newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)

                # Should have 3 rows: header + 2 data rows
                assert len(rows) == 3

                # First row should be headers
                assert rows[0] == ["timestamp", "name", "value", "flag"]

                # Data rows should have timestamp + our data
                assert len(rows[1]) == 4  # timestamp + 3 values
                assert rows[1][1:] == ["item1", "42", "True"]
                assert len(rows[2]) == 4
                assert rows[2][1:] == ["item2", "-1.5", "False"]

                # Check that timestamps are numeric
                assert float(rows[1][0])  # Should convert to float without error
                assert float(rows[2][0])
        finally:
            recorder.close()

    def test_write_incorrect_data_length(self, csv_path: Path):
        """Test write method rejects data with incorrect length."""
        headers = ["col1", "col2"]
        recorder = CsvRecorder(file_path=csv_path, headers=headers)

        try:
            # Try to write too few values
            with pytest.raises(ValueError, match="Expected 2 values, got 1"):
                recorder.write(["only_one_value"])

            # Try to write too many values
            with pytest.raises(ValueError, match="Expected 2 values, got 3"):
                recorder.write(["value1", "value2", "extra_value"])
        finally:
            recorder.close()

    def test_timestamps_are_sequential(self, csv_path: Path):
        """Test that timestamps are sequential and close to current time."""
        headers = ["value"]
        recorder = CsvRecorder(file_path=csv_path, headers=headers)

        # Get current time for reference
        start_time = time.time()

        try:
            # Write a few rows with small delays
            recorder.write([1])
            time.sleep(0.01)  # Small delay
            recorder.write([2])
            recorder.close()

            # Read and check timestamps
            with open(csv_path, newline="") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                rows = list(reader)

                timestamp1 = float(rows[0][0])
                timestamp2 = float(rows[1][0])

                # Timestamps should be close to current time
                assert abs(timestamp1 - start_time) < 1.0

                # Second timestamp should be greater than first
                assert timestamp2 > timestamp1
        finally:
            recorder.close()

    def test_close_and_reopen(self, csv_path: Path):
        """Test closing and reopening a CSV file."""
        # Create and close a recorder
        recorder = CsvRecorder(file_path=csv_path, headers=["data"])
        recorder.write([42])
        recorder.close()

        # Check that the file exists
        assert csv_path.exists(), "CSV file does not exist after closing"

        # Check that writing after close raises an error
        with pytest.raises(RuntimeError, match="Recorder is already closed"):
            recorder.write([100])
