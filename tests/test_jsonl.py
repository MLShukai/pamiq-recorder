"""Tests for the JSON Lines recorder module."""

import json
import time
from pathlib import Path

import pytest

from pamiq_recorder.jsonl import JsonlRecorder


class TestJsonlRecorder:
    """Test suite for the JsonlRecorder class."""

    def test_default_file_name_format(self):
        assert Path(JsonlRecorder.DEFAULT_FILE_NAME_FORMAT).suffix == ".jsonl"

    @pytest.fixture
    def jsonl_path(self, tmp_path: Path) -> Path:
        """Create a temporary path for the JSON Lines file."""
        return tmp_path / "test_data.jsonl"

    def test_init_and_file_creation(self, jsonl_path: Path):
        """Test recorder initialization creates a file."""
        recorder = JsonlRecorder(file_path=jsonl_path)

        try:
            # Check that the JSONL file exists
            assert jsonl_path.exists(), "JSONL file was not created"

            # Check recorder attributes
            assert recorder.file_path == jsonl_path
            assert recorder.ensure_ascii is True
        finally:
            recorder.close()

    def test_write_simple_data(self, jsonl_path: Path):
        """Test writing simple data types."""
        recorder = JsonlRecorder(file_path=jsonl_path)

        try:
            # Write different data types
            recorder.write("string value")
            recorder.write(42)
            recorder.write(3.14)
            recorder.write(True)
            recorder.write(None)
            recorder.close()

            # Read and verify the file content
            with open(jsonl_path, encoding="utf-8") as f:
                lines = f.readlines()

            # Should have 5 lines
            assert len(lines) == 5

            # Parse each line and verify structure
            for line in lines:
                entry = json.loads(line)
                assert "timestamp" in entry
                assert "data" in entry
                assert isinstance(entry["timestamp"], (int, float))

            # Verify specific data values
            assert json.loads(lines[0])["data"] == "string value"
            assert json.loads(lines[1])["data"] == 42
            assert json.loads(lines[2])["data"] == 3.14
            assert json.loads(lines[3])["data"] is True
            assert json.loads(lines[4])["data"] is None
        finally:
            recorder.close()

    def test_write_complex_data(self, jsonl_path: Path):
        """Test writing complex data structures."""
        recorder = JsonlRecorder(file_path=jsonl_path)

        # Complex nested data
        complex_data = {
            "name": "Test Object",
            "values": [1, 2, 3],
            "nested": {"a": 1, "b": [4, 5, 6], "c": {"x": "y"}},
        }

        try:
            recorder.write(complex_data)
            recorder.close()

            # Read and verify the file content
            with open(jsonl_path, encoding="utf-8") as f:
                line = f.read().strip()

            entry = json.loads(line)
            assert entry["data"] == complex_data
        finally:
            recorder.close()

    def test_write_non_serializable_data(self, jsonl_path: Path):
        """Test that writing non-serializable data raises TypeError."""
        recorder = JsonlRecorder(file_path=jsonl_path)

        try:
            # A set is not JSON serializable
            with pytest.raises(TypeError, match="not JSON serializable"):
                recorder.write({1, 2, 3})  # Sets are not JSON serializable
        finally:
            recorder.close()

    def test_ensure_ascii_option(self, jsonl_path: Path):
        """Test the ensure_ascii option."""
        # With ensure_ascii=True (default)
        recorder1 = JsonlRecorder(file_path=jsonl_path)
        try:
            recorder1.write("こんにちは")  # Japanese text
            recorder1.close()

            with open(jsonl_path, encoding="utf-8") as f:
                content_with_ascii = f.read()

            # Should contain escape sequences
            assert "\\u" in content_with_ascii
        finally:
            recorder1.close()

        # With ensure_ascii=False
        recorder2 = JsonlRecorder(file_path=str(jsonl_path) + "_2", ensure_ascii=False)
        try:
            recorder2.write("こんにちは")
            recorder2.close()

            with open(str(jsonl_path) + "_2", encoding="utf-8") as f:
                content_without_ascii = f.read()

            # Should contain actual Unicode characters
            assert "こんにちは" in content_without_ascii
            assert "\\u" not in content_without_ascii
        finally:
            recorder2.close()

    def test_timestamps_are_sequential(self, jsonl_path: Path):
        """Test that timestamps are sequential and close to current time."""
        recorder = JsonlRecorder(file_path=jsonl_path)

        # Get current time for reference
        start_time = time.time()

        try:
            # Write a few entries with small delays
            recorder.write("entry1")
            time.sleep(0.01)  # Small delay
            recorder.write("entry2")
            recorder.close()

            # Read and check timestamps
            with open(jsonl_path, encoding="utf-8") as f:
                lines = f.readlines()

            timestamp1 = json.loads(lines[0])["timestamp"]
            timestamp2 = json.loads(lines[1])["timestamp"]

            # Timestamps should be close to current time
            assert abs(timestamp1 - start_time) < 1.0

            # Second timestamp should be greater than first
            assert timestamp2 > timestamp1
        finally:
            recorder.close()

    def test_close_and_reopen(self, jsonl_path: Path):
        """Test closing and reopening a JSON Lines file."""
        # Create and close a recorder
        recorder = JsonlRecorder(file_path=jsonl_path)
        recorder.write({"key": "value"})
        recorder.close()

        # Check that the file exists
        assert jsonl_path.exists(), "JSONL file does not exist after closing"

        # Check that writing after close raises an error
        with pytest.raises(RuntimeError, match="Recorder is already closed"):
            recorder.write({"another": "value"})
