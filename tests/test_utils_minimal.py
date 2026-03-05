"""Tests for utilities that don't require numpy"""

import datetime as dt
from pathlib import Path

import pytest

from carlee_tools.utils import (
    dt_to_str,
    str_to_dt,
    current_dt_str,
    to_kv_pairs,
    to_kv_str,
    prepend_to_stem,
    append_to_stem,
    NUMERICAL_DT_FORMAT,
)


class TestDatetimeUtils:
    """Test datetime utility functions"""

    def test_current_dt_str(self):
        """Test getting current datetime as string"""
        result = current_dt_str()
        assert isinstance(result, str)
        assert len(result) == 14  # YYYYMMDDHHMMSS format
        assert result.isdigit()

    def test_dt_to_str_basic(self):
        """Test converting datetime to string"""
        test_dt = dt.datetime(2024, 1, 15, 10, 30, 45)
        result = dt_to_str(test_dt)
        assert result == "20240115103045"

    def test_str_to_dt_basic(self):
        """Test parsing datetime string"""
        test_str = "20240115103045"
        result = str_to_dt(test_str)
        assert isinstance(result, dt.datetime)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_dt_roundtrip(self):
        """Test datetime conversion roundtrip"""
        original = dt.datetime(2024, 1, 15, 10, 30, 45)
        as_str = dt_to_str(original)
        back_to_dt = str_to_dt(as_str)
        assert original == back_to_dt


class TestKeyValueUtils:
    """Test key-value pair utilities"""

    def test_to_kv_str(self):
        """Test converting dict to key-value string"""
        d = {"foo": "bar", "baz": 123}
        result = to_kv_str(d)
        assert "foo-bar" in result
        assert "baz-123" in result

    def test_to_kv_pairs_basic(self):
        """Test parsing key-value pairs from string"""
        s = "foo-bar_baz-123"
        result = to_kv_pairs(s)
        assert result["foo"] == "bar"
        assert result["baz"] == "123"

    def test_to_kv_pairs_from_path(self):
        """Test parsing key-value pairs from Path"""
        p = Path("/some/path/foo-bar_baz-123.txt")
        result = to_kv_pairs(p)
        assert result["foo"] == "bar"
        assert result["baz"] == "123"


class TestPathUtils:
    """Test path manipulation utilities"""

    def test_prepend_to_stem(self):
        """Test prepending to file stem"""
        original = Path("/path/to/file.txt")
        result = prepend_to_stem("prefix_", original)
        assert result.name == "prefix_file.txt"
        assert result.parent == original.parent

    def test_append_to_stem(self):
        """Test appending to file stem"""
        original = Path("/path/to/file.txt")
        result = append_to_stem(original, "_suffix")
        assert result.name == "file_suffix.txt"
        assert result.parent == original.parent
