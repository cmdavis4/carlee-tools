"""Tests for utilities that require numpy"""

import pytest

# Skip all tests in this file if numpy is not available
pytest.importorskip("numpy")

import numpy as np

from cloudy_core.utils import (
    is_evenly_spaced,
    spacing,
    maybe_random_choice,
    maybe_cast_to_float,
)


class TestArrayUtils:
    """Test numpy array utility functions"""

    def test_is_evenly_spaced_true(self):
        """Test detecting evenly spaced arrays"""
        arr = np.array([1, 2, 3, 4, 5])
        assert is_evenly_spaced(arr) is True

    def test_is_evenly_spaced_false(self):
        """Test detecting unevenly spaced arrays"""
        arr = np.array([1, 2, 4, 7, 11])
        assert is_evenly_spaced(arr) is False

    def test_is_evenly_spaced_single_element(self):
        """Test single element array (should be considered evenly spaced)"""
        arr = np.array([5])
        assert is_evenly_spaced(arr) is True

    def test_spacing_basic(self):
        """Test calculating spacing of array"""
        arr = np.array([0, 5, 10, 15, 20])
        result = spacing(arr)
        assert result == 5

    def test_spacing_raises_if_uneven(self):
        """Test that spacing raises error for uneven arrays"""
        arr = np.array([1, 2, 4, 7])
        with pytest.raises(ValueError):
            spacing(arr, raise_if_not_evenly_spaced=True)

    def test_maybe_random_choice_smaller_than_size(self):
        """Test random choice when array is smaller than requested size"""
        arr = np.array([1, 2, 3])
        result = maybe_random_choice(arr, size=10)
        assert len(result) == 3
        assert np.array_equal(result, arr)

    def test_maybe_random_choice_larger_than_size(self):
        """Test random choice when array is larger than requested size"""
        arr = np.arange(100)
        result = maybe_random_choice(arr, size=10)
        assert len(result) == 10
        assert all(x in arr for x in result)

    def test_maybe_cast_to_float_success(self):
        """Test casting to float when possible"""
        arr = np.array([1, 2, 3])
        result = maybe_cast_to_float(arr)
        assert result.dtype == np.float64

    def test_maybe_cast_to_float_failure(self):
        """Test that failed cast returns original array"""
        arr = np.array(['a', 'b', 'c'])
        result = maybe_cast_to_float(arr)
        assert result.dtype.kind == 'U'  # Unicode string
        assert np.array_equal(result, arr)
