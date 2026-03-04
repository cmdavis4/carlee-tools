"""Tests for type definitions in skyutils.types_skyutils"""

import sys
from pathlib import Path

import pytest

from skyutils.types_skyutils import PathLike


def test_pathlike_import():
    """Test that PathLike type can be imported"""
    assert PathLike is not None


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="NumpyNumeric requires Python 3.10+ for Union in TypeAlias without typing_extensions"
)
def test_numpy_numeric_import():
    """Test that NumpyNumeric type can be imported"""
    try:
        from skyutils.types_skyutils import NumpyNumeric
        assert NumpyNumeric is not None
    except ImportError:
        pytest.skip("NumPy not available")
