"""Tests for type definitions in skyutils.types_skyutils"""

import sys
from pathlib import Path

import pytest

from skyutils.types_skyutils import PathLike, ConfigDict


def test_pathlike_import():
    """Test that PathLike type can be imported"""
    assert PathLike is not None


def test_configdict_import():
    """Test that ConfigDict type can be imported"""
    assert ConfigDict is not None


def test_blender_types_import():
    """Test that Blender types can be imported (should work even without Blender)"""
    from skyutils.types_skyutils import BlenderObject, BlenderCollection

    # These should be importable but will be Any when bpy is not available
    assert BlenderObject is not None
    assert BlenderCollection is not None


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
