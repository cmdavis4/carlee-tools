"""
cloudy-core: Core utilities and type definitions for cloudy atmospheric modeling packages.

This package provides shared utilities and type definitions that are used across
the cloudy ecosystem of atmospheric modeling and visualization packages.
"""

__version__ = "1.0.0"

# Import core modules
from . import utils
from . import types_core

# Import commonly used types for convenience
from .types_core import (
    PathLike,
    ConfigDict,
    BlenderObject,
    BlenderCollection,
    NumpyNumeric,
)

# Import commonly used utilities for convenience
from .utils import (
    dt_to_str,
    str_to_dt,
    current_dt_str,
    TwoWayDict,
    to_kv_pairs,
    to_kv_str,
    read_file,
    write_file,
    raise_if_exists,
)

__all__ = [
    # Modules
    "utils",
    "types_core",
    # Types
    "PathLike",
    "ConfigDict",
    "BlenderObject",
    "BlenderCollection",
    "NumpyNumeric",
    # Utilities
    "dt_to_str",
    "str_to_dt",
    "current_dt_str",
    "TwoWayDict",
    "to_kv_pairs",
    "to_kv_str",
    "read_file",
    "write_file",
    "raise_if_exists",
]
