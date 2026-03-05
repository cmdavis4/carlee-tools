"""
carlee_tools: Core utilities, types, and plotting for atmospheric science.

Provides shared utilities and type definitions for atmospheric modeling packages.
"""

__version__ = "1.0.0"

# Import submodules
from . import utils
from . import types_carlee_tools
from . import plotting

# Re-export commonly used types
from .types_carlee_tools import (
    PathLike,
    NumpyNumeric,
)

# Re-export commonly used utilities
from .utils import *

__all__ = [
    # Modules
    "utils",
    "types_carlee_tools",
    "plotting",
    # Types
    "PathLike",
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
