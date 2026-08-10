"""
Type definitions for the carlee_tools package.

This module provides type aliases and custom types used throughout the package.
"""

from pathlib import Path
import sys
from typing import Union
import datetime as dt
import pandas as pd
import numpy as np

# pyright: reportRedeclaration=false

# Handle TypeAlias for Python < 3.10 compatibility
if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    try:
        from typing_extensions import TypeAlias
    except ImportError:
        # Fallback for when typing_extensions is not available
        TypeAlias = None

# Path-like objects (strings or Path instances)
if TypeAlias is not None:
    PathLike: TypeAlias = Union[str, Path]
    DatetimeLike: TypeAlias = Union[dt.datetime, pd.Timestamp, np.datetime64]
else:
    PathLike = Union[str, Path]

# Optional type if numpy is present
try:
    import numpy as np

    NumpyNumeric: TypeAlias = Union[np.integer, np.floating]
    ArrayLike: TypeAlias = Union[list, tuple, np.ndarray]
except ImportError:
    pass


def is_arraylike(maybe_arr):
    return hasattr(maybe_arr, "__iter__") and not isinstance(maybe_arr, str)


def maybe_cast_to_float(arr: np.ndarray) -> np.ndarray:
    """
    Attempt to cast array to float, returning original array if cast fails.

    Args:
        arr: Array to cast

    Returns:
        Float array if successful, original array otherwise
    """
    try:
        return arr.astype(float)
    except ValueError:
        return arr
