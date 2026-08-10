"""
Type definitions for the carlee_tools package.

This module provides type aliases and custom types used throughout the package.
"""

import datetime as dt
import sys
from pathlib import Path
from typing import Union

import numpy as np

# pyright: reportRedeclaration=false

# TypeAlias moved into typing in 3.10; on 3.8-3.9 it comes from the
# typing-extensions dependency (declared in pyproject for python_version<'3.10').
if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

# Path-like objects (strings or Path instances)
PathLike: TypeAlias = Union[str, Path]

# Numpy scalar / array-like aliases. numpy is a hard dependency, so these are
# always defined.
NumpyNumeric: TypeAlias = Union[np.integer, np.floating]
ArrayLike: TypeAlias = Union[list, tuple, np.ndarray]

# Datetime-like objects. pandas is an optional dependency, so only fold
# pd.Timestamp into the alias when pandas is actually installed.
try:
    import pandas as pd

    DatetimeLike: TypeAlias = Union[dt.datetime, pd.Timestamp, np.datetime64]
except ImportError:
    DatetimeLike: TypeAlias = Union[dt.datetime, np.datetime64]


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
