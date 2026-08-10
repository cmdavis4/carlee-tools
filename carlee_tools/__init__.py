"""
carlee_tools: Core utilities, types, and plotting for atmospheric science.

Provides shared utilities and type definitions for atmospheric modeling packages.
"""

__version__ = "1.2.0"

# Import submodules
from . import dt
from . import io
from . import kv
from . import spacing
from . import utils
from . import types_carlee_tools
from . import plotting

# Re-export commonly used types
from .types_carlee_tools import (
    PathLike,
    NumpyNumeric,
)

# Re-export commonly used utilities from their topical modules. These functions
# moved out of `utils` into the dt/kv/io/spacing modules during the reorg, but
# stay importable from the top-level `carlee_tools` namespace for convenience.
from .dt import dt_to_str, str_to_dt, current_dt_str
from .kv import to_kv_pairs, to_kv_str
from .io import read_file, write_file, raise_if_exists
from .spacing import warn_if_not_evenly_spaced
from .utils import *

__all__ = [
    # Modules
    "dt",
    "io",
    "kv",
    "spacing",
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
    "warn_if_not_evenly_spaced",
]
