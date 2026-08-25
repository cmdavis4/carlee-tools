"""
carlee_tools: Core utilities, types, and plotting for atmospheric science.

Provides shared utilities and type definitions for atmospheric modeling packages.
"""

__version__ = "1.2.0"

# Import submodules
from . import time
from . import io
from . import kv
from . import spacing
from . import utils
from . import xr
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
from .time import dt_to_str, str_to_dt, current_dt_str, td_to_seconds
from .kv import to_kv_pairs, to_kv_str
from .io import read_file, write_file, raise_if_exists, read_or_cache_to
from .spacing import (
    warn_if_not_evenly_spaced,
    spacing,
    bin_edges,
    bin_edges_from_centers,
    layer_thickness,
)
from .utils import *
from .types_carlee_tools import *

__all__ = [
    # Modules
    "time",
    "io",
    "kv",
    "spacing",
    "utils",
    "xr",
    "types_carlee_tools",
    "plotting",
    # Types
    "PathLike",
    "NumpyNumeric",
    # Utilities
    "dt_to_str",
    "str_to_dt",
    "td_to_seconds",
    "current_dt_str",
    "TwoWayDict",
    "to_kv_pairs",
    "to_kv_str",
    "read_file",
    "write_file",
    "raise_if_exists",
    "read_or_cache_to",
    "warn_if_not_evenly_spaced",
    "spacing",
    "bin_edges",
    "bin_edges_from_centers",
    "layer_thickness",
]
