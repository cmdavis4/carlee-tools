"""Utility functions for datetime handling, file operations, and data processing."""

import re
import datetime as dt
import numpy as np
from pathlib import Path
import sys
import importlib
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from warnings import warn
import pickle as pkl
from functools import wraps
import xarray as xr

from .types_carlee_tools import PathLike, ArrayLike, NumpyNumeric, DatetimeLike

DEFAULT_SEED = 137504983571204
NUMERICAL_DT_FORMAT = r"%Y%m%d%H%M%S"
HUMAN_DT_FORMAT = r"%Y-%m-%d %H:%M:%S"

ALL_CUSTOM_DT_FORMATS = [NUMERICAL_DT_FORMAT, HUMAN_DT_FORMAT]


class TwoWayDict:
    """
    A wrapper for nested dictionaries that allows indexing by outer keys or inner keys.

    When indexed by an outer key, returns the corresponding inner dictionary.
    When indexed by an inner key, returns a dictionary mapping outer keys to their
    corresponding inner values for that key.

    Example:
        >>> data = {
        ...     'mesh1': {'type': 'trajectory', 'opacity': 0.5},
        ...     'mesh2': {'type': 'simulation', 'opacity': 0.8},
        ...     'mesh3': {'type': 'trajectory', 'opacity': 0.3}
        ... }
        >>> accessor = NestedDictAccessor(data)
        >>> accessor['mesh1']  # Returns {'type': 'trajectory', 'opacity': 0.5}
        >>> accessor['type']   # Returns {'mesh1': 'trajectory', 'mesh2': 'simulation', 'mesh3': 'trajectory'}
        >>> accessor['opacity'] # Returns {'mesh1': 0.5, 'mesh2': 0.8, 'mesh3': 0.3}
    """

    def __init__(self, nested_dict: Dict[Any, Dict[Any, Any]]):
        """
        Initialize with a dictionary of dictionaries.

        Args:
            nested_dict: Dictionary where values are themselves dictionaries.
        """
        self._data = nested_dict

        self._inner_keys = set()
        for inner_dict in nested_dict.values():
            if isinstance(inner_dict, dict):
                self._inner_keys.update(inner_dict.keys())

    def __getitem__(self, key):
        """
        Get item by key, supporting both outer and inner key access.

        Args:
            key: Either an outer key or an inner key.

        Returns:
            If key is an outer key: the corresponding inner dictionary.
            If key is an inner key: dictionary mapping outer keys to inner values.

        Raises:
            KeyError: If key is found in neither outer nor inner keys.
        """
        # Try as outer key first
        if key in self._data:
            return self._data[key]

        # Try as inner key
        if key in self._inner_keys:
            result = {}
            for outer_key, inner_dict in self._data.items():
                if isinstance(inner_dict, dict) and key in inner_dict:
                    result[outer_key] = inner_dict[key]
            return result

        # Key not found anywhere
        raise KeyError(f"Key '{key}' not found in outer keys or inner keys")

    def __contains__(self, key):
        """Check if key exists in either outer or inner keys."""
        return key in self._data or key in self._inner_keys

    def __iter__(self):
        """Iterate over outer keys."""
        return iter(self._data)

    def keys(self):
        """Return outer keys."""
        return self._data.keys()

    def inner_keys(self):
        """Return all unique inner keys."""
        return self._inner_keys

    def items(self):
        """Return outer key-value pairs."""
        return self._data.items()

    @property
    def values(self):
        return [x for l in self._data.values() for x in l.values()]

    def __repr__(self):
        """Return HTML table representation of the TwoWayDict."""
        if not self._data or not self._inner_keys:
            return "TwoWayDict({})"

        # Get sorted keys for consistent output
        outer_keys = sorted(self._data.keys())
        inner_keys = sorted(self._inner_keys)

        # Start building HTML table
        html = [
            '<table border="1" style="border-collapse: collapse; font-family:'
            ' monospace;">'
        ]

        # Build header row
        html.append("  <thead>")
        html.append("    <tr>")
        html.append(
            '      <th style="padding: 4px 8px; background-color:'
            ' #f0f0f0;"></th>'
        )  # Empty top-left cell
        for inner_key in inner_keys:
            html.append(
                '      <th style="padding: 4px 8px; background-color:'
                f' #f0f0f0;">{str(inner_key)}</th>'
            )
        html.append("    </tr>")
        html.append("  </thead>")

        # Build data rows
        html.append("  <tbody>")
        for outer_key in outer_keys:
            html.append("    <tr>")
            html.append(
                '      <th style="padding: 4px 8px; background-color:'
                f' #f8f8f8;">{str(outer_key)}</th>'
            )

            inner_dict = self._data.get(outer_key, {})
            for inner_key in inner_keys:
                if inner_key in inner_dict:
                    value = inner_dict[inner_key]
                    # Check if value is iterable (but not string) and has length
                    try:
                        if hasattr(value, "__len__") and not isinstance(
                            value, str
                        ):
                            cell_value = str(len(value))
                        else:
                            cell_value = "✓"
                    except TypeError:
                        cell_value = "✓"
                else:
                    cell_value = ""

                html.append(
                    '      <td style="padding: 4px 8px; text-align:'
                    f' center;">{cell_value}</td>'
                )

            html.append("    </tr>")
        html.append("  </tbody>")
        html.append("</table>")

        return "\n".join(html)


def to_t_minutes(time_values, start_time):
    # Massage start time if needed
    try:
        start_time = start_time.to_numpy()
    except AttributeError:
        pass
    delta_minutes = (time_values - start_time) / np.timedelta64(1, "m")
    if isinstance(delta_minutes, (np.ndarray, xr.DataArray)):
        return delta_minutes.astype(int)
    return int(delta_minutes)


def dt_to_str(
    dt_like: DatetimeLike, date_format: str = NUMERICAL_DT_FORMAT
) -> str:
    """
    Convert datetime-like objects to formatted strings.

    Args:
        dt_like: Datetime-like object (datetime, numpy.datetime64, pandas.Timestamp, string, etc.)
        date_format: strftime format string

    Returns:
        Formatted datetime string

    Raises:
        ValueError: If dt_like cannot be converted to datetime
        TypeError: If date_format is invalid
    """

    # Handle None/empty inputs
    if dt_like is None:
        raise ValueError("Cannot convert None to datetime string")

    # If it's already a string, try to parse it first
    if isinstance(dt_like, str):
        dt_like = str_to_dt(dt_like)

    # Handle native Python datetime objects (including subclasses like pandas.Timestamp)
    if hasattr(dt_like, "strftime"):
        try:
            return dt_like.strftime(date_format)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Failed to format datetime with format '{date_format}': {e}"
            )

    # Handle numpy datetime64
    if hasattr(dt_like, "astype") and hasattr(dt_like, "dtype"):
        if np.issubdtype(dt_like.dtype, np.datetime64):
            try:
                # Convert to datetime64[s] first to avoid precision issues
                dt_as_seconds = dt_like.astype("datetime64[s]")
                # Then convert to Python datetime
                py_datetime = dt_as_seconds.astype(dt.datetime)
                return py_datetime.strftime(date_format)
            except (ValueError, TypeError, OverflowError) as e:
                raise ValueError(
                    f"Failed to convert numpy datetime64 to string: {e}"
                )

    # Handle time.struct_time
    if hasattr(dt_like, "tm_year"):
        try:
            py_datetime = dt.datetime(*dt_like[:6])
            return py_datetime.strftime(date_format)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to convert struct_time to string: {e}")

    # Handle timestamp (Unix epoch) - both int and float
    if isinstance(dt_like, (int, float)) and dt_like > 0:
        try:
            py_datetime = dt.datetime.fromtimestamp(dt_like)
            return py_datetime.strftime(date_format)
        except (ValueError, TypeError, OSError) as e:
            raise ValueError(
                f"Failed to convert timestamp {dt_like} to string: {e}"
            )

    # Try pandas Timestamp if pandas is available
    try:
        import pandas as pd

        if isinstance(dt_like, pd.Timestamp):
            return dt_like.strftime(date_format)
    except ImportError:
        pass

    # Last resort: try to convert to datetime using str_to_dt
    try:
        parsed_dt = str_to_dt(str(dt_like))
        return parsed_dt.strftime(date_format)
    except (ValueError, TypeError):
        pass

    raise ValueError(
        f"Cannot convert object of type {type(dt_like)} to datetime string:"
        f" {dt_like}"
    )


def str_to_dt(
    s: str,
    date_format: Optional[str] = None,
    try_digits_only=True,
    raise_if_failure: bool = True,
) -> Optional[dt.datetime]:
    """
    Coerce datetime-like strings to native datetime objects.

    Args:
        s: String to parse
        date_format: Specific format to try, or None to try all formats
        raise_if_failure: Whether to raise exception on parsing failure

    Returns:
        datetime object or None (if raise_if_failure=False)
    """

    def _parse_str(s):
        if not date_format:
            possible_calls = [
                lambda s: dt.datetime.fromisoformat(s),
            ] + [
                lambda s, fmt=fmt: dt.datetime.strptime(s, fmt)
                for fmt in ALL_CUSTOM_DT_FORMATS
            ]

            # Add pandas parsing if available
            try:
                import pandas as pd

                possible_calls.append(lambda s: pd.Timestamp(s).to_pydatetime())
            except ImportError:
                pass

            # Add dateutil parsing if available
            try:
                from dateutil.parser import parse as dateutil_parse

                possible_calls.append(lambda s: dateutil_parse(s))
            except ImportError:
                pass
        else:
            possible_calls = [lambda s: dt.datetime.strptime(s, date_format)]

        for possible_call in possible_calls:
            try:
                return possible_call(s)
            except (ValueError, TypeError, AttributeError):
                continue
        return None

    this_dt = _parse_str(s)
    # Also try just stripping everything that's not a digit
    if not this_dt and try_digits_only:
        this_dt = _parse_str("".join([c for c in s if c.isdigit()]))

    if not this_dt and raise_if_failure:
        raise ValueError(f"Could not coerce string '{s}' to datetime")
    return this_dt


class RaiseIfExistsException(Exception):
    # Type of error specifically for trying to write if something exists
    # Having this makes it easier to catch only this exception
    pass


def to_kv_pairs(
    s: Union[str, Path],
    parse_datetimes: bool = False,
    parse_floats: bool = False,
    parse_bools: bool = False,
) -> Dict[str, Any]:
    """
    Parse key-value pairs from a string or Path stem.

    Args:
        s: String or Path to parse (if Path, uses stem)
        parse_datetimes: Whether to attempt parsing values as datetime objects
        parse_floats: Whether to attempt parsing values as floats

    Returns:
        Dictionary of key-value pairs
    """
    # Handle some convenient cases
    if isinstance(s, Path):
        s = str(s.stem)
    else:
        s = str(s)
    d = {}
    for kv_pair in s.split("_"):
        # If there's no key-value form, skip it
        if "-" not in kv_pair:
            continue
        splits = kv_pair.split("-")
        # Handle the case in which there's more than one - in the name, e.g.
        # if the value is a negative number
        k = splits[0]
        v = "-".join(splits[1:])
        if parse_datetimes:
            try:
                # Try to parse as datetime first
                v = str_to_dt(v)
            except:
                pass
        if parse_floats and isinstance(v, str):
            try:
                v = float(v)
            except:
                pass
        if parse_bools and isinstance(v, str):
            if v in ["true", "True"]:
                v = True
            elif v in ["false", "False"]:
                v = False
        d[k] = v
    return d


def to_kv_str(d: Dict[str, Any]) -> str:
    """
    Convert a dictionary to a key-value filename string.

    Args:
        d: Dictionary to convert

    Returns:
        String in format "key1-value1_key2-value2_..."
    """
    sanitize = lambda x: str(x).replace("_", "")
    return "_".join([f"{sanitize(k)}-{sanitize(v)}" for k, v in d.items()])


def key_in_selector(
    key: Dict[str, Any], selector: Dict[str, Union[str, List[str]]]
) -> bool:
    """
    Check if a key matches the selector criteria.

    Args:
        key: Dictionary to check
        selector: Selection criteria

    Returns:
        True if key matches all selector criteria
    """
    key = dict(key)
    selector = dict(selector)
    for k, v in selector.items():
        if not isinstance(v, ArrayLike):
            v = [v]
        if key.get(k) not in v:
            return False
    return True


def filter_paths_by_selector(
    paths: List[PathLike],
    selector: Dict[str, List[Any]],
    parse_floats: bool = True,
) -> List[PathLike]:
    """
    Filter a list of paths based on key-value selector criteria.

    Args:
        paths: List of file paths to filter
        selector: Dictionary of selection criteria
        parse_floats: Whether to parse numeric values as floats

    Returns:
        Filtered list of paths
    """
    filtered = []
    for this_path in paths:
        # Pull out all of the keys from the directory name
        this_path_kv_pairs = to_kv_pairs(
            Path(this_path).stem, parse_floats=parse_floats
        )
        for selector_key, selector_values in selector.items():
            if this_path_kv_pairs.get(selector_key) not in selector_values:
                continue
        filtered.append(this_path)
    return filtered


def current_dt_str(format: str = NUMERICAL_DT_FORMAT) -> str:
    """
    Get current datetime as formatted string.

    Args:
        format: strftime format string

    Returns:
        Current datetime as formatted string
    """
    return dt.datetime.now().strftime(format)


def filter_to_points(
    da: Any, as_dicts: bool = True
) -> Union[List[Dict[str, Any]], np.ndarray]:
    """
    Filter a DataArray to extract points where values are True.

    Args:
        da: xarray DataArray to filter
        as_dicts: If True, return list of dictionaries; if False, return numpy array

    Returns:
        Filtered points as either list of dicts or numpy array
    """
    # Assume da is a DataArray of the boolean we want to filter by
    # assert set(da.dims) == set(['x', 'y', 'z'])
    s = da.to_series()
    filtered_points = s.loc[s.astype(bool)]
    # Dumb that this is the only way I can figure out to convert the array of tuples to a 2D array
    if as_dicts:
        return [
            {dim_name: l[dim_ix] for dim_ix, dim_name in enumerate(da.dims)}
            for l in filtered_points.index
        ]
    else:
        return np.array([np.array(l) for l in filtered_points.index])


def raise_if_exists(fpath: PathLike) -> None:
    """
    Raise OSError if the given file path exists.

    Args:
        fpath: File path to check

    Raises:
        OSError: If the file exists
    """
    if Path(fpath).exists():
        raise OSError(
            f"Output path {str(fpath)} exists and exist_ok=False was passed"
        )


def maybe_random_choice(
    arr: np.ndarray, size: int, seed: int = DEFAULT_SEED
) -> np.ndarray:
    """
    Return at most `size` elements from array, handling case where array is smaller than size.

    Args:
        arr: Array to sample from
        size: Maximum number of elements to return
        seed: Random seed for reproducibility

    Returns:
        Array with at most `size` elements
    """
    # Return at most `size` elements from arr; just handles the case where `arr`
    # is smaller than `size`
    if size >= len(arr):
        return arr
    else:
        return np.random.default_rng(seed=seed).choice(
            arr, size=size, replace=False
        )


def prepend_to_stem(to_prepend: str, fpath: PathLike) -> Path:
    """
    Prepend text to the stem of a file path.

    Args:
        to_prepend: Text to prepend
        fpath: File path to modify

    Returns:
        Path with modified stem
    """
    fpath = Path(fpath)
    return fpath.with_stem(to_prepend + fpath.stem)


def append_to_stem(fpath: PathLike, to_append: str) -> Path:
    """
    Append text to the stem of a file path.

    Args:
        fpath: File path to modify
        to_append: Text to append

    Returns:
        Path with modified stem
    """
    # Opposite argument order of prepend_to_stem, probably bad design but I like it
    # this way
    fpath = Path(fpath)
    return fpath.with_stem(fpath.stem + to_append)


def fps(
    simulation_minutes_per_second: float, simulation_time_per_frame: Any
) -> float:
    """
    Calculate frames per second from simulation parameters.

    Args:
        simulation_minutes_per_second: Simulation time rate
        simulation_time_per_frame: Time duration per frame

    Returns:
        Frames per second
    """
    simulation_seconds_per_frame = simulation_time_per_frame.nanos / 1e9
    return (simulation_minutes_per_second * 60) / simulation_seconds_per_frame


def is_evenly_spaced(arr: ArrayLike, exact: bool = True) -> bool:
    """
    Check if array elements are evenly spaced.

    Args:
        arr: Array to check
        exact: If True, require exact spacing; if False, use approximate comparison

    Returns:
        True if elements are evenly spaced
    """
    if len(arr) > 1:
        diffs = np.diff(np.array(arr))
        are_evenly_spaced = (
            all(diffs == diffs[0]) if exact else np.allclose(diffs, diffs[0])
        )
        return are_evenly_spaced
    else:
        return True


def raise_if_not_evenly_spaced_(arr: ArrayLike, exact: bool = True) -> None:
    """
    Raise ValueError if array elements are not evenly spaced.

    Args:
        arr: Array to check
        exact: If True, require exact spacing; if False, use approximate comparison

    Raises:
        ValueError: If array is not evenly spaced
    """
    if not is_evenly_spaced(arr, exact=exact):
        raise ValueError(f"Array is not evenly spaced")


def warn_if_not_evenly_spaced(arr: ArrayLike, exact: bool = True) -> None:
    """
    Warn if array elements are not evenly spaced.

    Args:
        arr: Array to check
        exact: If True, require exact spacing; if False, use approximate comparison
    """
    if not is_evenly_spaced(arr, exact=exact):
        warn(
            "Uneven array spacing; returning difference between first two"
            " elements"
        )


def spacing(
    arr: ArrayLike, raise_if_not_evenly_spaced: bool = True, exact: bool = True
) -> NumpyNumeric:
    """
    Calculate spacing between array elements.

    Args:
        arr: Array to calculate spacing for
        raise_if_not_evenly_spaced: Whether to raise error if not evenly spaced
        exact: If True, require exact spacing; if False, use approximate comparison

    Returns:
        Spacing between elements as numpy integer or float

    Raises:
        ValueError: If array is not evenly spaced and raise_if_not_evenly_spaced is True
    """
    if raise_if_not_evenly_spaced:
        raise_if_not_evenly_spaced_(arr=arr, exact=exact)
    else:
        warn_if_not_evenly_spaced(arr=arr, exact=exact)
    return arr[1] - arr[0]


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


def recursive_reload(module: Any, silent=False) -> None:
    """Recursively reload a module and all its submodules"""
    # Get all submodules that start with the module's name
    module_name = module.__name__
    submodules_to_reload = []

    for name, mod in sys.modules.items():
        if name.startswith(module_name + ".") and mod is not None:
            submodules_to_reload.append((name, mod))

    # Sort by depth (deeper modules first) to avoid dependency issues
    submodules_to_reload.sort(key=lambda x: x[0].count("."), reverse=True)

    # Reload submodules first
    for name, mod in submodules_to_reload:
        try:
            importlib.reload(mod)
        except Exception as e:
            if not silent:
                print(f"Failed to reload {name}: {e}")

    # Finally reload the main module
    importlib.reload(module)
    if not silent:
        print(f"Reloaded {module_name}")


def delete_directory_contents(
    dir_path: PathLike, delete_directory: bool = False
):
    import shutil

    dir_path = Path(dir_path)
    assert dir_path.is_dir()
    if dir_path.exists():
        shutil.rmtree(dir_path)
    if not delete_directory:
        dir_path.mkdir(parents=False, exist_ok=False)


def read_file(filepath, *args, **kwargs):
    """Read a file using the appropriate library based on its extension.

    Args:
        filepath: Path to the file to read

    Returns:
        File contents loaded with the appropriate library

    Supported formats:
        - .nc: xarray.open_dataset
        - .npy: numpy.load
        - .csv: pandas.read_csv
        - .parquet: pandas.read_parquet
        - .json: json.load
        - .pkl: pickle
    """
    path = Path(filepath)
    if path.is_dir():
        raise ValueError("Passed value of `filepath` is a directory")
    ext = path.suffix.lower()

    if ext == ".nc":
        import xarray as xr

        return xr.open_dataset(path, *args, **kwargs)
    elif ext == ".npy":
        return np.load(path, *args, **kwargs)
    elif ext == ".csv":
        import pandas as pd

        return pd.read_csv(path, *args, **kwargs)
    elif ext == ".parquet":
        import pandas as pd

        return pd.read_parquet(path, *args, **kwargs)
    elif ext == ".json":
        import json

        with path.open("r") as f:
            return json.load(f, *args, **kwargs)
    elif ext == ".pkl":
        with path.open("rb") as f:
            return pkl.load(f, *args, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def write_file(obj, filepath):
    path = Path(filepath)
    ext = path.suffix.lower()

    ext_write_method_mappings = {
        ".nc": "to_netcdf",
        ".csv": "to_csv",
        ".parquet": "to_parquet",
        ".pkl": "to_pkl",
    }

    def _pkl_write(obj, fpath):
        with Path(fpath).open("wb") as f:
            pkl.dump(obj, f)

    def _json_write(obj, fpath):
        import json

        with Path(fpath).open("w") as f:
            json.dump(obj, f)

    def _npy_write(obj, fpath):
        # Only need to do this one because np.save does (fpath, obj) order
        return np.save(fpath, obj)

    ext_write_fn_mappings = {
        ".npy": _npy_write,
        ".pkl": _pkl_write,
        ".json": _json_write,
    }

    # First check methods
    wrote = False
    if ext in ext_write_method_mappings and hasattr(
        obj, ext_write_method_mappings[ext]
    ):
        # It should be writeable using a method
        getattr(obj, ext_write_method_mappings[ext])(path)
        wrote = True
    elif ext in ext_write_fn_mappings:
        # Didn't have a method or not one writeable to with methods
        ext_write_fn_mappings[ext](obj, path)
        wrote = True

    if not wrote:
        raise ValueError(
            f"No write function or method known for extension {ext}"
        )


def read_or_cache_to(
    filepath, fail_ok=True, not_exist_ok=False, force_compute=False
):
    """Decorator that caches function results to a pickle file.

    If the filepath exists, loads and returns the cached result.
    Otherwise, calls the function, saves the result to the filepath, and returns it.

    Args:
        filepath: Path to the pickle file for caching
    """

    filepath = Path(filepath)
    if not filepath.parent.exists() and not not_exist_ok:
        raise FileNotFoundError(
            f"Parent directory of {str(filepath)} does not exist"
        )

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            path = Path(filepath)
            if path.exists() and not force_compute:
                result = read_file(filepath)
                print(f"Read result from {str(filepath)}")
                return result
            else:
                if force_compute and path.exists():
                    print(
                        "`force_compute=True` passed, overwriting existing"
                        f" file at {str(filepath)}"
                    )
                else:
                    print(f"No file at {str(filepath)}, computing")
                result = func(*args, **kwargs)
                # These computations are generally expensive so be sure to return
                # the result even if the filepath to cache to is bad
                try:
                    write_file(result, filepath)
                    print(f"Wrote computation result to {str(filepath)}")
                    return result
                except Exception as e:
                    if fail_ok:
                        warn(
                            f"Failed to write to {str(filepath)}; returning"
                            " result without caching\nCaught exception"
                            f" was:\n{str(e)}"
                        )
                        return result
                    else:
                        raise

        return wrapper

    return decorator


def is_arraylike(maybe_arr):
    return hasattr(maybe_arr, "__iter__") and not isinstance(maybe_arr, str)


def list_if_single(maybe_single):
    return (
        maybe_single
        if is_arraylike(maybe_single)
        else [
            maybe_single,
        ]
    )


def nth_value(listlike_view, n):
    return list(listlike_view)[n]


def first_value(d):
    return nth_value(d.values(), 0)


def first_key(d):
    return nth_value(d.keys(), 0)


def first_item(d):
    return nth_value(d.items(), 0)


def td_to_seconds(td):
    return td / np.timedelta64(1, "s")


def nice_keys(
    keys,
    *to_kv_pairs_args,
    scientific_notation_threshold=1e4,
    **to_kv_pairs_kwargs,
):
    # Parse each key string into a dict of key-value pairs
    parsed = [
        to_kv_pairs(key, *to_kv_pairs_args, **to_kv_pairs_kwargs)
        for key in keys
    ]

    # Collect all parameter keys that appear across any entry
    all_param_keys = set().union(*(d.keys() for d in parsed))

    # Drop keys whose values are identical across all entries — they don't differentiate
    drop_keys = {
        k for k in all_param_keys if len({d.get(k) for d in parsed}) == 1
    }
    differentiating_keys = all_param_keys - drop_keys

    if not scientific_notation_threshold:
        return {
            keys[ix]: to_kv_str(
                {k: v for k, v in parsed[ix].items() if k not in drop_keys}
            )
            for ix in range(len(keys))
        }

    # For each differentiating key, decide if values should be formatted in sci notation
    # and if so, how many significant figures are needed to keep them distinct.
    # "Large" means |value| >= 1000.
    sci_notation_sigfigs = (
        {}
    )  # param_key -> minimum sigfigs needed (int), or absent if not sci
    for param_key in differentiating_keys:
        # Collect the string values for this param key across all entries that have it
        string_values_for_key = [d[param_key] for d in parsed if param_key in d]

        # Attempt to convert all values to float; skip this key if any are non-numeric
        try:
            float_values_for_key = [float(v) for v in string_values_for_key]
        except (ValueError, TypeError):
            continue

        # Only apply scientific notation if at least one value is "large"
        if not any(
            abs(v) >= scientific_notation_threshold
            for v in float_values_for_key
        ):
            continue

        # Find the minimum number of significant figures that keeps all formatted
        # values distinct from one another
        for candidate_sigfigs in range(1, 16):
            # sigfigs=1 → 0 decimal places in scientific notation, sigfigs=2 → 1, etc.
            decimal_places = candidate_sigfigs - 1
            formatted_at_this_sigfig = [
                f"{v:.{decimal_places}e}" for v in float_values_for_key
            ]
            if len(set(formatted_at_this_sigfig)) == len(float_values_for_key):
                sci_notation_sigfigs[param_key] = candidate_sigfigs
                break

    # Build the nice label for each original key
    result = {}
    for ix, original_key in enumerate(keys):
        # Build display-value dict, applying sci notation where computed above
        display_kv = {}
        for param_key, raw_value in parsed[ix].items():
            if param_key in drop_keys:
                continue
            if param_key in sci_notation_sigfigs:
                # Format this numeric value in scientific notation with the right sigfigs
                decimal_places = sci_notation_sigfigs[param_key] - 1
                display_kv[param_key] = f"{float(raw_value):.{decimal_places}e}"
            else:
                display_kv[param_key] = raw_value
        result[original_key] = to_kv_str(display_kv)
    return result


def bin_edges(ds, dims=["x", "y", "z"]):
    return {
        dim: np.concatenate([
            # Leading edge: use 0 as the lower boundary of the first bin
            np.array([0]),
            # Interior edges: midpoint between each pair of adjacent cell centers
            (ds[dim] - (ds[dim].diff(dim) / 2)).values,
            # Trailing edge: extrapolate one half-spacing past the last center
            np.array([
                ds[dim].values[-1]
                + (ds[dim].values[-1] - ds[dim].values[-2]) / 2
            ]),
        ])
        for dim in dims
    }


def bin_to_grid(
    parcel_ds,
    grid_ds,
    dims=["x", "y", "z"],
    parcel_id_var="parcel_ix",
    weights=None,
):
    from xhistogram.xarray import histogram

    grid_ds = grid_ds.copy()
    # Build bin edges: prepend a leading 0, compute interior midpoints between
    # adjacent grid cell centers, then extrapolate one final edge past the last center.
    # np.append only accepts two arrays; use np.concatenate for three-way joins.
    grid_bin_edges = bin_edges(grid_ds, dims=dims)
    # block_size=None forces xhistogram to process each dask block as a single
    # chunk instead of running its auto-block heuristic. For a 3-D grid the joint
    # bin count N = (nx+2)(ny+2)(nz+2) can exceed xhistogram's hardcoded
    # _MAX_CHUNK_SIZE (10_000_000); the heuristic then computes
    # block_size = _MAX_CHUNK_SIZE // N = 0 and divides by it (ZeroDivisionError).
    # The data is already chunked by time, so one chunk per block is fine.
    binned = histogram(
        *(parcel_ds[dim] for dim in dims),
        bins=list(grid_bin_edges.values()),
        dim=[parcel_id_var],
        weights=weights,
        block_size=None,
    ).rename({f"{dim}_bin": dim for dim in dims})
    return binned
