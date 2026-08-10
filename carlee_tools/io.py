from functools import wraps
import json
from pathlib import Path
import pickle as pkl
import shutil
from warnings import warn

import numpy as np


class RaiseIfExistsException(Exception):
    # Type of error specifically for trying to write if something exists
    # Having this makes it easier to catch only this exception
    pass


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
