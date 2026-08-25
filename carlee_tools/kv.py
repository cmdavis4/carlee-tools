from pathlib import Path
from typing import Any, Dict, List, Union

from .time import str_to_dt
from .types_carlee_tools import PathLike, is_arraylike


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
        if not is_arraylike(v):
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
