import datetime as dt

import numpy as np
import xarray as xr


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


NUMERICAL_DT_FORMAT = r"%Y%m%d%H%M%S"
HUMAN_DT_FORMAT = r"%Y-%m-%d %H:%M:%S"

ALL_CUSTOM_DT_FORMATS = [NUMERICAL_DT_FORMAT, HUMAN_DT_FORMAT]


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


def current_dt_str(format: str = NUMERICAL_DT_FORMAT) -> str:
    """
    Get current datetime as formatted string.

    Args:
        format: strftime format string

    Returns:
        Current datetime as formatted string
    """
    return dt.datetime.now().strftime(format)


def td_to_seconds(td):
    return td / np.timedelta64(1, "s")
