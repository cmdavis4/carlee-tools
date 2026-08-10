from warnings import warn

import numpy as np


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
