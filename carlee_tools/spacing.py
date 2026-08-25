from warnings import warn

import numpy as np

from .types_carlee_tools import ArrayLike, NumpyNumeric


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


def raise_if_uneven(arr: ArrayLike, exact: bool = True) -> None:
    """
    Raise ValueError if array elements are not evenly spaced.

    Args:
        arr: Array to check
        exact: If True, require exact spacing; if False, use approximate comparison

    Raises:
        ValueError: If array is not evenly spaced
    """
    if not is_evenly_spaced(arr, exact=exact):
        raise ValueError("Array is not evenly spaced")


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
        raise_if_uneven(arr=arr, exact=exact)
    else:
        warn_if_not_evenly_spaced(arr=arr, exact=exact)
    return arr[1] - arr[0]


def bin_edges_from_centers(centers: ArrayLike) -> np.ndarray:
    """CM1-style bin edges from a 1-D array of cell centers.

    Builds the edges of the bins whose centers are ``centers``: a leading 0 (the
    lower domain boundary), interior edges at the midpoint between each pair of
    adjacent centers, then a trailing edge extrapolated one half-spacing past the
    last center. This is the single source of truth for the edge convention used
    across the grid-binning helpers (``bin_edges``, ``layer_thickness``) so they
    can never drift apart.
    """
    centers = np.asarray(centers)
    return np.concatenate([
        # Leading edge: 0 as the lower boundary of the first bin
        np.array([0.0]),
        # Interior edges: midpoint between each pair of adjacent cell centers
        (centers[:-1] + centers[1:]) / 2.0,
        # Trailing edge: extrapolate one half-spacing past the last center
        np.array([centers[-1] + (centers[-1] - centers[-2]) / 2.0]),
    ])


def bin_edges(ds, dims=["x", "y", "z"]):
    """Per-dimension CM1-style bin edges for the grid dataset ``ds``.

    Returns ``{dim: edge_array}`` for each requested dim, each built from that
    dim's cell centers by ``bin_edges_from_centers``.
    """
    return {dim: bin_edges_from_centers(ds[dim].values) for dim in dims}


def layer_thickness(centers: ArrayLike) -> np.ndarray:
    """Per-cell layer thickness for cells whose centers are ``centers``.

    The width of each CM1-style bin: ``np.diff`` of ``bin_edges_from_centers``
    (leading 0, interior midpoints, extrapolated top). This is the ``dz`` any
    vertical integral / mass-flux-per-height calculation must use so it stays
    consistent with the grid binning.
    """
    return np.diff(bin_edges_from_centers(centers))
