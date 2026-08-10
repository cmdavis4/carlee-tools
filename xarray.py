import numpy as np

from typing import Any, Dict, List

from carlee_tools.spacing import bin_edges


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
