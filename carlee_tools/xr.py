"""xarray helpers: binning parcel/point data onto a grid and extracting points.

Named ``xr`` (not ``xarray``) so it never shadows the real ``xarray`` package
on ``sys.path``.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import xarray as xr

from .spacing import bin_edges


def bin_points_to_grid(
    sample_coords: Mapping[str, Any],
    edges: Mapping[str, Any],
    coords: Optional[Mapping[str, Any]] = None,
    weights: Optional[Any] = None,
    dtype: Any = np.float32,
    name: str = "binned",
) -> "xr.DataArray":
    """Histogram a long-format set of point samples onto an N-D grid.

    Generic ``np.histogramdd`` wrapper -- the shared core behind both the CM1
    event/parcel-time binning and any arbitrary-grid profile binning. Each entry
    of ``sample_coords`` is the per-sample coordinate along one output axis, and
    ``edges`` gives that axis's bin edges; axis order in the output follows the
    ``edges`` insertion order.

    Unlike ``bin_to_grid`` (which histograms an xarray/dask ``parcel_ds`` lazily
    via xhistogram), this takes already-materialized 1-D sample arrays -- e.g. the
    columns of a long-format pandas event table -- and bins them eagerly. Pass a
    reduced ``edges`` to bin straight to a reduction (e.g. only ``z`` for a
    time-height Hovmoller) without ever materializing the full N-D field.

    Parameters
    ----------
    sample_coords : mapping {axis_name: 1-D array over samples}
        Coordinate of each sample along each binned axis; all arrays share the
        same length ``n_samples``. Must contain exactly the keys of ``edges``. A
        pandas DataFrame works directly (column access by name).
    edges : mapping {axis_name: 1-D array of bin edges}
        Bin edges per axis. The mapping's ordering fixes the output dim order. For
        a CM1 grid axis use ``carlee_tools.spacing.bin_edges``; for an integer
        index axis (e.g. a parcel-time index) use ``np.arange(n + 1) - 0.5``.
    coords : mapping {axis_name: 1-D array}, optional
        Output coordinate labels per axis (length = number of bins on that axis).
        Defaults to the edge midpoints. Supply this to label an integer time axis
        with real datetimes, or a CM1 axis with its exact cell centers (whose
        midpoints differ from the edge midpoints because the leading edge is 0).
    weights : 1-D array over samples, optional
        Per-sample weights; unweighted gives plain counts.
    dtype : np.dtype
        Output dtype. ``float32`` halves the footprint of the (usually mostly
        empty) grid relative to ``histogramdd``'s ``float64``.
    name : str
        Name of the returned DataArray.

    Returns
    -------
    xr.DataArray on the ``edges`` axes.
    """
    dims = list(edges)
    edge_arrays = [np.asarray(edges[dim]) for dim in dims]
    # Assemble the (n_samples, n_axes) sample matrix in output-axis order.
    sample = np.column_stack(
        [np.asarray(sample_coords[dim]) for dim in dims]
    )
    # Dense histogram over every requested axis.
    counts, _ = np.histogramdd(
        sample,
        bins=edge_arrays,
        weights=None if weights is None else np.asarray(weights),
    )
    # Label each axis: caller-supplied centers where given, else edge midpoints.
    coords = coords or {}
    out_coords = {
        dim: (
            np.asarray(coords[dim])
            if dim in coords
            else (edge[:-1] + edge[1:]) / 2.0
        )
        for dim, edge in zip(dims, edge_arrays)
    }
    return xr.DataArray(
        counts.astype(dtype), dims=dims, coords=out_coords, name=name
    )


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
