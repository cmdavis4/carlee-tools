import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections.abc import Iterable
import matplotlib.colors as colors
from tqdm.notebook import tqdm
import numpy as np

import xarray as xr


def _get_bounds_single(da: xr.DataArray, verbose: bool = True):
    if verbose:
        print("Computing bounds of DataArray...")
    return da.min().compute().item(), da.max().compute().item()


def get_bounds(da_or_das: Iterable | xr.DataArray, verbose: bool = True):
    if isinstance(da_or_das, xr.DataArray):
        return _get_bounds_single(da_or_das, verbose=verbose)
    elif isinstance(da_or_das, Iterable) and all(
        [isinstance(x, xr.DataArray) for x in da_or_das]
    ):
        # List-like of dataarrays
        if verbose:
            print(f"Computing bounds over {len(da_or_das)} DataArrays...")
        mins, maxes = zip(*[_get_bounds_single(da, verbose=False) for da in da_or_das])
        return min(mins), max(maxes)
    else:
        raise ValueError(
            "Argument to get_bounds must be a single xr.DataArray or a list-like of"
            " DataArrays"
        )


def make_colorbar(da, ax=None, cax=None, vcenter=None, norm=None, cmap="viridis"):
    if not ax and not cax:
        raise ValueError("Must pass one of ax or cax")
    # Get limits of data
    if not norm:
        vmin, vmax = get_bounds(da)
        if vcenter:
            norm = colors.TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
        else:
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
    return plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, cax=cax)


def _ax_list(axs):
    return list(np.ravel(np.asarray(axs)))


def _resolve_ax(a, axs):
    if isinstance(a, (int, np.integer)):
        return _ax_list(axs)[a]
    return a


def _normalize_mappables(out, axs):
    """Convert animate_fn return -> {ax_obj: mappable}."""
    if out is None:
        return {}
    if isinstance(out, dict):
        return {_resolve_ax(k, axs): v for k, v in out.items() if v is not None}
    return {_ax_list(axs)[0]: out}  # bare mappable shorthand


def _spec_has_explicit_bounds(spec):
    return (
        ("norm" in spec and spec["norm"] is not None)
        or ("vmin" in spec and "vmax" in spec)
        or "data" in spec
    )


def _normalize_colorbars(colorbars, axs, single_frame):
    if not colorbars:
        return []
    ax_objs = _ax_list(axs)
    if isinstance(colorbars, str):
        specs = [{"axes": [a], "mode": colorbars} for a in ax_objs]
    elif isinstance(colorbars, dict):
        specs = [dict(colorbars)]
    else:
        specs = [dict(s) for s in colorbars]

    out = []
    for spec in specs:
        spec.setdefault("axes", [ax_objs[0]])
        spec["axes"] = [_resolve_ax(a, axs) for a in spec["axes"]]
        if "mode" not in spec:
            spec["mode"] = "global" if _spec_has_explicit_bounds(spec) else "per_frame"
        if (
            single_frame
            and spec["mode"] == "global"
            and not _spec_has_explicit_bounds(spec)
        ):
            spec["mode"] = "per_frame"
        out.append(spec)
    return out


def _build_norm(spec, vmin, vmax):
    if spec.get("norm") is not None:
        return spec["norm"]
    vmin = spec.get("vmin", vmin)
    vmax = spec.get("vmax", vmax)
    vcenter = spec.get("vcenter")
    if vcenter is not None:
        return colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    return colors.Normalize(vmin=vmin, vmax=vmax)


def animate_grouped_xarray(
    fig, axs, animation_fn, grouped_ds, colorbars=None, single_frame_ix=None
):
    """
    Animate over a grouped xarray dataset.

    animation_fn(fig, axs, group_name, group_data) returns:
        - a single mappable (bound to axs.flat[0]),
        - a {ax_index_or_obj: mappable} dict, or
        - None (no colorbar handling).

    colorbars:
        None / False  -> no colorbars
        "per_frame"   -> one cbar per axis, recomputed each frame
        "global"      -> one cbar per axis, shared norm (pre-pass to auto-bound)
        list[dict]    -> per-cbar specs. Keys:
            axes:    list of ax indices/objs sharing this cbar/norm (default [0])
            mode:    "per_frame" | "global" (inferred from bounds fields if absent)
            data:    DataArray | iterable for get_bounds() in global mode
            vmin, vmax, vcenter, norm, cmap: explicit overrides
            location: "right"|"bottom"|... (passed to fig.colorbar)
            label:   colorbar label
            cax:     explicit cbar axes (opt out of space-stealing)

    With single_frame_ix set, "global" specs lacking explicit bounds silently
    fall back to "per_frame" (no pre-pass for a single frame).
    """
    groups = list(grouped_ds)
    single_frame = single_frame_ix is not None
    specs = _normalize_colorbars(colorbars, axs, single_frame)

    # 1. Resolve global norms (explicit -> direct; else collect via one pre-pass).
    prepass_specs = []
    for spec in specs:
        if spec["mode"] != "global":
            continue
        if spec.get("norm") is not None:
            spec["_norm"] = spec["norm"]
        elif "vmin" in spec and "vmax" in spec:
            spec["_norm"] = _build_norm(spec, spec["vmin"], spec["vmax"])
        elif "data" in spec:
            vmin, vmax = get_bounds(spec["data"])
            spec["_norm"] = _build_norm(spec, vmin, vmax)
        else:
            prepass_specs.append(spec)

    if prepass_specs:
        print(f"Pre-pass over {len(groups)} frames for global cbar bounds...")
        bounds_per_spec = {id(s): [np.inf, -np.inf] for s in prepass_specs}
        for name, ds in tqdm(groups):
            for ax in np.ravel(axs):
                ax.clear()
            out = animation_fn(fig, axs, name, ds.squeeze())
            mappables = _normalize_mappables(out, axs)
            for spec in prepass_specs:
                spec_ax_ids = {id(a) for a in spec["axes"]}
                for ax, m in mappables.items():
                    if id(ax) not in spec_ax_ids:
                        continue
                    arr = m.get_array()
                    if arr is None:
                        continue
                    arr = np.asarray(arr)
                    finite = np.isfinite(arr)
                    if not finite.any():
                        continue
                    b = bounds_per_spec[id(spec)]
                    b[0] = min(b[0], float(arr[finite].min()))
                    b[1] = max(b[1], float(arr[finite].max()))
        for ax in np.ravel(axs):
            ax.clear()
        for spec in prepass_specs:
            vmin, vmax = bounds_per_spec[id(spec)]
            spec["_norm"] = _build_norm(spec, vmin, vmax)

    # 2. Pre-create persistent colorbars bound to placeholder mappables.
    cbar_by_ax = {}  # id(ax) -> (Colorbar, spec)
    for spec in specs:
        sm = mpl.cm.ScalarMappable(norm=spec.get("_norm"), cmap=spec.get("cmap"))
        cbar_kwargs = {k: spec[k] for k in ("location", "label") if k in spec}
        if "cax" in spec:
            cb = fig.colorbar(sm, cax=spec["cax"], **cbar_kwargs)
        else:
            cb = fig.colorbar(sm, ax=spec["axes"], **cbar_kwargs)
        for a in spec["axes"]:
            cbar_by_ax[id(a)] = (cb, spec)

    def update(i):
        for ax in np.ravel(axs):
            ax.clear()
        name, ds = groups[i]
        out = animation_fn(fig, axs, name, ds.squeeze())
        mappables = _normalize_mappables(out, axs)
        for ax, m in mappables.items():
            entry = cbar_by_ax.get(id(ax))
            if entry is None:
                continue
            cb, spec = entry
            if spec["mode"] == "global":
                m.set_norm(spec["_norm"])
                if "cmap" in spec:
                    m.set_cmap(spec["cmap"])
            else:
                cb.update_normal(m)
        return []

    if single_frame:
        update(single_frame_ix)
        return None
    return FuncAnimation(
        fig, update, frames=tqdm(range(len(groups))), blit=False, repeat=True
    )
