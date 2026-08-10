"""Plotting utilities for atmospheric data visualization."""

from .colors import (
    get_cmap,
    get_next_color,
    get_nth_color,
    last_color,
    match_color,
    sequential_cmap,
    shifted_colormap,
    single_color_cmap,
    transparent_under_cmap,
)
from .core import (
    last_artist,
    clean_legend,
    contour_legend,
    add_row_header,
    fig_multisave,
    share_axes,
    scale_axes_ticks,
    prepend_axes_letters,
    gif_from_pngs,
)

from .animation import animate_grouped_xarray

__all__ = [
    "match_color",
    "last_artist",
    "last_color",
    "clean_legend",
    "contour_legend",
    "get_nth_color",
    "get_next_color",
    "get_cmap",
    "add_row_header",
    "shifted_colormap",
    "fig_multisave",
    "sequential_cmap",
    "single_color_cmap",
    "transparent_under_cmap",
    "share_axes",
    "scale_axes_ticks",
    "prepend_axes_letters",
    "gif_from_pngs",
]
