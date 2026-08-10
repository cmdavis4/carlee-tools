"""Plotting utilities for atmospheric data visualization."""

from .artists import (
    last_artist,
    last_color,
    match_color,
)
from .colors import (
    adjust_luminosity,
    get_cmap,
    get_next_color,
    get_nth_color,
    sequential_cmap,
    shifted_blues,
    shifted_colormap,
    shifted_greens,
    shifted_oranges,
    single_color_cmap,
    transparent_under_cmap,
)
from .core import (
    add_row_header,
    clean_legend,
    contour_legend,
    fig_multisave,
    gif_from_pngs,
    label_columns,
    label_rows,
    n_subplots,
    prepend_axes_letters,
    scale_axes_ticks,
    share_axes,
)
from .animation import (
    animate_grouped_xarray,
    get_bounds,
    make_colorbar,
)

__all__ = [
    # artists
    "last_artist",
    "last_color",
    "match_color",
    # colors
    "adjust_luminosity",
    "get_cmap",
    "get_next_color",
    "get_nth_color",
    "sequential_cmap",
    "shifted_blues",
    "shifted_colormap",
    "shifted_greens",
    "shifted_oranges",
    "single_color_cmap",
    "transparent_under_cmap",
    # core
    "add_row_header",
    "clean_legend",
    "contour_legend",
    "fig_multisave",
    "gif_from_pngs",
    "label_columns",
    "label_rows",
    "n_subplots",
    "prepend_axes_letters",
    "scale_axes_ticks",
    "share_axes",
    # animation
    "animate_grouped_xarray",
    "get_bounds",
    "make_colorbar",
]
