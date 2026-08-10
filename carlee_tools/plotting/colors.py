import colorsys

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np


def get_nth_color(n):
    return plt.rcParams["axes.prop_cycle"].by_key()["color"][n]


def get_next_color(ax):
    """Get the next color from the axes color cycle.

    Args:
        ax: Matplotlib axes object

    Returns:
        str: The next color in the cycle
    """
    # In modern matplotlib, we need to plot an invisible line to advance the color cycle
    # and get the color that was used
    line = ax.plot([], [])[0]
    color = line.get_color()
    line.remove()
    return color


def get_cmap(name):
    """
    Get a matplotlib colormap object for a given colormap name.
    This is admittedly silly to make a function for, but I can never remember
    how to do it from matplotlib directly.

    Args:
        name (str): Name of the colormap

    Returns:
        matplotlib.colors.Colormap: The colormap object
    """
    return mpl.colormaps[name]


def shifted_colormap(cmap, new_range, n=256):
    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]
    colors_list = cmap(np.linspace(new_range[0], new_range[1], n))
    return colors.LinearSegmentedColormap.from_list("new", colors_list)


# Define a few shifted colormaps
shifted_blues = shifted_colormap("Blues", (0.2, 1.0))
shifted_greens = shifted_colormap("Greens", (0.3, 1.0))
shifted_oranges = shifted_colormap("Oranges", (0.3, 1.0))


# `match_color`, `last_color`, and `last_artist` now live in the `artists`
# module (which `core` also uses) so that `colors` and `core` no longer import
# each other. Re-exported here for backward compatibility.
from .artists import last_artist, last_color, match_color  # noqa: E402,F401


def sequential_cmap(colors_list, name=None, N=512):
    colors_list = [
        colors_list.to_rgb(color) if isinstance(color, str) else color
        for color in colors_list
    ]
    return colors.LinearSegmentedColormap.from_list(
        name or f"cd_{str(colors_list)}", colors_list, N=N
    )


def single_color_cmap(color, linear_opacity=False, name=None, N=512):
    if isinstance(color, str):
        color = colors.to_rgb(color)
    start_color = (
        (color[0], color[1], color[2], 0) if linear_opacity else (1, 1, 1, 1)
    )
    return sequential_cmap([start_color, color], name=name, N=N)


def transparent_under_cmap(cmap, bad=True):
    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]
    cmap = cmap.copy()
    cmap.set_under((0, 0, 0, 0))
    if bad:
        cmap.set_bad((0, 0, 0, 0))
    return cmap


def adjust_luminosity(color, amount):
    """
    From https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib:

    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """

    try:
        c = colors.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*colors.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
