import colorsys
from typing import Any

import matplotlib as mpl
import matplotlib.axes
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from carlee_tools.plotting.core import last_artist


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


def match_color(artist: Any) -> Any:
    """
    Return the display color of a matplotlib artist, generically across types.

    The point is to recover "the" color a viewer associates with an artist so a
    second, corresponding artist can be drawn with the same color, e.g.
    ``ax.axhline(y, color=match_color(line))``. Different artist types expose
    their color through different (and sometimes plural) accessors, so this
    function centralizes the per-type rules. ``clean_legend`` uses it to pick
    each legend entry's text color.

    Handled artist types:
      - Line2D (line plots): the line color.
      - Collection (scatter points, stackplot/fill_between bands, LineCollection):
        the face color when the face is opaque, otherwise the edge color (e.g.
        unfilled markers), otherwise the face color even if transparent.
      - Patch (bars, histogram rectangles, polygons): the face color, falling
        back to the edge color when the face is transparent (e.g. step
        histograms drawn with ``histtype="step"``).
      - ErrorbarContainer: the color of the central data line/marker
        (``container.lines[0]``), since the container's child artists carry
        auto-generated labels rather than the display color.

    Args:
        artist: The matplotlib artist (or container) to read a color from.

    Returns:
        The artist's color. For lines this is whatever ``get_color`` reports
        (a color spec such as ``"C0"`` or an RGBA tuple); for collections and
        patches it is an RGBA tuple. Returns ``None`` when no color can be
        determined (e.g. a fully invisible collection with neither a face nor
        an edge color).

    Raises:
        TypeError: If the artist type is not recognized.
    """
    # Line2D: the color is unambiguous, just read it off directly.
    if isinstance(artist, mpl.lines.Line2D):
        return artist.get_color()

    # ErrorbarContainer: the label lives on the container, but the display
    # color lives on the central data line (its first child line).
    if isinstance(artist, mpl.container.ErrorbarContainer):
        return match_color(artist.lines[0])

    # Collection: covers scatter markers, stackplot/fill_between bands, and
    # LineCollections. Face and edge colors are arrays (one row per element),
    # so reduce to a single RGBA tuple.
    if isinstance(artist, mpl.collections.Collection):
        facecolors = artist.get_facecolor()
        edgecolors = artist.get_edgecolor()
        # Prefer the face color when the face is actually opaque (filled markers,
        # filled bands).
        has_opaque_face = facecolors.size > 0 and facecolors[0][3] > 0
        if has_opaque_face:
            return facecolors[0]
        # Fall back to the edge color when the face is transparent (unfilled
        # markers drawn with only an outline).
        if edgecolors.size > 0:
            return edgecolors[0]
        # Last resort: return the (transparent) face color if there is one, else
        # signal that the artist has no determinable color.
        if facecolors.size > 0:
            return facecolors[0]
        return None

    # Patch: bars, histogram rectangles, and polygons.
    if isinstance(artist, mpl.patches.Patch):
        facecolor = artist.get_facecolor()
        # A transparent face (alpha == 0) means the fill is invisible, e.g. a
        # step histogram, so the outline carries the color instead.
        if facecolor[3] == 0:
            return artist.get_edgecolor()
        return facecolor

    raise TypeError(
        "match_color does not know how to get a color from a"
        f" {type(artist).__name__}"
    )


def last_color(ax: Optional[matplotlib.axes.Axes] = None) -> Any:
    """
    Return the color of the most recently added artist on an axes.

    Convenience wrapper around ``match_color(last_artist(ax))`` for the common
    case of matching a new artist's color to the one just drawn, e.g.::

        ax.plot(x, y, label="data")
        ax.axhline(y.mean(), color=last_color(), linestyle="--")

    Args:
        ax: Axes to read from. Defaults to the current axes (``plt.gca()``).

    Returns:
        The color of the last-added artist (see ``match_color`` for the exact
        per-type rules and return format).
    """
    return match_color(last_artist(ax))


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
