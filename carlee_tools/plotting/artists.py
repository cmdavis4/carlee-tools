"""Low-level helpers that read properties (identity, color) off matplotlib
artists.

These are the foundation the higher-level plotting modules build on: ``core``
(legends) and ``colors`` both depend on this module, and this module depends on
nothing else in the package — which keeps the ``core`` <-> ``colors`` import
graph acyclic.
"""

from typing import Any, Optional

import matplotlib as mpl
import matplotlib.axes
import matplotlib.pyplot as plt


def last_artist(ax: Optional[matplotlib.axes.Axes] = None) -> Any:
    """
    Return the data artist most recently added to an axes.

    Saves you from having to capture and unpack whatever a plotting call
    returned (a single Line2D, a list of them, a container, a collection, ...)
    just to refer back to it. Pairs with ``match_color`` via ``last_color``.

    Matplotlib keeps every data artist (lines, collections, patches, images) in
    a single insertion-ordered list on the axes, ``ax._children``, from which
    the public ``ax.lines`` / ``ax.collections`` / ``ax.patches`` views are
    filtered. The last element is therefore the most recently added artist,
    regardless of type. Container-producing calls (``bar``, ``errorbar``) put
    their individual primitives in this list too, so the last child is the last
    bar / the errorbar's cap collection — whose color still matches the call.

    Args:
        ax: Axes to read from. Defaults to the current axes (``plt.gca()``),
            mirroring the rest of pyplot's implicit-axes convention.

    Returns:
        The most recently added data artist.

    Raises:
        ValueError: If the axes has no data artists yet.
    """
    # Default to the current axes so `last_artist()` reads the axes just plotted
    # into, the same way plt.plot()/plt.gca() do.
    if ax is None:
        ax = plt.gca()
    # `_children` is private but has been the unified, insertion-ordered artist
    # list since matplotlib 3.5; the public per-type lists are just views on it.
    children = ax._children
    if not children:
        raise ValueError("Axes has no data artists to take the last one from.")
    return children[-1]


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
