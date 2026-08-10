"""Plotting utilities for atmospheric data visualization and analysis.

This module provides functions for creating faceted plots, animations, legends,
and specialized atmospheric plots like soundings and hodographs.
"""

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.figure as mplfig
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation
from tqdm.notebook import tqdm
import matplotlib.animation as mplanim
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import metpy.calc as mpc
from metpy.units import units
from metpy.plots import SkewT, Hodograph
import matplotlib as mpl
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Iterable, Callable
import matplotlib.figure
import matplotlib.axes
import colorsys

from ..types_carlee_tools import PathLike


def format_t_str(t: Any, strftime: str = "%Y-%m-%d %H:%M:%S") -> Any:
    """
    Format timestamp to string, handling numpy datetime64 objects.

    Args:
        t: Timestamp object (numpy datetime64, datetime, etc.)
        strftime: Format string for datetime formatting

    Returns:
        Formatted string or original object if formatting fails
    """
    try:
        # First try numpy datetime64
        return t.astype("datetime64[s]").item().strftime(strftime)
    except (AttributeError, TypeError):
        # Try standard datetime strftime
        try:
            return t.strftime(strftime)
        except (AttributeError, TypeError):
            # If it's not coerceable to a datetime, just leave it as is
            return t


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


def clean_legend(
    ax: matplotlib.axes.Axes,
    include_artists: Optional[List[Any]] = None,
    sort: bool = False,
    use_alphas: bool = False,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """
    Create a clean legend with customizable styling and sorting.

    Args:
        ax: Matplotlib axes to add legend to
        include_artists: List of specific artists to include in legend
        sort: Whether to sort legend entries by maximum values
        use_alphas: Whether to apply alpha values to legend text colors
        **kwargs: Additional keyword arguments passed to legend()

    Returns:
        The modified axes object
    """
    if include_artists is None:
        include_artists = []

    # Create a function to filter out artists that aren't in include_artists
    def filter_artists(artists):
        if not include_artists:
            return artists
        else:
            return [artist for artist in artists if artist in include_artists]

    # Want to handle lines, collections (i.e. scatter plots/points), and patches (i.e. histograms/polygons)
    # Get the max values of each line
    all_maxes = {
        line.get_label(): (
            {
                "max": max(line.get_ydata()),
                "color": match_color(line),
                "alpha": line.get_alpha(),
            }
            if sort
            else {"color": match_color(line), "alpha": line.get_alpha()}
        )
        for line in filter_artists(ax.get_lines())
    }
    collection_maxes = {}
    has_printed_not_implemented = False
    collections_to_iterate = []
    # for collection in filter_artists(ax.collections):
    #     if collection in collections_to_iterate:
    #         continue
    #     if isinstance(collection, mpl.contour.QuadContourSet):
    #         collections_to_iterate += [x for x in collection.collections]
    #     else:
    #         collections_to_iterate.append(collection)
    # for collection in collections_to_iterate:
    for collection in filter_artists(ax.collections):
        # Reduce the collection's (possibly plural) colors to a single RGBA tuple
        # so it is a valid `labelcolor` entry: face color when opaque, else edge.
        color = match_color(collection)
        try:
            collection_maxes[collection.get_label()] = (
                {
                    "max": max([x[1] for x in collection.get_offsets().data]),
                    "color": color,
                }
                if sort
                else {"color": color}
            )
        # stackplot/fill_between bands are PolyCollections with no offsets, so
        # `get_offsets()` yields an empty array and `max([])` raises ValueError;
        # skip those entries when sorting rather than crashing.
        except (NotImplementedError, ValueError):
            if not has_printed_not_implemented:
                print(
                    "clean_legend not implemented for some elements of figure,"
                    " skipping"
                )
                has_printed_not_implemented = True
    all_maxes.update(collection_maxes)
    # Need to iterate through patches to handle the alpha
    patches_dict = {}
    for patch in filter_artists(ax.patches):
        # Face color for a normal (filled) histogram bar; edge color when the
        # face is transparent, as for a step histogram.
        this_color = match_color(patch)
        patches_dict[patch.get_label()] = (
            {
                "max": max([xy[1] for xy in patch.get_xy()]),
                "color": this_color,
            }
            if sort
            else {"color": this_color}
        )
    all_maxes.update(patches_dict)
    # Handle errorbars: each ax.errorbar() call produces an ErrorbarContainer in
    # ax.containers whose label is what appears in the legend, but whose child
    # Line2D/LineCollection artists get auto "_childN" labels that are excluded from
    # the legend — so they never land in the line/collection loops above.
    # The main data line (container.lines[0]) carries the display color.
    container_dict = {}
    for container in filter_artists(ax.containers):
        if not isinstance(container, mpl.container.ErrorbarContainer):
            continue
        label = container.get_label()
        data_line = container.lines[0]  # the central marker/line
        color = match_color(container)
        if sort:
            ydata = data_line.get_ydata()
            container_dict[label] = {
                "max": max(ydata),
                "color": color,
                "alpha": data_line.get_alpha(),
            }
        else:
            container_dict[label] = {
                "color": color,
                "alpha": data_line.get_alpha(),
            }
    all_maxes.update(container_dict)
    handles, labels = ax.get_legend_handles_labels()
    if sort:
        # Get the right order of line names
        label_order_desc = [
            k
            for k, v in sorted(
                all_maxes.items(), key=lambda item: item[1]["max"], reverse=True
            )
        ]
        # Now get the existing legend
        # Get the indexes that the order we want corresponds to in the existing labels/handles
    else:
        label_order_desc = labels
    # Filter to only labels that were successfully tracked in all_maxes;
    # some artist types (e.g. fill_between bands) produce legend entries but
    # aren't handled above, so they won't have a color entry.
    label_order_desc = [k for k in label_order_desc if k in all_maxes]
    order = [labels.index(x) for x in label_order_desc]
    # Need to order the *handles* correctly so matplotlib can connect them to the actual lines,
    # even though we hide them
    labelcolors = [all_maxes[k]["color"] for k in label_order_desc]
    if use_alphas:
        new_labelcolors = []
        for k_ix, k in enumerate(label_order_desc):
            this_color = list(labelcolors[k_ix])
            this_color[3] = all_maxes[k]["alpha"]
            new_labelcolors.append(tuple(this_color))
        labelcolors = new_labelcolors
    legend = ax.legend(
        [handles[ix] for ix in order],
        [labels[ix] for ix in order],  # This is the same as line_order_desc
        # Hide the handles and make the text color match the line color
        handletextpad=0.0,
        handlelength=0.0,
        handleheight=0.0,
        markerscale=0.0,
        labelcolor=labelcolors,
        #         scatterpoints=0,
        **kwargs,
    )
    # Get rid of any remaining little rectangular blips
    for item in legend.legend_handles:
        item.set_visible(False)
    return ax


def contour_legend(contour_set, **kwargs):
    handles = [x for x in contour_set.legend_elements()[0]]
    return contour_set.figure.legend(
        handles=handles,
        # Hide the handles and make the text color match the line color
        handletextpad=0.0,
        handlelength=0.0,
        handleheight=0.0,
        markerscale=0.0,
        # labelcolor='black',
        # facecolor=contour_edgecolors,
        ncol=len(handles),
        loc="upper left",
        **kwargs,
    )


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


def add_row_header(ax, text, pad=None, **kwargs):
    """
    Add a row header to the left of an axes with automatic spacing.

    Args:
        ax: Matplotlib axes object
        text: Text for the row header (can be multi-line)
        pad: Manual padding offset. If None, calculated automatically
        **kwargs: Additional arguments passed to annotate()

    Returns:
        The annotation object
    """
    # Split text into lines to handle multi-line text
    lines = text.split("\n") if isinstance(text, str) else [str(text)]
    n_lines = len(lines)

    # Calculate automatic padding based on various factors
    if pad is None:
        # Base padding from ylabel
        base_pad = ax.yaxis.labelpad if hasattr(ax.yaxis, "labelpad") else 5

        # Additional padding for tick labels
        tick_width = 0

        # For log-scaled axes, we need to handle tick labels differently
        # Force the axis to update its ticks before we calculate widths
        ax.figure.canvas.draw_idle()

        if ax.yaxis.get_ticklabels():
            # Get the actual rendered width of tick labels in display coordinates
            # This works better than counting characters, especially for log scales
            try:
                # Get the bounding boxes of all tick labels
                tick_bboxes = []
                for label in ax.yaxis.get_ticklabels():
                    if label.get_text():  # Only consider non-empty labels
                        # Get bbox in figure coordinates
                        bbox = label.get_window_extent(
                            renderer=ax.figure.canvas.get_renderer()
                        )
                        tick_bboxes.append(bbox)

                if tick_bboxes:
                    # Find the maximum width in points
                    max_width = max(bbox.width for bbox in tick_bboxes)
                    # Convert from pixels to points (assuming 72 DPI for points)
                    tick_width = (
                        max_width * 0.75
                    )  # Scale factor to convert pixels to approximate points
            except (AttributeError, TypeError):
                # Fallback to character counting if rendering fails
                tick_texts = [
                    label.get_text() for label in ax.yaxis.get_ticklabels()
                ]
                if tick_texts:
                    max_tick_len = max(len(str(t)) for t in tick_texts if t)
                    tick_width = (
                        max_tick_len * 8
                    )  # Approximate character width in points

        # Additional padding for ylabel if present
        ylabel_width = 0
        if ax.yaxis.get_label().get_text():
            ylabel_width = 20  # Approximate width for rotated ylabel

        # Extra padding for multi-line text
        multiline_pad = (n_lines - 1) * 10 if n_lines > 1 else 0

        # Calculate total padding
        pad = base_pad + tick_width + ylabel_width + multiline_pad + 15

    # Handle multi-line text positioning
    if n_lines > 1:
        # For multi-line text, adjust vertical alignment
        va = kwargs.get("va", "center")
        # Join lines back together for display
        display_text = "\n".join(lines)
    else:
        va = kwargs.get("va", "center")
        display_text = text

    # Set up annotation arguments with automatic spacing
    annotation_kwargs = {
        "xy": (0, 0.5),
        "xytext": (-pad, 0),
        "xycoords": "axes fraction",  # More reliable than ax.yaxis.label
        "textcoords": "offset points",
        "rotation": 90,
        "fontsize": kwargs.get("fontsize", 16),  # Slightly smaller default
        "ha": "center",
        "va": va,
        "fontweight": "bold",
    }

    # Update with user-provided kwargs
    annotation_kwargs.update(kwargs)

    return ax.annotate(display_text, **annotation_kwargs)


def shifted_colormap(cmap, new_range, n=256):
    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]
    colors_list = cmap(np.linspace(new_range[0], new_range[1], n))
    return colors.LinearSegmentedColormap.from_list("new", colors_list)


# Define some shifted colormaps
shifted_blues = shifted_colormap("Blues", (0.2, 1.0))
shifted_greens = shifted_colormap("Greens", (0.3, 1.0))
shifted_oranges = shifted_colormap("Oranges", (0.3, 1.0))


def plot_sounding(ds: xr.Dataset) -> matplotlib.figure.Figure:

    # Exclude the fake level if present
    if ds["z"].values[0] < 0:
        ds = ds.isel(z=slice(1, len(ds["z"])))

    this_ds_mean = ds.squeeze().mean(["x", "y"])

    fig = plt.figure(figsize=(9, 9))
    skewt = SkewT(fig, rotation=30)
    skewt.plot(
        this_ds_mean["P"].values,
        (this_ds_mean["T"].values * units("K")).to("degC").magnitude,
        "r",
    )
    skewt.plot(
        this_ds_mean["P"].values,
        (this_ds_mean["dewpoint"].values * units("K")).to("degC"),
        "blue",
    )
    # fig.suptitle(sounding_time)

    # Calculate and plot parcel profile
    parcel_path = mpc.parcel_profile(
        this_ds_mean["P"].values * units.hPa,
        this_ds_mean["T"].values[0] * units.K,
        this_ds_mean["dewpoint"].values[0] * units.K,
    )
    skewt.plot(
        this_ds_mean["P"].values,
        parcel_path,
        color="grey",
        linestyle="dashed",
        linewidth=2,
    )

    # Create a hodograph
    ax_hod = inset_axes(skewt.ax, "40%", "40%", loc=1)
    h = Hodograph(ax_hod, component_range=35.0)
    h.add_grid(increment=10)
    h.plot_colormapped(
        this_ds_mean["UC"].values,
        this_ds_mean["VC"].values,
        this_ds_mean["z"].values * units("m"),
    )

    skewt.ax.set_xlabel("Temperature (°C)")
    skewt.ax.set_ylabel("Pressure (hPa)")

    ax_hod.set_xlabel("U (m/s)")
    ax_hod.set_ylabel("V (m/s)")

    fig.suptitle("Initial sounding")

    return fig


def fig_multisave(
    fig: matplotlib.figure.Figure,
    name: Union[str, PathLike],
    dirs: Union[PathLike, List[PathLike]],
    no_title_version: bool = False,
    resize_to_width: Optional[float] = None,
    extensions: Optional[List[str]] = None,
) -> matplotlib.figure.Figure:
    """
    Save a figure into multiple directories (with the same filename). That's easily accomplished
    with a for loop on its own, but the no_title_version flag also allows for the creation of two copies
    of the figure in each directory: one with its suptitle (if present), and one without. These copies
    will be suffixed with 'title-yes' and 'title-no' respectively.
    """
    # Make the dirs argument a list if a single directory was passed
    if not isinstance(dirs, list):
        dirs = [dirs]

    # Handle mutable default argument
    if extensions is None:
        extensions = [".pdf", ".png"]

    # Clean up the file extensions
    if isinstance(extensions, str):
        extensions = [extensions]
    extensions = [
        "." + extension if extension[0] != "." else extension
        for extension in extensions
    ]

    # Remove any file extensions from name
    name = Path(name).stem

    if resize_to_width:
        # Resize the figure with the same aspect ratio if we should
        current_size = fig.get_size_inches()
        current_ar = current_size[1] / current_size[0]
        fig.set_size_inches(resize_to_width, current_ar * resize_to_width)

    # Ignore the no_title_version argument if the figure doesn't have a suptitle
    if not fig._suptitle:
        no_title_version = False
    for this_dir in dirs:
        suffix = "_title-yes" if no_title_version else ""
        for extension in extensions:
            fig.savefig(this_dir.joinpath(name + suffix + extension))
    # Make a no title version if we're doing that
    if no_title_version:
        fig._suptitle.remove()
        fig._suptitle = None
        for this_dir in dirs:
            for extension in extensions:
                fig.savefig(this_dir.joinpath(name + f"_title-no{extension}"))
    return fig


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


def share_axes(axs, x=True, y=True):
    # Just share them all with the first one
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    base_ax = axs[0]
    for ax in axs[1:]:
        if x:
            ax.sharex(base_ax)
        if y:
            ax.sharey(base_ax)
    return axs


def scale_axes_ticks(ax, scale=1000, x=True, y=True):
    # Scale the horizontal axes to km rather than m
    ticks = mpl.ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x / scale))
    if x:
        ax.xaxis.set_major_formatter(ticks)
    if y:
        ax.yaxis.set_major_formatter(ticks)


def prepend_axes_letters(
    axs: Union[matplotlib.axes.Axes, np.ndarray],
    start_letter: str = "a",
    format_string: str = "({}) ",
) -> Union[matplotlib.axes.Axes, np.ndarray]:
    """
    Prepend letters to the title of each axes in row-major order.

    Given a set of axes (as returned by plt.subplots), this function prepends
    sequential letters to each axes title. The lettering proceeds in row-major
    order (left-to-right within each row, then top-to-bottom across rows).

    Args:
        axs: Single axes or array of axes from plt.subplots()
        start_letter: The starting letter (default: 'a')
        format_string: Format string for the letter label. Use {} as placeholder
                      for the letter (default: '({}) ' produces '(a) ', '(b) ', etc.)

    Returns:
        The modified axes (same type as input)

    Examples:
        >>> fig, axs = plt.subplots(2, 3)
        >>> prepend_axes_letters(axs)
        # Produces: (a), (b), (c) in first row
        #           (d), (e), (f) in second row

        >>> fig, ax = plt.subplots()
        >>> ax.set_title("My Plot")
        >>> prepend_axes_letters(ax)
        # Produces: (a) My Plot
    """
    # Handle single axes case
    if isinstance(axs, matplotlib.axes.Axes):
        axs_flat = [axs]
    else:
        # Flatten in row-major order (C-style)
        axs_flat = axs.flatten() if isinstance(axs, np.ndarray) else axs

    # Get the starting letter as an integer (a=0, b=1, etc.)
    start_ord = ord(start_letter.lower())

    # Iterate through axes in row-major order
    for i, ax in enumerate(axs_flat):
        # Generate the letter label
        letter = chr(start_ord + i)
        label = format_string.format(letter)

        # Get current title and prepend the letter
        current_title = ax.get_title()
        new_title = label + current_title
        ax.set_title(new_title)

    return axs


def gif_from_pngs(gif_path, pngs_fpaths, fps=24):
    import imageio

    with imageio.get_writer(str(gif_path), mode="I", fps=fps) as writer:
        for png_fpath in tqdm(pngs_fpaths):
            image = imageio.imread(str(png_fpath))
            writer.append_data(image)


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


def n_subplots(
    n=None,
    ncols=None,
    nrows=None,
    width_per_axes=3,
    height_per_axes=3,
    width_padding=0,
    height_padding=1,
    titles=[],
    **kwargs,
):
    if not n and not (ncols and nrows):
        # Have to pass one
        raise ValueError("Must pass either `n` or (`ncols` and `nrows`)")

    if not nrows and not ncols:
        # Pick the smallest square that fits, then reduce rows to eliminate empty space
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))
    elif nrows and not ncols:
        ncols = np.ceil(n / nrows)
    elif ncols and not nrows:
        nrows = np.ceil(n / ncols)

    nrows = int(nrows)
    ncols = int(ncols)

    figsize = (
        width_per_axes * ncols + width_padding,
        height_per_axes * nrows + height_padding,
    )
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False, **kwargs
    )
    # Hide unnecessary axes
    for ax in axs.flatten()[n:]:
        ax.axis("off")
    axs = axs[:n]

    # Set the axes titles, if passed
    for title_ix, title in enumerate(titles):
        axs.flatten()[title_ix].set_title(title)

    return fig, axs


def label_columns(axs, labels):
    # Set each column's title from the corresponding label
    for ix in range(len(labels)):
        axs.flatten()[ix].set_title(labels[ix])


def label_rows(axs, labels, **kwargs):
    # Add a row header to the left-most axis of each row
    for ix in range(len(labels)):
        add_row_header(axs[ix, 0], labels[ix], **kwargs)
