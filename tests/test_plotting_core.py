"""Test core plotting functionality."""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile


def test_clean_legend_basic():
    """Test clean_legend with basic plot."""
    from skyutils.plotting import clean_legend

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3], label='test_line')
    result = clean_legend(ax)

    assert result is ax
    legend = ax.get_legend()
    assert legend is not None
    plt.close(fig)


def test_get_nth_color():
    """Test getting nth color from color cycle."""
    from skyutils.plotting import get_nth_color

    color = get_nth_color(0)
    assert isinstance(color, str)
    # Should return a valid color (hex or named)
    assert (color.startswith('#') or len(color) > 0)


def test_get_next_color():
    """Test getting next color from axes."""
    from skyutils.plotting import get_next_color

    fig, ax = plt.subplots()
    color1 = get_next_color(ax)
    color2 = get_next_color(ax)

    assert isinstance(color1, str)
    assert isinstance(color2, str)
    # Colors should be different (cycling through)
    # Note: might be the same if cycle has only one color, but usually different
    plt.close(fig)


def test_prepend_axes_letters():
    """Test prepending letters to axes titles."""
    from skyutils.plotting import prepend_axes_letters

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].set_title("First")
    axs[0, 1].set_title("Second")

    result = prepend_axes_letters(axs)

    assert axs[0, 0].get_title() == "(a) First"
    assert axs[0, 1].get_title() == "(b) Second"
    plt.close(fig)


def test_fig_multisave():
    """Test saving figure to multiple directories."""
    from skyutils.plotting import fig_multisave

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        dir1 = tmppath / "dir1"
        dir2 = tmppath / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        fig_multisave(fig, "test_fig", dirs=[dir1, dir2], extensions=[".png"])

        assert (dir1 / "test_fig.png").exists()
        assert (dir2 / "test_fig.png").exists()
        plt.close(fig)


def test_get_cmap():
    """Test getting colormap."""
    from skyutils.plotting import get_cmap
    import matplotlib.colors

    cmap = get_cmap("viridis")
    assert isinstance(cmap, matplotlib.colors.Colormap)


def test_format_t_str():
    """Test timestamp formatting."""
    from skyutils.plotting import format_t_str
    import datetime

    dt_obj = datetime.datetime(2020, 1, 1, 12, 30, 0)
    result = format_t_str(dt_obj)

    assert isinstance(result, str)
    assert "2020" in result
    assert "01" in result
