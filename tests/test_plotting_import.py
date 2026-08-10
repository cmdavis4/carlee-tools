"""Test plotting module imports."""

from carlee_tools.plotting.colors import (
    get_cmap,
    get_nth_color,
    shifted_colormap,
)


def test_plotting_module_imports():
    """Test that plotting subpackage imports correctly."""
    from carlee_tools import plotting

    assert plotting is not None


def test_plotting_functions():
    """Test that key plotting functions are importable."""
    from carlee_tools.plotting import (
        clean_legend,
        fig_multisave,
        prepend_axes_letters,
    )

    assert callable(clean_legend)
    assert callable(fig_multisave)
    assert callable(get_cmap)
    assert callable(prepend_axes_letters)
    assert callable(get_nth_color)
    assert callable(shifted_colormap)


def test_plotting_from_carlee_tools():
    """Test importing plotting from top-level carlee_tools."""
    import carlee_tools

    assert hasattr(carlee_tools, "plotting")
    assert hasattr(carlee_tools.plotting, "clean_legend")
    assert hasattr(carlee_tools.plotting, "fig_multisave")
