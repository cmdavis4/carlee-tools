"""Test plotting module imports."""

def test_plotting_module_imports():
    """Test that plotting subpackage imports correctly."""
    from carlee_tools import plotting
    assert plotting is not None


def test_plotting_functions():
    """Test that key plotting functions are importable."""
    from carlee_tools.plotting import (
        clean_legend,
        fig_multisave,
        plot_sounding,
        get_cmap,
        prepend_axes_letters,
        format_t_str,
        get_nth_color,
        shifted_colormap,
    )
    assert callable(clean_legend)
    assert callable(fig_multisave)
    assert callable(plot_sounding)
    assert callable(get_cmap)
    assert callable(prepend_axes_letters)
    assert callable(format_t_str)
    assert callable(get_nth_color)
    assert callable(shifted_colormap)


def test_plotting_from_carlee_tools():
    """Test importing plotting from top-level carlee_tools."""
    import carlee_tools
    assert hasattr(carlee_tools, 'plotting')
    assert hasattr(carlee_tools.plotting, 'clean_legend')
    assert hasattr(carlee_tools.plotting, 'fig_multisave')
