"""Miscellaneous utilities: TwoWayDict, sampling, module reloading, and
collection helpers.

Datetime helpers live in ``dt``; file I/O in ``io``; key-value/filename helpers
in ``kv``; grid-spacing helpers in ``spacing``.
"""

import importlib
import sys
from typing import Any, Dict

import numpy as np

from carlee_tools.types_carlee_tools import is_arraylike

DEFAULT_SEED = 137504983571204


class TwoWayDict:
    """
    A wrapper for nested dictionaries that allows indexing by outer keys or inner keys.

    When indexed by an outer key, returns the corresponding inner dictionary.
    When indexed by an inner key, returns a dictionary mapping outer keys to their
    corresponding inner values for that key.

    Example:
        >>> data = {
        ...     'mesh1': {'type': 'trajectory', 'opacity': 0.5},
        ...     'mesh2': {'type': 'simulation', 'opacity': 0.8},
        ...     'mesh3': {'type': 'trajectory', 'opacity': 0.3}
        ... }
        >>> accessor = NestedDictAccessor(data)
        >>> accessor['mesh1']  # Returns {'type': 'trajectory', 'opacity': 0.5}
        >>> accessor['type']   # Returns {'mesh1': 'trajectory', 'mesh2': 'simulation', 'mesh3': 'trajectory'}
        >>> accessor['opacity'] # Returns {'mesh1': 0.5, 'mesh2': 0.8, 'mesh3': 0.3}
    """

    def __init__(self, nested_dict: Dict[Any, Dict[Any, Any]]):
        """
        Initialize with a dictionary of dictionaries.

        Args:
            nested_dict: Dictionary where values are themselves dictionaries.
        """
        self._data = nested_dict

        self._inner_keys = set()
        for inner_dict in nested_dict.values():
            if isinstance(inner_dict, dict):
                self._inner_keys.update(inner_dict.keys())

    def __getitem__(self, key):
        """
        Get item by key, supporting both outer and inner key access.

        Args:
            key: Either an outer key or an inner key.

        Returns:
            If key is an outer key: the corresponding inner dictionary.
            If key is an inner key: dictionary mapping outer keys to inner values.

        Raises:
            KeyError: If key is found in neither outer nor inner keys.
        """
        # Try as outer key first
        if key in self._data:
            return self._data[key]

        # Try as inner key
        if key in self._inner_keys:
            result = {}
            for outer_key, inner_dict in self._data.items():
                if isinstance(inner_dict, dict) and key in inner_dict:
                    result[outer_key] = inner_dict[key]
            return result

        # Key not found anywhere
        raise KeyError(f"Key '{key}' not found in outer keys or inner keys")

    def __contains__(self, key):
        """Check if key exists in either outer or inner keys."""
        return key in self._data or key in self._inner_keys

    def __iter__(self):
        """Iterate over outer keys."""
        return iter(self._data)

    def keys(self):
        """Return outer keys."""
        return self._data.keys()

    def inner_keys(self):
        """Return all unique inner keys."""
        return self._inner_keys

    def items(self):
        """Return outer key-value pairs."""
        return self._data.items()

    @property
    def values(self):
        return [x for l in self._data.values() for x in l.values()]

    def __repr__(self):
        """Return HTML table representation of the TwoWayDict."""
        if not self._data or not self._inner_keys:
            return "TwoWayDict({})"

        # Get sorted keys for consistent output
        outer_keys = sorted(self._data.keys())
        inner_keys = sorted(self._inner_keys)

        # Start building HTML table
        html = [
            '<table border="1" style="border-collapse: collapse; font-family:'
            ' monospace;">'
        ]

        # Build header row
        html.append("  <thead>")
        html.append("    <tr>")
        html.append(
            '      <th style="padding: 4px 8px; background-color:'
            ' #f0f0f0;"></th>'
        )  # Empty top-left cell
        for inner_key in inner_keys:
            html.append(
                '      <th style="padding: 4px 8px; background-color:'
                f' #f0f0f0;">{str(inner_key)}</th>'
            )
        html.append("    </tr>")
        html.append("  </thead>")

        # Build data rows
        html.append("  <tbody>")
        for outer_key in outer_keys:
            html.append("    <tr>")
            html.append(
                '      <th style="padding: 4px 8px; background-color:'
                f' #f8f8f8;">{str(outer_key)}</th>'
            )

            inner_dict = self._data.get(outer_key, {})
            for inner_key in inner_keys:
                if inner_key in inner_dict:
                    value = inner_dict[inner_key]
                    # Check if value is iterable (but not string) and has length
                    try:
                        if hasattr(value, "__len__") and not isinstance(
                            value, str
                        ):
                            cell_value = str(len(value))
                        else:
                            cell_value = "✓"
                    except TypeError:
                        cell_value = "✓"
                else:
                    cell_value = ""

                html.append(
                    '      <td style="padding: 4px 8px; text-align:'
                    f' center;">{cell_value}</td>'
                )

            html.append("    </tr>")
        html.append("  </tbody>")
        html.append("</table>")

        return "\n".join(html)


def maybe_random_choice(
    arr: np.ndarray, size: int, seed: int = DEFAULT_SEED
) -> np.ndarray:
    """
    Return at most `size` elements from array, handling case where array is smaller than size.

    Args:
        arr: Array to sample from
        size: Maximum number of elements to return
        seed: Random seed for reproducibility

    Returns:
        Array with at most `size` elements
    """
    # Return at most `size` elements from arr; just handles the case where `arr`
    # is smaller than `size`
    if size >= len(arr):
        return arr
    else:
        return np.random.default_rng(seed=seed).choice(
            arr, size=size, replace=False
        )


def fps(
    simulation_minutes_per_second: float, simulation_time_per_frame: Any
) -> float:
    """
    Calculate frames per second from simulation parameters.

    Args:
        simulation_minutes_per_second: Simulation time rate
        simulation_time_per_frame: Time duration per frame

    Returns:
        Frames per second
    """
    simulation_seconds_per_frame = simulation_time_per_frame.nanos / 1e9
    return (simulation_minutes_per_second * 60) / simulation_seconds_per_frame


def recursive_reload(module: Any, silent=False) -> None:
    """Recursively reload a module and every already-imported submodule of it.

    Reloading only the top-level package is enough to pick up edits made
    anywhere in its subtree.

    The subtlety this solves: `importlib.reload()` re-executes a module *in
    place*, but it does NOT rebind the names that *other* modules imported via
    `from x import y` — those still point at the old objects. For such
    references to refresh, a module must be reloaded only AFTER every module it
    imports from has itself been reloaded. A naive depth-based ordering gets
    this wrong whenever two modules at the same nesting depth import from each
    other. Instead we discover the actual import dependencies among the
    package's modules and reload them in dependency-first (topological) order.
    """
    # Name of the top-level module the user asked to reload
    top_level_name = module.__name__

    # Collect the top-level module itself plus every already-imported submodule
    # of it. The exact-name test catches the package; the "name." prefix test
    # catches its submodules (and not sibling packages that merely share a prefix).
    modules_in_package = {
        name: mod
        for name, mod in sys.modules.items()
        if mod is not None
        and (name == top_level_name or name.startswith(top_level_name + "."))
    }

    # A module object, used below to distinguish `import pkg.sub` attributes
    # (whose value IS a module) from `from pkg.sub import f` attributes.
    module_type = type(module)

    # Build the import-dependency graph: for each module, which *other* modules
    # in this package does it hold references into? Two reference styles matter:
    #   - `import pkg.sub`        -> the attribute value is itself a module object
    #   - `from pkg.sub import f` -> the attribute is some object (function,
    #                                class, instance, ...) whose `__module__`
    #                                names the module that defined it
    module_dependencies = {name: set() for name in modules_in_package}
    for name, mod in modules_in_package.items():
        for attribute_value in vars(mod).values():
            if isinstance(attribute_value, module_type):
                # Plain `import`: the attribute is a submodule object
                dependency_name = getattr(attribute_value, "__name__", None)
            else:
                # `from ... import ...`: trace the object back to its home module
                dependency_name = getattr(attribute_value, "__module__", None)
            # Only keep edges to other modules within this package (no self-edges)
            if (
                dependency_name in modules_in_package
                and dependency_name != name
            ):
                module_dependencies[name].add(dependency_name)

    # Topologically sort so each module is reloaded only after the modules it
    # depends on (Kahn's algorithm). Depth = number of dots in the dotted name.
    depth_of = lambda name: name.count(".")
    reload_order = []
    remaining_dependencies = {
        name: set(deps) for name, deps in module_dependencies.items()
    }
    while remaining_dependencies:
        # A module is "ready" once none of its dependencies are still waiting
        ready_to_reload = [
            name
            for name, deps in remaining_dependencies.items()
            if not (deps & remaining_dependencies.keys())
        ]
        if not ready_to_reload:
            # Nothing is free of unsatisfied deps -> an import cycle (Python
            # packages can legally contain these). Break it by taking the
            # deepest remaining module so we always make forward progress.
            ready_to_reload = [max(remaining_dependencies, key=depth_of)]
        # Deepest-first within the ready set gives a stable, sensible order
        # (ready modules never depend on each other, so this is safe)
        ready_to_reload.sort(key=depth_of, reverse=True)
        for name in ready_to_reload:
            reload_order.append(name)
            del remaining_dependencies[name]

    # Reload in dependency order. The top-level package naturally lands last,
    # since everything else sits (directly or transitively) beneath it.
    for name in reload_order:
        try:
            importlib.reload(modules_in_package[name])
        except Exception as e:
            if not silent:
                print(f"Failed to reload {name}: {e}")

    if not silent:
        print(f"Reloaded {top_level_name} ({len(reload_order)} modules)")


def list_if_single(maybe_single):
    return (
        maybe_single
        if is_arraylike(maybe_single)
        else [
            maybe_single,
        ]
    )


def nth_value(listlike_view, n):
    return list(listlike_view)[n]


def first_value(d):
    return nth_value(d.values(), 0)


def first_key(d):
    return nth_value(d.keys(), 0)


def first_item(d):
    return nth_value(d.items(), 0)
