# carlee-tools

Core utilities, types, and plotting for atmospheric science.

## Overview

`carlee-tools` provides shared utilities and type definitions that are used across the carlee ecosystem of atmospheric modeling and visualization packages. This package serves as a common dependency for other carlee packages.

## Features

- **Type Definitions**: Common type aliases for paths and configurations
- **Datetime Utilities**: Functions for parsing and formatting datetime objects
- **File I/O**: Convenient file reading/writing functions for common scientific data formats (NetCDF, CSV, pickle, NumPy)
- **Data Utilities**: Helper functions for working with arrays, key-value pairs, and data structures
- **Plotting Utilities**: Matplotlib helpers for creating publication-quality atmospheric science figures

## Installation

Install from PyPI:

```bash
pip install carlee-tools
```

Or install from source:

```bash
pip install -e /path/to/carlee-tools
```

## Usage

```python
import carlee_tools
from carlee_tools import PathLike, dt_to_str, read_file, TwoWayDict
from carlee_tools.utils import current_dt_str
from carlee_tools.types_carlee_tools import ConfigDict
from carlee_tools.plotting import clean_legend, get_cmap

# Use datetime utilities
timestamp = current_dt_str()

# Read scientific data files
data = read_file("data.nc")  # Automatically uses xarray for .nc files

# Work with nested dictionaries
nested = TwoWayDict({'a': {'x': 1, 'y': 2}, 'b': {'x': 3, 'y': 4}})

# Create clean matplotlib legends
clean_legend()
```

## Type Definitions

- `PathLike`: Union of `str` and `Path` for file paths
- `NumpyNumeric`: Union of NumPy integer and floating types

## Key Utilities

### Datetime Functions
- `dt_to_str()`: Convert datetime-like objects to formatted strings
- `str_to_dt()`: Parse datetime strings with flexible format detection
- `current_dt_str()`: Get current time as formatted string

### File I/O
- `read_file()`: Automatically detect file type and read with appropriate library
- `write_file()`: Write data with format detection

### Data Structures
- `TwoWayDict`: Dictionary accessor that allows indexing by outer or inner keys

## Dependencies

Core dependencies:
- numpy
- matplotlib>=3.0
- xarray
- metpy
- tqdm
- imageio
- typing-extensions (Python < 3.10)

Optional dependencies:
- pandas (for enhanced CSV support)
- netcdf4 (for NetCDF file writing)

## License

MIT License
