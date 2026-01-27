# cloudy-core

Core utilities and type definitions for cloudy atmospheric modeling packages.

## Overview

`cloudy-core` provides shared utilities and type definitions that are used across the cloudy ecosystem of atmospheric modeling and visualization packages. This package serves as a common dependency for other cloudy packages.

## Features

- **Type Definitions**: Common type aliases for paths, configurations, and Blender objects
- **Datetime Utilities**: Functions for parsing and formatting datetime objects
- **File I/O**: Convenient file reading/writing functions for common scientific data formats (NetCDF, CSV, pickle, NumPy)
- **Data Utilities**: Helper functions for working with arrays, key-value pairs, and data structures

## Installation

Install from a local copy:

```bash
pip install -e /path/to/cloudy-core
```

## Usage

```python
from cloudy_core import PathLike, dt_to_str, read_file, TwoWayDict
from cloudy_core.utils import current_dt_str
from cloudy_core.types_core import ConfigDict

# Use datetime utilities
timestamp = current_dt_str()

# Read scientific data files
data = read_file("data.nc")  # Automatically uses xarray for .nc files

# Work with nested dictionaries
nested = TwoWayDict({'a': {'x': 1, 'y': 2}, 'b': {'x': 3, 'y': 4}})
```

## Type Definitions

- `PathLike`: Union of `str` and `Path` for file paths
- `ConfigDict`: Dictionary for storing arbitrary configuration data
- `BlenderObject`: Type alias for Blender objects (gracefully handles non-Blender environments)
- `BlenderCollection`: Type alias for Blender collections
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

- numpy
- typing-extensions (Python < 3.10)

Optional dependencies for full functionality:
- xarray (for NetCDF support)
- pandas (for CSV support and enhanced datetime parsing)

## License

MIT License
