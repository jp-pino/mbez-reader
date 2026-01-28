# mbez-reader

A Python tool for reading and processing `.mbez` files - compressed sonar log files from Blueye underwater robotics systems.

> **Note**: This project was developed through pair-programming with GitHub Copilot, an AI coding assistant.

This project demonstrates how to:

- Parse `.mbez` files using the `blueye.sdk` library
- Extract multibeam sonar ping telemetry data
- Transform polar coordinate sonar data to Cartesian fan-shaped images
- Visualize sonar data for analysis and image processing

## Features

- **Parse MBEZ Files**: Read gzip-compressed binary log files containing protobuf-encoded telemetry
- **Extract Sonar Pings**: Extract `MultibeamPingTel` messages from log streams
- **Fan Transformation**: Convert polar (range, bearing) sonar data to Cartesian coordinates
- **Batch Processing**: Process multiple pings and save visualizations
- **CLI Tool**: Command-line interface for easy processing

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Clone the repository
git clone https://github.com/jp-pino/mbez-reader.git
cd mbez-reader

# Install dependencies
uv sync

# Install in development mode (optional)
uv pip install -e .
```

## Usage

### Command Line

Process sonar data from an `.mbez` file:

```bash
# Process a single ping (the first one by default)
uv run python mbez_reader.py mbez/20241022_135915.oculus.mbez

# Process multiple pings starting from index 100
uv run python mbez_reader.py mbez/20241022_135915.oculus.mbez --start 100 -n 10

# Save to a custom output directory
uv run python mbez_reader.py mbez/20241022_135915.oculus.mbez -o results/

# Process without showing the visualization
uv run python mbez_reader.py mbez/20241022_135915.oculus.mbez --no-show
```

If installed with the project script entry point:

```bash
fan-transform mbez/20241022_135915.oculus.mbez --start 0 -n 5
```

### Python API

```python
from pathlib import Path
from mbez_reader import extract_pings, create_fan_image, visualize_transformation
import numpy as np

# Extract pings from file
mbez_file = Path("mbez/20241022_135915.oculus.mbez")
pings = extract_pings(mbez_file, start_index=0, count=1)

# Get the first ping
ping = pings[0]

# Convert ping data to numpy array
raw_data = np.asarray(ping.ping_data).reshape(
    (ping.number_of_ranges, ping.number_of_beams)
)
bearings = np.asarray(ping.bearings)

# Create fan-transformed image
fan_image = create_fan_image(raw_data, bearings)

# Visualize
visualize_transformation(
    raw_data,
    bearings,
    fan_image,
    output_path="output.png"
)
```

### Jupyter Notebook

An example Jupyter notebook is provided for interactive exploration:

```bash
# Start Jupyter
uv run jupyter notebook

# Open mbez-reader.ipynb
```

## Development

### Running Tests

```bash
# Install dev dependencies
uv sync

# Run all tests
uv run pytest tests/

# Run with verbose output
uv run pytest tests/ -v

# Run with coverage report
uv run pytest tests/ -v --cov=mbez_reader --cov-report=term-missing

# Run a specific test
uv run pytest tests/test_mbez_reader.py::test_create_fan_image_basic -v
```

### Project Structure

```
mbez-reader/
├── mbez/                      # Example .mbez data files (Git LFS)
├── tests/                     # Test suite
│   ├── __init__.py
│   └── test_mbez_reader.py
├── .github/
│   └── workflows/
│       └── test.yml           # GitHub Actions CI
├── mbez_reader.py             # Main module
├── pyproject.toml             # Project configuration
└── README.md
```

## File Format

`.mbez` files are gzip-compressed binary log files containing:

- Format: `YYYYMMDD_HHMMSS.oculus.mbez` (timestamp-based naming)
- Content: Protobuf messages serialized via `blueye.protocol`
- Primary message: `MultibeamPingTel` (multibeam sonar ping telemetry)

### MultibeamPingTel Fields

- `number_of_ranges`: Number of range bins (height of raw data)
- `number_of_beams`: Number of beams (width of raw data)
- `ping_data`: Raw intensity data as flat array
- `bearings`: Angle of each beam in radians

## Algorithm: Polar to Cartesian Transformation

The fan transformation converts sonar data from polar coordinates (range, bearing) to Cartesian coordinates (x, y):

1. **Input**: Raw data matrix (ranges × beams) + bearing angles
2. **Output**: Fan-shaped image where each pixel maps to a polar coordinate
3. **Method**:
   - Create meshgrid of output image coordinates
   - Compute polar coordinates (r, θ) for each pixel
   - Find closest bearing angle for each θ
   - Sample intensity from raw data at (r, bearing_index)
   - Uses vectorized numpy operations for performance

This creates a fan-shaped visualization suitable for image processing and analysis.

## Dependencies

- **blueye-protocol**: Protobuf message definitions
- **blueye-sdk**: LogStream parser for `.mbez` files
- **numpy**: Numerical operations and array handling
- **matplotlib**: Visualization and image generation

## Known Limitations

- Current implementation loads entire `.mbez` file into memory
- Not optimized for very large files (>1GB)
- This is an example/demonstration project

## Contributing

Contributions welcome! Please ensure tests pass before submitting PRs:

```bash
uv run pytest tests/ -v
```

## License

This project is provided as an example for working with Blueye sonar data.
