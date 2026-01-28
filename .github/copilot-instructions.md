# MBEZ Reader - AI Coding Agent Instructions

## Project Overview

This is a Python example project demonstrating how to read and process `.mbez` files (compressed sonar log files from Blueye underwater robotics systems). The project uses Jupyter notebooks for interactive data exploration.

## Architecture & Data Flow

### File Format

- `.mbez` files are gzip-compressed binary log files containing telemetry data
- Format: `YYYYMMDD_HHMMSS.oculus.mbez` (timestamp-based naming)
- Contains protobuf messages serialized via the `blueye.protocol` library
- Primary message type of interest: `MultibeamPingTel` (multibeam sonar ping telemetry)

### Core Dependencies

- **blueye.protocol**: Protobuf message definitions for Blueye telemetry
- **blueye.sdk.logs.LogStream**: Generator-based streaming parser for `.mbez` files
- Python 3.13+ with virtual environment (`.venv/`)

## Key Patterns & Workflows

### Reading MBEZ Files

The standard pattern for parsing `.mbez` files:

```python
from blueye.sdk.logs import LogStream
from pathlib import Path

def parse_logfile(log: Path) -> LogStream:
    log_bytes = b""
    with open(log, "rb") as f:
        log_bytes = f.read()
    return LogStream(log_bytes)

# Use as generator
generator = parse_logfile(Path("file.mbez"))
for unix_ts, delta, msg_type, msg in generator:
    # Process messages
```

**Important**: This approach loads the entire file into memory (`f.read()`), which is not optimal for large files (100MB+). This is acknowledged in the notebook as a basic illustration.

### Message Filtering Pattern

To extract specific message types from the log stream:

```python
unix_ts, delta, msg_type, msg = next(generator)
while msg_type.__name__ != "MultibeamPingTel":
    unix_ts, delta, msg_type, msg = next(generator)

# Cast to specific type
ping: bp.MultibeamPingTel = msg
```

## Development Environment

### Setup

- Virtual environment located at `.venv/` (Python 3.13)
- **Note**: The venv may not have `pip` installed - use system Python or recreate venv if needed
- Required packages: `blueye-protocol`, `blueye-sdk`

### Running Code

- Primary workflow: Jupyter notebook ([mbez-reader.ipynb](mbez-reader.ipynb))
- Execute cells sequentially to parse and analyze sonar data
- Sample data file included: `20241022_135915.oculus.mbez` (~112MB)

## Project Conventions

### Code Style

- Type hints used for protobuf message types (e.g., `ping: bp.MultibeamPingTel`)
- Pathlib preferred over string paths
- Generator pattern for streaming large datasets

### File Organization

- Notebook-first approach (`.ipynb` files for exploration)
- Sample data files stored alongside code
- No additional configuration files or build systems

## Known Limitations

- Current implementation loads entire `.mbez` file into memory - not suitable for production use with large files
- This is an example/demo project showing basic usage patterns
