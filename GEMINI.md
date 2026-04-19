# GEMINI.md - pyfastspm

## Project Overview
`pyfastspm` is a Python package designed for loading, processing, and exporting scanning probe microscopy (SPM) movies acquired with the FAST module. It provides tools for artefact removal (creep, drift, noise), image correction, and tracking.

### Main Technologies
- **Language:** Python (>= 3.10)
- **Data Handling:** `h5py`, `numpy`, `scipy`
- **Image Processing:** `scikit-image`, `pillow`
- **Visualization:** `matplotlib`
- **Packaging:** `setuptools`, `setuptools_scm`
- **Documentation:** `sphinx`

### Architecture
The project is structured around the `FastMovie` class (`pyfastspm/fast_movie.py`), which acts as the primary interface for interacting with `.h5` movie files. Core functionality is divided into submodules:
- `pyfastspm.artefact_removal`: Contains logic for drift, creep, FFT filtering, and general image corrections.
- `pyfastspm.tools`: Helper functions for file handling, exporting (FFMPEG, gsf), and frame rendering.
- `pyfastspm.tracking`: Tools for pixel-level trace analysis.

## Building and Running

### Development Setup
To install the package in editable mode with development dependencies:
```bash
pip install -e .
```

### Running Tests
The project uses `pytest` for testing, including notebook validation via `nbmake`.
```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -vx

# Run notebook tests
pytest --nbmake examples/pyfastspm_converter.ipynb
```

### Building the Package
```bash
python -m build
```

### Building Documentation
Documentation is built using Sphinx. The `docs/Makefile` includes a custom target that runs `sphinx-apidoc` before building.
```bash
cd docs
make html
```

## Development Conventions

### Coding Style
- **Formatter:** `black`
- **Import Sorting:** `isort` (profile: black)
- **Linting:** `flake8` (configured in `pyproject.toml`)
- **Type Hints:** Encouraged where appropriate for clarity and IDE support.

### Versioning
The project uses `setuptools_scm` to automatically manage versions based on Git tags. Version information is stored in `pyfastspm/_version.py` (generated during build).

### Logging
Logging is configured at the package level in `pyfastspm/__init__.py`. Individual files should use `logging.getLogger(__name__)`. The `FastMovie` class also supports logging processing operations to a `.log` file alongside the source data.

### Testing Practices
- Tests are located in the `tests/` directory.
- Test files use `pytest` and often use `parametrize` for coverage across different data shapes (single vs. multi-frame).
- Example data for tests can be found in `examples/`.
