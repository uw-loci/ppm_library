# PPM Library

Image processing library for polarized light microscopy (PPM) and general microscopy imaging.

> **Part of the QPSC (QuPath Scope Control) system**
> For complete installation instructions, see: https://github.com/uw-loci/QPSC
>
> **Note:** This library can also be used standalone for general microscopy image processing.

## Features

- **PPM Processing**: Polarizer calibration, birefringence analysis, rotation sensitivity diagnostics
- **Image Processing**: Background correction, flatfield correction, tissue detection
- **Debayering**: CPU and GPU-based Bayer pattern demosaicing
- **Camera Calibration**: JAI camera white balance calibration
- **TIFF I/O**: TIFF writing with metadata support

## Installation

**Part of [QPSC (QuPath Scope Control)](https://github.com/uw-loci/QPSC)**

**Requirements:**
- Python 3.9 or later
- pip (Python package installer)
- Git (for `pip install git+https://...` commands)

This library has no dependencies on other QPSC packages and can be used standalone.
See the [QPSC Installation Guide](https://github.com/uw-loci/QPSC#automated-installation-windows) for complete QPSC setup.

### Quick Install (from GitHub)

**Standard installation:**
```bash
pip install git+https://github.com/uw-loci/ppm_library.git
```

**With GPU support:**
```bash
pip install "git+https://github.com/uw-loci/ppm_library.git#egg=ppm-library[gpu]"
```

### Development Install (editable mode)

```bash
git clone https://github.com/uw-loci/ppm_library.git
cd ppm_library
pip install -e .

# Or with GPU support:
pip install -e ".[gpu]"
```

**For automated setup**, use the [QPSC setup script](https://github.com/uw-loci/QPSC/blob/main/PPM-QuPath.ps1).

### Troubleshooting Installation

#### Problem: `ModuleNotFoundError: No module named 'ppm_library'`

**Cause:** Editable install issue with package structure.

**Solution:**

The repository has been updated to fix this issue. If you encounter this error:

1. **Update to latest version:**
   ```bash
   cd ppm_library
   git pull
   ```

2. **Verify `pyproject.toml` has the correct configuration:**
   ```toml
   [tool.hatch.build.targets.wheel]
   packages = ["."]
   ```

3. **Reinstall in editable mode:**
   ```bash
   pip install -e . --force-reinstall --no-deps
   ```

4. **Verify installation:**
   ```bash
   pip show ppm-library
   ```

#### Problem: Circular dependency error

**Symptom:** Error importing `EmptyRegionDetector` from `microscope_control`

**Solution:** This has been fixed in the latest version. The problematic import has been removed from `__init__.py` to maintain standalone design. If needed, import directly:
```python
from microscope_control.autofocus.tissue_detection import EmptyRegionDetector
```

For more troubleshooting, see the [QPSC Installation Guide](https://github.com/uw-loci/QPSC#troubleshooting-python-package-installation).

## Quick Start

```python
from ppm_library.imaging.background import BackgroundCorrectionUtils
from ppm_library.debayering import CPUDebayer

# Background correction
corrector = BackgroundCorrectionUtils()
corrected = corrector.apply_flatfield(raw_image, background_image)

# Debayering
debayer = CPUDebayer(pattern='RGGB')
rgb_image = debayer.debayer(bayer_image)
```

## Testing

This package includes two types of testing tools:

### Hardware Diagnostic Tools

PPM-specific diagnostic and calibration tools are located in the source modules:
- **`ppm/birefringence_test.py`** - PPM birefringence maximization testing
- **`ppm/sensitivity_test.py`** - PPM rotation sensitivity testing

These tools are called from the QuPath QPSC extension GUI during PPM microscope setup. They:
- Require live microscope hardware and server connection
- Generate diagnostic plots, visualizations, and CSV data
- Test PPM signal optimization and angular precision
- Validate mechanical repeatability and intensity sensitivity

**Not intended for automated CI/CD** - these are interactive diagnostic tools for hardware characterization.

### Automated Unit Tests

Automated pytest-compatible unit tests are located in the `tests/` directory:
- **`tests/test_debayering.py`** - Tests for Bayer pattern debayering (all 4 patterns)
- **`tests/test_background_correction.py`** - Tests for flat-field correction and background detection

These tests:
- Run without hardware (use synthetic test images)
- Can be integrated into CI/CD pipelines
- Test pure-function image processing components

**Running Unit Tests:**

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run specific test file
pytest tests/test_debayering.py

# Run with coverage report
pytest --cov=ppm --cov-report=html

# View coverage report
open htmlcov/index.html  # or xdg-open on Linux
```

**Test Coverage:**

Current automated tests achieve ~70-80% coverage for testable components:
- ✅ Debayering (all 4 Bayer patterns: RGGB, GRBG, GBRG, BGGR)
- ✅ Background correction (flat-field division & subtraction methods)
- ✅ Background mode detection
- ⏸️ PPM calibration (requires hardware, use diagnostic tools)
- ⏸️ Birefringence computation (complex - future HIGH priority test)

## License

MIT License
