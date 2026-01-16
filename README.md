# PPM Library

Unified library for polarized light microscopy (PPM) - acquisition support and image analysis.

> **Part of the QPSC (QuPath Scope Control) system**
> For complete installation instructions, see: https://github.com/uw-loci/QPSC
>
> **Note:** This library can also be used standalone for general microscopy image processing and PPM analysis.

![PPM Analysis Workflow](docs/images/angle_map_demo.png)

*PPM workflow: Original tissue image (left) to extracted fiber orientation angles (right) - colors indicate fiber direction (0-180 degrees)*

**[See the Complete Walkthrough with Examples](docs/WALKTHROUGH.md)**

## Features

### Acquisition Support
- **Hardware Polarizer Calibration**: Find crossed polarizer positions
- **PPM Rotation Testing**: Sensitivity testing and birefringence analysis
- **Background Correction**: Flatfield correction utilities
- **Debayering**: CPU and GPU-based Bayer pattern demosaicing
- **TIFF I/O**: TIFF writing with metadata support

### Image Analysis
- **Hue-to-Angle Calibration**: Extract fiber angles from PPM images using sunburst calibration slides
- **PPM Image Loading**: Load and analyze PPM images with HSV extraction
- **Fiber Angle Extraction**: Convert hue values to fiber orientation angles (0-180 degrees)
- **White Balance Correction**: Hue shifting and preprocessing
- **Complete Analysis Workflows**: End-to-end PPM analysis with masking and statistics

## Installation

**Requirements:**
- Python 3.10 or later
- pip (Python package installer)
- Git (for `pip install git+https://...` commands)

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
pip install -e ".[dev]"
```

## Quick Start

### Acquisition Support

```python
from ppm_library import BackgroundCorrectionUtils, CPUDebayer

# Background correction
corrector = BackgroundCorrectionUtils()
corrected = corrector.apply_flatfield(raw_image, background_image)

# Debayering
debayer = CPUDebayer(pattern='RGGB')
rgb_image = debayer.debayer(bayer_image)
```

### Image Analysis - Fiber Angle Extraction

```python
from ppm_library import RadialCalibrator, PPMImage

# Step 1: Create calibration from sunburst slide
calibrator = RadialCalibrator(n_spokes=16)
calibration = calibrator.calibrate("sunburst_slide.tif", debug_plot=True)

# Save calibration for later use
calibration.save("my_calibration.npz")

# Step 2: Load PPM sample image
ppm_image = PPMImage.load("sample.tif")

# Step 3: Convert to fiber angles
angle_map = ppm_image.to_angle_map(calibration)

# Step 4: Analyze results
from ppm_library import analyze_ppm

result = analyze_ppm(
    calibration_input="my_calibration.npz",
    ppm_image_path="sample.tif",
    mask_image_path="tissue_mask.tif",
    threshold=128  # Analyze bright regions of mask
)

result.print_summary()
# PPM Analysis Results
# ========================================
# Valid pixels analyzed: 125,432
# Mean fiber angle: 45.23 degrees
# Std deviation: 12.87 degrees
# Calibration R-squared: 0.9876
```

### Hue Correction (White Balance)

```python
from ppm_library import hue_shift, compute_hue_shift_from_reference

# Correct white balance by shifting hue
corrected = hue_shift(image, angle_degrees=15.0)

# Or compute shift automatically from a known reference
shift, measured_hue = compute_hue_shift_from_reference(
    image,
    reference_angle=45.0,  # Known fiber angle in ROI
    roi_mask=reference_roi
)
corrected = hue_shift(image, shift)
```

## Module Reference

### `ppm_library.ppm` - Hardware/Acquisition Support
- `PolarizerCalibrationUtils` - Find crossed polarizer positions
- `PPMRotationSensitivityTester` - Test PPM rotation precision
- `PPMBirefringenceMaximizationTester` - Optimize birefringence signal

### `ppm_library.calibration` - Hue-to-Angle Calibration
- `RadialCalibrator` - Radial sampling for connected sunburst patterns (recommended)
- `SunburstCalibrator` - Region-based segmentation for separated rectangles
- `RadialCalibrationResult` - Calibration data with hue_to_angle() method
- `HistogramCalibration` - Correct optical anisotropy in hue histograms
- `compute_hue_histogram()` - Compute hue histogram from RGB image

### `ppm_library.imaging` - Image Processing
- `PPMImage` - Container for PPM image data with HSV extraction
- `AngleMap` - Fiber angle extraction results with analysis methods
- `load_ppm_image()` - Convenience function to load PPM images
- `hue_shift()` - White balance correction by shifting hue
- `compute_hue_shift_from_reference()` - Auto white balance from reference
- `preprocess_ppm_image()` - Standard Gaussian + median preprocessing
- `BackgroundCorrectionUtils` - Flatfield correction
- `TifWriterUtils` - TIFF writing with metadata

### `ppm_library.analysis` - Complete Workflows
- `analyze_ppm()` - Complete workflow for PPM image analysis
- `PPMAnalysisResult` - Analysis results with statistics and visualization

### `ppm_library.debayering` - Bayer Demosaicing
- `CPUDebayer` - CPU-based Bayer pattern demosaicing
- `GPUDebayer` - GPU-accelerated debayering (requires CuPy)

## Testing

### Automated Unit Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run specific test file
pytest tests/test_debayering.py
pytest tests/test_sunburst.py

# Run with coverage report
pytest --cov=ppm_library --cov-report=html
```

### Hardware Diagnostic Tools

PPM-specific diagnostic tools for hardware characterization:
- `ppm/birefringence_test.py` - PPM birefringence maximization testing
- `ppm/sensitivity_test.py` - PPM rotation sensitivity testing

These are called from the QuPath QPSC extension GUI during microscope setup.

## Visual Workflow

### 1. Calibration Slide

Use a sunburst calibration slide with spokes at known orientations (synthetic phantom shown):

![Synthetic Calibration Phantom](docs/images/synthetic_calibration_phantom.png)

### 2. Create Calibration Model

The calibrator fits a linear regression between hue and angle:

![Calibration Results](docs/images/calibration_updated.png)

### 3. Extract Fiber Angles

Apply calibration to PPM images to get fiber orientation angles:

![Angle Map](docs/images/angle_map_demo.png)

**For detailed instructions, see the [Complete Walkthrough](docs/WALKTHROUGH.md).**

## Examples

See the `examples/` directory for complete examples:
- `calibration_example.py` - Sunburst calibration demo
- `create_phantom.py` - Create synthetic calibration phantoms for testing

## License

MIT License

## Authors

- Mike Nelson (msnelson8@wisc.edu)
- Bin Li (bli346@wisc.edu)
- Jenu Chacko (jenu.chacko@wisc.edu)
