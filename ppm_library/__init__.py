"""
PPM Library - Image Processing for Polarized Light Microscopy
===============================================================

A unified library for polarized light microscopy (PPM) that provides:

**Acquisition Support:**
- Hardware polarizer calibration (finding crossed polarizer positions)
- PPM rotation sensitivity testing and birefringence analysis
- Background/flatfield correction
- Bayer pattern debayering (CPU and GPU)
- TIFF writing with metadata

**Image Analysis:**
- Hue-to-angle calibration (extracting fiber angles from PPM images)
- PPM image loading and fiber angle extraction
- White balance correction and preprocessing
- Complete analysis workflows

This library has no dependencies on microscope hardware or control systems,
making it suitable for both real-time acquisition and offline analysis.

Example Usage - Acquisition Support:
-----------------------------------
from ppm_library import BackgroundCorrectionUtils, CPUDebayer

# Background correction
corrector = BackgroundCorrectionUtils()
corrected_image = corrector.apply_flatfield(raw_image, background_image)

# Debayering
debayer = CPUDebayer(pattern='RGGB')
rgb_image = debayer.debayer(bayer_image)

Example Usage - Image Analysis:
------------------------------
from ppm_library import RadialCalibrator, PPMImage, analyze_ppm

# Create calibration from sunburst slide
calibrator = RadialCalibrator(n_spokes=16)
calibration = calibrator.calibrate("sunburst_slide.tif")

# Analyze a PPM sample
result = analyze_ppm(
    calibration_input="calibration.npz",
    ppm_image_path="sample.tif",
    mask_image_path="mask.tif",
    threshold=128
)
print(f"Mean fiber angle: {result.mean_angle:.1f} degrees")
"""

__version__ = "1.3.0"
__author__ = "Mike Nelson, Bin Li, Jenu Chacko"

# =============================================================================
# Hardware/Acquisition Support (existing)
# =============================================================================
from ppm_library.ppm.polarizer_calibration import PolarizerCalibrationUtils
from ppm_library.imaging.background import BackgroundCorrectionUtils
from ppm_library.imaging.writer import TifWriterUtils
from ppm_library.debayering.cpu import CPUDebayer

# =============================================================================
# Hue-to-Angle Calibration (from PSTACS ppmlibrary)
# =============================================================================
from ppm_library.calibration import (
    RadialCalibrator,
    RadialCalibrationResult,
    HistogramCalibration,
    compute_hue_histogram,
)

# =============================================================================
# PPM Image Analysis (from PSTACS ppmlibrary)
# =============================================================================
from ppm_library.imaging import (
    PPMImage,
    AngleMap,
    load_ppm_image,
    hue_shift,
    compute_hue_shift_from_reference,
    preprocess_ppm_image,
)

# =============================================================================
# Analysis Workflows (from PSTACS ppmlibrary)
# =============================================================================
from ppm_library.analysis import analyze_ppm, PPMAnalysisResult

# =============================================================================
# Region-Based Analysis (for QuPath integration and standalone use)
# =============================================================================
from ppm_library.analysis import (
    analyze_region,
    compute_angles_from_rgb,
    compute_ppm_positive_mask,
    compute_masked_angles,
    compute_angle_histogram,
    compute_circular_statistics,
    filter_angles_by_range,
)

__all__ = [
    # Hardware/Acquisition Support
    "PolarizerCalibrationUtils",
    "BackgroundCorrectionUtils",
    "TifWriterUtils",
    "CPUDebayer",
    # Hue-to-Angle Calibration
    "RadialCalibrator",
    "RadialCalibrationResult",
    "HistogramCalibration",
    "compute_hue_histogram",
    # PPM Image Analysis
    "PPMImage",
    "AngleMap",
    "load_ppm_image",
    "hue_shift",
    "compute_hue_shift_from_reference",
    "preprocess_ppm_image",
    # File-Based Analysis Workflows
    "analyze_ppm",
    "PPMAnalysisResult",
    # Region-Based Analysis
    "analyze_region",
    "compute_angles_from_rgb",
    "compute_ppm_positive_mask",
    "compute_masked_angles",
    "compute_angle_histogram",
    "compute_circular_statistics",
    "filter_angles_by_range",
]
