"""
Imaging package - General image processing utilities.

This package contains image processing functionality including:
- Background/flatfield correction
- TIFF writing with metadata
- PPM image loading and angle extraction
- Hue correction and preprocessing

Modules:
    background: Background correction utilities (BackgroundCorrectionUtils)
    writer: TIFF writing utilities (TifWriterUtils)
    ppm_image: PPM image loading and angle extraction (PPMImage, AngleMap)
    hue_correction: Hue correction and preprocessing utilities

Note:
    JAI camera calibration has been moved to the microscope_control.jai package.
    Import from there: from microscope_control.jai import JAIWhiteBalanceCalibrator, JAICameraProperties
"""

from ppm_library.imaging.writer import TifWriterUtils
from ppm_library.imaging.background import BackgroundCorrectionUtils
from ppm_library.imaging.ppm_image import PPMImage, AngleMap, load_ppm_image
from ppm_library.imaging.hue_correction import (
    hue_shift,
    compute_hue_shift_from_reference,
    apply_gaussian_smoothing,
    apply_median_filter,
    preprocess_ppm_image,
)

__all__ = [
    # Existing
    "TifWriterUtils",
    "BackgroundCorrectionUtils",
    # PPM image processing
    "PPMImage",
    "AngleMap",
    "load_ppm_image",
    # Hue correction
    "hue_shift",
    "compute_hue_shift_from_reference",
    "apply_gaussian_smoothing",
    "apply_median_filter",
    "preprocess_ppm_image",
]
