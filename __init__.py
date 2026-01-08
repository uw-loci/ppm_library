"""
PPM Library - Image Processing for Polarized Light Microscopy
===============================================================

A standalone image processing library for polarized light microscopy (PPM)
and general microscopy imaging. Provides tools for:

- PPM calibration and birefringence analysis
- Background/flatfield correction
- Tissue detection
- Bayer pattern debayering (CPU and GPU)
- Camera calibration
- TIFF writing with metadata

This library has no dependencies on microscope hardware or control systems,
making it suitable for both real-time acquisition and offline analysis.

Example Usage:
-------------
from ppm_library.ppm.calibration import PolarizerCalibrationUtils
from ppm_library.imaging.background import BackgroundCorrectionUtils
from ppm_library.debayering.cpu import CPUDebayer

# Background correction
corrector = BackgroundCorrectionUtils()
corrected_image = corrector.apply_flatfield(raw_image, background_image)

# Debayering
debayer = CPUDebayer(pattern='RGGB')
rgb_image = debayer.debayer(bayer_image)
"""

__version__ = "1.0.0"
__author__ = "Mike Nelson, Bin Li, Jenu Chacko"

# Make key classes easily accessible
from ppm_library.ppm.calibration import PolarizerCalibrationUtils
from ppm_library.imaging.background import BackgroundCorrectionUtils
from microscope_control.autofocus.tissue_detection import EmptyRegionDetector
from ppm_library.imaging.writer import TifWriterUtils
from ppm_library.debayering.cpu import CPUDebayer

__all__ = [
    "PolarizerCalibrationUtils",
    "BackgroundCorrectionUtils",
    "EmptyRegionDetector",
    "TifWriterUtils",
    "CPUDebayer",
]
