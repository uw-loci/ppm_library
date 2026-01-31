"""
Calibration module for PPM Library.

Provides tools for creating hue-to-angle calibration from calibration slides.

- RadialCalibrator: Radial sampling from sunburst/fan pattern slides

Histogram correction:
- HistogramCalibration: Correct for optical anisotropy in hue histograms
- compute_hue_histogram: Compute hue histogram from RGB image

Note: This module handles HUE-TO-ANGLE calibration (extracting fiber angles
from PPM images). For HARDWARE polarizer calibration (finding crossed
polarizer positions), see ppm_library.ppm.polarizer_calibration.
"""

from ppm_library.calibration.radial import RadialCalibrator, RadialCalibrationResult
from ppm_library.calibration.histogram_correction import (
    HistogramCalibration,
    compute_hue_histogram,
)

__all__ = [
    "RadialCalibrator",
    "RadialCalibrationResult",
    "HistogramCalibration",
    "compute_hue_histogram",
]
