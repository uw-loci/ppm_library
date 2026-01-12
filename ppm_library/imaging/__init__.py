"""
Imaging package - General image processing utilities.

This package contains image processing functionality that is
NOT specific to any particular modality (PPM, brightfield, etc).

Modules:
    background: Background correction utilities (BackgroundCorrectionUtils)
    writer: TIFF writing utilities (TifWriterUtils)
    jai_calibration: JAI camera white balance calibration (JAIWhiteBalanceCalibrator)
"""

from ppm_library.imaging.writer import TifWriterUtils
from ppm_library.imaging.background import BackgroundCorrectionUtils

__all__ = ["TifWriterUtils", "BackgroundCorrectionUtils"]

# Optional: JAI calibration (may not be needed on all systems)
try:
    from ppm_library.imaging.jai_calibration import (
        JAIWhiteBalanceCalibrator,
        WhiteBalanceResult,
        CalibrationConfig,
    )
    __all__.extend(["JAIWhiteBalanceCalibrator", "WhiteBalanceResult", "CalibrationConfig"])
except ImportError:
    pass
