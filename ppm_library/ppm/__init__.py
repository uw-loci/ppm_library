"""
PPM package - Polarized light microscopy (PPM) specific tools.

This package contains all PPM-specific functionality including
hardware polarizer calibration, rotation sensitivity analysis, and
birefringence processing.

Note: This module handles HARDWARE polarizer calibration (finding crossed
polarizer positions). For HUE-TO-ANGLE calibration (extracting fiber angles
from PPM images), see ppm_library.calibration.

Modules:
    polarizer_calibration: Hardware polarizer calibration (PolarizerCalibrationUtils)
    sensitivity_test: PPM rotation sensitivity testing (PPMRotationSensitivityTester)
    sensitivity_analysis: Rotation sensitivity analysis (PPMRotationAnalyzer)
    birefringence_test: Birefringence optimization (PPMBirefringenceMaximizationTester)
"""

from ppm_library.ppm.polarizer_calibration import PolarizerCalibrationUtils

__all__ = ["PolarizerCalibrationUtils"]

# Optional imports for PPM testing tools (may have additional dependencies)
try:
    from ppm_library.ppm.sensitivity_test import PPMRotationSensitivityTester

    __all__.append("PPMRotationSensitivityTester")
except ImportError:
    pass

try:
    from ppm_library.ppm.sensitivity_analysis import PPMRotationAnalyzer

    __all__.append("PPMRotationAnalyzer")
except ImportError:
    pass

try:
    from ppm_library.ppm.birefringence_test import PPMBirefringenceMaximizationTester

    __all__.append("PPMBirefringenceMaximizationTester")
except ImportError:
    pass
