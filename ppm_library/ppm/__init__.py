"""
PPM package - Polarized light microscopy (PPM) specific tools.

This package contains all PPM-specific functionality including
polarizer calibration, rotation sensitivity analysis, and
birefringence processing.

Modules:
    calibration: Polarizer calibration utilities (PolarizerCalibrationUtils)
    sensitivity_test: PPM rotation sensitivity testing (PPMRotationSensitivityTester)
    sensitivity_analysis: Rotation sensitivity analysis (PPMRotationAnalyzer)
    birefringence_test: Birefringence optimization (PPMBirefringenceMaximizationTester)
"""

from ppm_library.ppm.calibration import PolarizerCalibrationUtils

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
