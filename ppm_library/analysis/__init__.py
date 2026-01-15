"""
Analysis module for PPM Library.

Provides high-level workflow functions for PPM image analysis.

Main functions:
- analyze_ppm: Complete workflow for analyzing PPM images with calibration and masking
"""

from ppm_library.analysis.workflow import analyze_ppm, PPMAnalysisResult

__all__ = ["analyze_ppm", "PPMAnalysisResult"]
