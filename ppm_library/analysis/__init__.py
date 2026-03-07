"""
Analysis module for PPM Library.

Provides high-level workflow functions for PPM image analysis.

Main functions:
- analyze_ppm: Complete file-based workflow for analyzing PPM images
- analyze_region: Region-based analysis (numpy arrays in/out, no file I/O)
- compute_angles_from_rgb: Angle extraction from RGB image data
- compute_ppm_positive_mask: Birefringence thresholding for collagen detection
- compute_masked_angles: Combined angle + PPM masking
- compute_circular_statistics: Circular mean/std for fiber angles
- filter_angles_by_range: Highlight pixels in a fiber angle range
- analyze_perpendicularity: Surface perpendicularity analysis (simple + PS-TACS)
"""

from ppm_library.analysis.workflow import analyze_ppm, PPMAnalysisResult
from ppm_library.analysis.region_analysis import (
    analyze_region,
    compute_angles_from_rgb,
    compute_ppm_positive_mask,
    compute_masked_angles,
    compute_angle_histogram,
    compute_circular_statistics,
    filter_angles_by_range,
)
from ppm_library.analysis.surface_analysis import (
    analyze_perpendicularity,
    rasterize_geojson_to_mask,
    compute_boundary_contour,
    compute_contour_normals,
    compute_border_zone_mask,
    compute_simple_perpendicularity,
    compute_tacs_scores,
)

__all__ = [
    "analyze_ppm",
    "PPMAnalysisResult",
    "analyze_region",
    "compute_angles_from_rgb",
    "compute_ppm_positive_mask",
    "compute_masked_angles",
    "compute_angle_histogram",
    "compute_circular_statistics",
    "filter_angles_by_range",
    "analyze_perpendicularity",
    "rasterize_geojson_to_mask",
    "compute_boundary_contour",
    "compute_contour_normals",
    "compute_border_zone_mask",
    "compute_simple_perpendicularity",
    "compute_tacs_scores",
]
