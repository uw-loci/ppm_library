"""
Region-based PPM analysis functions.

Provides building-block functions for analyzing PPM image regions using
in-memory numpy arrays. These are designed to be called from any context
(QuPath via Appose/socket, standalone scripts, Jupyter notebooks) and
do not depend on file I/O or any GUI framework.

All functions accept numpy arrays and return numpy arrays or simple
Python types (dicts, floats).

Typical usage from QuPath integration:
    1. Java reads a tile from the sum image server -> RGB uint8 array
    2. Java reads corresponding tile from biref server -> uint16 array
    3. Call compute_masked_angle_stats(rgb, biref, calibration, ...)
    4. Java receives dict with angles, histogram, stats

Standalone usage:
    >>> import numpy as np
    >>> from ppm_library.analysis.region_analysis import compute_angles_from_rgb
    >>> from ppm_library.calibration.radial import RadialCalibrationResult
    >>>
    >>> calibration = RadialCalibrationResult.load("calibration.npz")
    >>> rgb = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    >>> result = compute_angles_from_rgb(rgb, calibration)
    >>> print(f"Valid pixels: {result['n_valid']}")
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from skimage import color

from ppm_library.calibration.radial import RadialCalibrationResult


def load_calibration(calibration_input):
    """Load a calibration from a path or return it if already loaded.

    Args:
        calibration_input: RadialCalibrationResult, or path to .npz file

    Returns:
        RadialCalibrationResult
    """
    if isinstance(calibration_input, RadialCalibrationResult):
        return calibration_input
    return RadialCalibrationResult.load(str(calibration_input))


def compute_angles_from_rgb(
    rgb_array,
    calibration,
    saturation_threshold=0.2,
    value_threshold=0.2,
):
    """Compute fiber orientation angles from an RGB image region.

    Converts RGB to HSV, extracts hue, and applies the calibration's
    linear regression to map hue values to fiber angles (0-180 degrees).

    Args:
        rgb_array: RGB image as numpy array (H, W, 3), uint8
        calibration: RadialCalibrationResult or path to .npz file
        saturation_threshold: minimum HSV saturation for valid pixels (0-1)
        value_threshold: minimum HSV value/brightness for valid pixels (0-1)

    Returns:
        dict with:
            'angles': float64 array (H, W), 0-180 degrees, NaN where invalid
            'valid_mask': bool array (H, W), True where pixel is measurable
            'hue': float64 array (H, W), raw hue values 0-1
            'saturation': float64 array (H, W), saturation 0-1
            'value': float64 array (H, W), value/brightness 0-1
            'n_valid': int, number of valid pixels
    """
    calibration = load_calibration(calibration)

    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError(f"Expected RGB array (H, W, 3), got shape {rgb_array.shape}")

    # Ensure uint8
    if rgb_array.dtype != np.uint8:
        if rgb_array.max() <= 1.0:
            rgb_array = (rgb_array * 255).astype(np.uint8)
        else:
            rgb_array = rgb_array.astype(np.uint8)

    # Convert to HSV (skimage returns float64 in range [0, 1])
    hsv = color.rgb2hsv(rgb_array)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Valid mask: sufficient color information for measurement
    valid_mask = (saturation >= saturation_threshold) & (value >= value_threshold)

    # Apply calibration: hue -> angle
    angles = np.full(hue.shape, np.nan, dtype=np.float64)
    if np.any(valid_mask):
        angles[valid_mask] = calibration.hue_to_angle(hue[valid_mask])

    return {
        'angles': angles,
        'valid_mask': valid_mask,
        'hue': hue,
        'saturation': saturation,
        'value': value,
        'n_valid': int(np.sum(valid_mask)),
    }


def compute_ppm_positive_mask(biref_array, threshold):
    """Create a PPM-positive mask from a birefringence image.

    Pixels with birefringence intensity above the threshold are considered
    PPM-positive (collagen-containing). Works with both uint16 single-channel
    and RGB birefringence images.

    Args:
        biref_array: Birefringence image. Either:
            - (H, W) single-channel uint16 or float
            - (H, W, 3) RGB (will use max across channels)
        threshold: Minimum value for PPM-positive classification.
            For uint16 images: 0-65535 range
            For float images: depends on data range

    Returns:
        bool array (H, W), True for PPM-positive pixels
    """
    if biref_array.ndim == 3:
        # Multi-channel: use max across channels
        biref_gray = np.max(biref_array, axis=2).astype(np.float64)
    elif biref_array.ndim == 2:
        biref_gray = biref_array.astype(np.float64)
    else:
        raise ValueError(f"Expected 2D or 3D array, got {biref_array.ndim}D")

    return biref_gray >= threshold


def compute_masked_angles(
    rgb_array,
    biref_array,
    calibration,
    biref_threshold,
    saturation_threshold=0.2,
    value_threshold=0.2,
):
    """Compute fiber angles masked to PPM-positive regions only.

    Combines angle computation from the sum/RGB image with birefringence
    thresholding to produce angles only in collagen-positive regions.

    Args:
        rgb_array: Sum image as RGB numpy array (H, W, 3), uint8
        biref_array: Birefringence image (H, W) or (H, W, 3)
        calibration: RadialCalibrationResult or path to .npz
        biref_threshold: Minimum biref value for PPM-positive
        saturation_threshold: Minimum HSV saturation for valid hue
        value_threshold: Minimum HSV value for valid hue

    Returns:
        dict with:
            'angles': float64 array (H, W), NaN where not PPM+/valid
            'combined_mask': bool array (H, W), True where PPM+ AND color-valid
            'ppm_positive_mask': bool array (H, W), True where biref > threshold
            'color_valid_mask': bool array (H, W), True where saturation/value OK
            'n_combined': int, pixels in combined mask
            'n_ppm_positive': int, pixels above biref threshold
            'n_color_valid': int, pixels with valid color
    """
    # Compute angles
    angle_result = compute_angles_from_rgb(
        rgb_array, calibration, saturation_threshold, value_threshold
    )

    # Compute PPM+ mask
    ppm_mask = compute_ppm_positive_mask(biref_array, biref_threshold)

    # Verify dimensions match
    if ppm_mask.shape != angle_result['valid_mask'].shape:
        raise ValueError(
            f"Dimension mismatch: RGB region {angle_result['valid_mask'].shape} "
            f"vs biref region {ppm_mask.shape}"
        )

    # Combined mask: must be both color-valid AND PPM-positive
    combined = angle_result['valid_mask'] & ppm_mask

    # Mask angles
    masked_angles = angle_result['angles'].copy()
    masked_angles[~combined] = np.nan

    return {
        'angles': masked_angles,
        'combined_mask': combined,
        'ppm_positive_mask': ppm_mask,
        'color_valid_mask': angle_result['valid_mask'],
        'n_combined': int(np.sum(combined)),
        'n_ppm_positive': int(np.sum(ppm_mask)),
        'n_color_valid': angle_result['n_valid'],
    }


def compute_angle_histogram(angles, mask=None, bins=18):
    """Compute histogram of fiber orientation angles.

    Args:
        angles: Angle array (H, W), may contain NaN
        mask: Optional bool mask (H, W). If None, uses all non-NaN pixels.
        bins: Number of histogram bins (default 18 = 10-degree bins)

    Returns:
        dict with:
            'counts': int array of length bins
            'bin_edges': float array of length bins+1 (0 to 180)
            'bin_centers': float array of length bins
            'n_pixels': total pixels counted
    """
    if mask is not None:
        valid_angles = angles[mask & ~np.isnan(angles)]
    else:
        valid_angles = angles[~np.isnan(angles)]

    counts, bin_edges = np.histogram(valid_angles, bins=bins, range=(0, 180))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    return {
        'counts': counts,
        'bin_edges': bin_edges,
        'bin_centers': bin_centers,
        'n_pixels': int(len(valid_angles)),
    }


def compute_circular_statistics(angles, mask=None):
    """Compute circular statistics for fiber orientation angles.

    Fiber angles are axial data (0-180 degrees, symmetric), so standard
    mean/std are misleading near the 0/180 boundary. This function uses
    circular statistics: angles are doubled to 0-360, circular mean/std
    computed, then halved back to 0-180.

    Args:
        angles: Angle array (H, W or 1D), may contain NaN
        mask: Optional bool mask. If None, uses all non-NaN pixels.

    Returns:
        dict with:
            'circular_mean': float, circular mean angle (0-180 degrees)
            'circular_std': float, circular standard deviation (degrees)
            'resultant_length': float, mean resultant length R (0-1),
                where 1 = perfectly aligned, 0 = uniform/random
            'n_pixels': int, number of valid pixels
            'arithmetic_mean': float, simple arithmetic mean (for comparison)
            'arithmetic_std': float, simple std (for comparison)
    """
    if mask is not None:
        valid = angles[mask & ~np.isnan(angles)]
    else:
        flat = angles.ravel() if angles.ndim > 1 else angles
        valid = flat[~np.isnan(flat)]

    n = len(valid)
    if n == 0:
        return {
            'circular_mean': np.nan,
            'circular_std': np.nan,
            'resultant_length': np.nan,
            'n_pixels': 0,
            'arithmetic_mean': np.nan,
            'arithmetic_std': np.nan,
        }

    # Double angles for axial data (0-180 -> 0-360)
    doubled_rad = np.deg2rad(valid * 2.0)

    # Circular mean
    sin_sum = np.sum(np.sin(doubled_rad))
    cos_sum = np.sum(np.cos(doubled_rad))
    mean_doubled_rad = np.arctan2(sin_sum / n, cos_sum / n)
    circular_mean = (np.rad2deg(mean_doubled_rad) / 2.0) % 180.0

    # Mean resultant length (measure of concentration)
    R = np.sqrt(sin_sum**2 + cos_sum**2) / n

    # Circular standard deviation
    if R < 1e-10:
        circular_std = 90.0  # Maximum dispersion for axial data
    else:
        circular_std = np.rad2deg(np.sqrt(-2.0 * np.log(R))) / 2.0

    return {
        'circular_mean': float(circular_mean),
        'circular_std': float(circular_std),
        'resultant_length': float(R),
        'n_pixels': n,
        'arithmetic_mean': float(np.mean(valid)),
        'arithmetic_std': float(np.std(valid)),
    }


def analyze_region(
    rgb_array,
    calibration,
    biref_array=None,
    biref_threshold=100,
    saturation_threshold=0.2,
    value_threshold=0.2,
    histogram_bins=18,
):
    """Complete region analysis: angles + optional PPM masking + stats + histogram.

    This is the primary entry point for analyzing a single region. It combines
    all the building blocks into a single call that returns everything needed
    for display.

    Args:
        rgb_array: Sum image region as RGB (H, W, 3), uint8
        calibration: RadialCalibrationResult or path to .npz
        biref_array: Optional birefringence image region (H, W) or (H, W, 3).
            If provided, analysis is restricted to PPM-positive pixels.
        biref_threshold: Threshold for PPM-positive classification (only used
            if biref_array is provided)
        saturation_threshold: Minimum HSV saturation for valid hue
        value_threshold: Minimum HSV value for valid hue
        histogram_bins: Number of angle histogram bins

    Returns:
        dict with:
            'angles': float64 array (H, W), fiber angles 0-180, NaN where invalid
            'mask': bool array (H, W), True where measurements are valid
            'histogram': dict from compute_angle_histogram()
            'stats': dict from compute_circular_statistics()
            'ppm_positive_mask': bool array or None (if no biref provided)
            'color_valid_mask': bool array (H, W)
    """
    calibration = load_calibration(calibration)

    if biref_array is not None:
        masked = compute_masked_angles(
            rgb_array, biref_array, calibration,
            biref_threshold, saturation_threshold, value_threshold,
        )
        angles = masked['angles']
        mask = masked['combined_mask']
        ppm_positive_mask = masked['ppm_positive_mask']
        color_valid_mask = masked['color_valid_mask']
    else:
        result = compute_angles_from_rgb(
            rgb_array, calibration, saturation_threshold, value_threshold,
        )
        angles = result['angles']
        mask = result['valid_mask']
        ppm_positive_mask = None
        color_valid_mask = result['valid_mask']

    histogram = compute_angle_histogram(angles, mask, bins=histogram_bins)
    stats = compute_circular_statistics(angles, mask)

    return {
        'angles': angles,
        'mask': mask,
        'histogram': histogram,
        'stats': stats,
        'ppm_positive_mask': ppm_positive_mask,
        'color_valid_mask': color_valid_mask,
    }


def filter_angles_by_range(angles, mask, angle_low, angle_high):
    """Create a binary mask of pixels within a fiber angle range.

    Used for the hue range filter overlay: highlights regions where
    fibers are oriented within the specified range.

    Args:
        angles: Angle array (H, W), 0-180 degrees, NaN where invalid
        mask: Valid pixel mask (H, W)
        angle_low: Lower bound of angle range (degrees, 0-180)
        angle_high: Upper bound of angle range (degrees, 0-180)

    Returns:
        dict with:
            'range_mask': bool array (H, W), True where angle is in range AND valid
            'n_in_range': int, number of pixels in range
            'n_valid': int, total valid pixels
            'fraction_in_range': float, fraction of valid pixels in range
    """
    if angle_low > angle_high:
        # Handle wrap-around (e.g., 170 to 10 degrees)
        in_range = (angles >= angle_low) | (angles <= angle_high)
    else:
        in_range = (angles >= angle_low) & (angles <= angle_high)

    range_mask = mask & in_range & ~np.isnan(angles)
    n_in_range = int(np.sum(range_mask))
    n_valid = int(np.sum(mask & ~np.isnan(angles)))

    return {
        'range_mask': range_mask,
        'n_in_range': n_in_range,
        'n_valid': n_valid,
        'fraction_in_range': n_in_range / n_valid if n_valid > 0 else 0.0,
    }
