"""
Hue correction utilities for PPM images.

Provides functions for correcting hue values to account for white balance
differences and other systematic shifts between images.
"""

from typing import Optional, Tuple

import numpy as np
from scipy import ndimage
from skimage import color


def hue_shift(
    image: np.ndarray,
    angle_degrees: float,
    apply_median_filter: bool = True,
    filter_size: int = 3,
) -> np.ndarray:
    """Shift hue values by an angle offset for white balance correction.

    This function corrects for white balance discrepancies between images
    by shifting all hue values by a fixed angle. This is useful when the
    same fiber orientation appears as different colors in different images
    due to lighting or camera differences.

    Args:
        image: RGB image as numpy array (H, W, 3), uint8
        angle_degrees: Angle to shift hue by in degrees.
            Positive values shift toward higher hue (red->yellow->green).
            Negative values shift toward lower hue (green->yellow->red).
            Range is typically -180 to 180.
        apply_median_filter: Whether to apply median filter after shift
            to reduce noise artifacts (default True, matching MATLAB behavior)
        filter_size: Size of median filter kernel (default 3)

    Returns:
        Corrected RGB image as uint8 array (H, W, 3)

    Example:
        >>> from ppm_library.imaging import hue_shift
        >>> # Shift hue by 15 degrees to correct for white balance
        >>> corrected = hue_shift(image, 15.0)
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H, W, 3), got shape {image.shape}")

    # Convert to float if needed
    if image.dtype == np.uint8:
        image_float = image.astype(np.float64) / 255.0
    else:
        image_float = image.astype(np.float64)
        if image_float.max() > 1.0:
            image_float = image_float / 255.0

    # Convert to HSV
    hsv = color.rgb2hsv(image_float)

    # In PPM, hue spans 0-1 representing 0-180 degrees of fiber orientation
    # (fibers are symmetric, so 0 deg and 180 deg are equivalent)
    # Shift: angle_degrees / 180.0 in hue units
    hue_shift_amount = angle_degrees / 180.0

    # Apply shift with circular wrapping
    hsv[:, :, 0] = np.mod(hsv[:, :, 0] + hue_shift_amount, 1.0)

    # Apply median filter to hue channel if requested (reduces noise)
    if apply_median_filter:
        # Convert to 0-255 for filtering, then back
        hue_255 = (hsv[:, :, 0] * 255).astype(np.uint8)
        hue_255 = ndimage.median_filter(hue_255, size=filter_size)
        hsv[:, :, 0] = hue_255.astype(np.float64) / 255.0

    # Convert back to RGB
    rgb = color.hsv2rgb(hsv)

    return (rgb * 255).astype(np.uint8)


def compute_hue_shift_from_reference(
    image: np.ndarray,
    reference_angle: float,
    roi_mask: Optional[np.ndarray] = None,
    saturation_threshold: float = 0.2,
    value_threshold: float = 0.2,
) -> Tuple[float, float]:
    """Compute the hue shift needed to align an image with a reference angle.

    Given a region where fibers are known to be at a specific angle, this
    computes the hue shift needed to correct the image so that those fibers
    map to the correct hue value.

    This is useful for automatic white balance correction when you have
    a known reference (e.g., a fiber at a specific angle in the image).

    Args:
        image: RGB image as numpy array (H, W, 3), uint8
        reference_angle: The known fiber angle in degrees (0-180) at the ROI
        roi_mask: Boolean mask indicating pixels to use for reference.
            If None, uses all pixels with sufficient saturation/value.
        saturation_threshold: Minimum saturation for valid pixels (0-1)
        value_threshold: Minimum value/brightness for valid pixels (0-1)

    Returns:
        Tuple of (hue_shift_degrees, mean_hue_in_roi):
            - hue_shift_degrees: The angle shift to apply to correct the image
            - mean_hue_in_roi: The measured mean hue (0-1) in the reference region

    Example:
        >>> # I know fibers at this ROI are at 45 degrees
        >>> shift, measured_hue = compute_hue_shift_from_reference(image, 45.0, roi_mask)
        >>> corrected = hue_shift(image, shift)
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H, W, 3), got shape {image.shape}")

    # Convert to HSV
    if image.dtype == np.uint8:
        image_float = image.astype(np.float64) / 255.0
    else:
        image_float = image.astype(np.float64)

    hsv = color.rgb2hsv(image_float)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Create valid pixel mask
    valid = (saturation > saturation_threshold) & (value > value_threshold)

    if roi_mask is not None:
        valid = valid & roi_mask

    if not np.any(valid):
        raise ValueError("No valid pixels found in the specified region")

    # Compute mean hue in the reference region
    # Note: This is a simple mean, which may not be ideal for circular data
    # near the 0/1 boundary. For now, assume reference regions are away from wrap.
    mean_hue = np.mean(hue[valid])

    # Expected hue for the reference angle
    # In PPM, hue 0-1 maps to angle 0-180 degrees
    expected_hue = reference_angle / 180.0

    # Compute shift needed (in hue units, then convert to degrees)
    hue_diff = expected_hue - mean_hue

    # Handle wrap-around: choose the smaller shift direction
    if hue_diff > 0.5:
        hue_diff -= 1.0
    elif hue_diff < -0.5:
        hue_diff += 1.0

    hue_shift_degrees = hue_diff * 180.0

    return hue_shift_degrees, mean_hue


def apply_gaussian_smoothing(
    image: np.ndarray,
    sigma: float = 2.0,
) -> np.ndarray:
    """Apply Gaussian smoothing to an image.

    This is a preprocessing step often applied before hue extraction
    to reduce noise.

    Args:
        image: RGB image as numpy array (H, W, 3), uint8
        sigma: Standard deviation for Gaussian kernel (default 2.0)

    Returns:
        Smoothed RGB image as uint8 array
    """
    from skimage import filters

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H, W, 3), got shape {image.shape}")

    # Apply Gaussian filter to each channel
    smoothed = np.zeros_like(image, dtype=np.float64)
    for c in range(3):
        smoothed[:, :, c] = filters.gaussian(image[:, :, c].astype(np.float64), sigma=sigma)

    # Convert back to uint8
    if image.dtype == np.uint8:
        return (smoothed * 255).astype(np.uint8)
    return smoothed.astype(image.dtype)


def apply_median_filter(
    image: np.ndarray,
    size: int = 5,
) -> np.ndarray:
    """Apply median filter to each channel of an image.

    This is a preprocessing step to remove salt-and-pepper noise
    while preserving edges.

    Args:
        image: RGB image as numpy array (H, W, 3), uint8
        size: Size of the median filter kernel (default 5)

    Returns:
        Filtered RGB image as uint8 array
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H, W, 3), got shape {image.shape}")

    filtered = np.zeros_like(image)
    for c in range(3):
        filtered[:, :, c] = ndimage.median_filter(image[:, :, c], size=size)

    return filtered


def preprocess_ppm_image(
    image: np.ndarray,
    gaussian_sigma: float = 2.0,
    median_size: int = 5,
) -> np.ndarray:
    """Apply standard preprocessing to a PPM image.

    Applies Gaussian smoothing followed by median filtering to each channel.
    This matches the preprocessing in the MATLAB PIKLfun.m.

    Args:
        image: RGB image as numpy array (H, W, 3), uint8
        gaussian_sigma: Sigma for Gaussian smoothing (default 2.0)
        median_size: Size of median filter kernel (default 5)

    Returns:
        Preprocessed RGB image as uint8 array
    """
    # Apply Gaussian smoothing first
    smoothed = apply_gaussian_smoothing(image, sigma=gaussian_sigma)

    # Then apply median filter
    filtered = apply_median_filter(smoothed, size=median_size)

    return filtered
