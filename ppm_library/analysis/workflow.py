"""
PPM Analysis Workflow.

Provides a high-level workflow function for analyzing PPM images using
calibration data and a greyscale mask to define regions of interest.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from scipy import ndimage
from skimage import io, color

import ppm_library
from ppm_library.calibration.radial import RadialCalibrator, RadialCalibrationResult
from ppm_library.imaging.ppm_image import PPMImage, AngleMap


@dataclass
class PPMAnalysisResult:
    """Result of PPM image analysis.

    Contains the angle map, mask, calibration data, statistics,
    histogram, and RGB visualization.

    Attributes:
        angle_map: Full angle map with valid_mask indicating analyzed pixels
        mask: Final binary mask used for analysis (bool array)
        calibration: The RadialCalibrationResult used for hue-to-angle conversion

        mean_angle: Mean fiber angle in the ROI (degrees, 0-180)
        std_angle: Standard deviation of angles in the ROI (degrees)
        n_valid_pixels: Number of valid pixels that were analyzed

        histogram_counts: Histogram bin counts for angle distribution
        histogram_bin_edges: Histogram bin edges in degrees

        angle_rgb: RGB colormap visualization of angles (H, W, 3), uint8
    """

    # Core data
    angle_map: AngleMap
    mask: np.ndarray
    calibration: RadialCalibrationResult

    # Statistics
    mean_angle: float
    std_angle: float
    n_valid_pixels: int

    # Histogram
    histogram_counts: np.ndarray
    histogram_bin_edges: np.ndarray

    # Visualization
    angle_rgb: np.ndarray

    def save(self, path: Union[str, Path]) -> None:
        """Save analysis results to an NPZ file.

        Args:
            path: Output file path (should end in .npz)
        """
        path = Path(path)
        np.savez(
            path,
            angles=self.angle_map.angles,
            angle_valid_mask=self.angle_map.valid_mask,
            mask=self.mask,
            mean_angle=np.array([self.mean_angle]),
            std_angle=np.array([self.std_angle]),
            n_valid_pixels=np.array([self.n_valid_pixels]),
            histogram_counts=self.histogram_counts,
            histogram_bin_edges=self.histogram_bin_edges,
            angle_rgb=self.angle_rgb,
            # Calibration data
            cal_slope=np.array([self.calibration.slope]),
            cal_intercept=np.array([self.calibration.intercept]),
            cal_inv_slope=np.array([self.calibration.inv_slope]),
            cal_inv_intercept=np.array([self.calibration.inv_intercept]),
            cal_r_squared=np.array([self.calibration.r_squared]),
            cal_hue_offset=np.array([self.calibration.hue_offset]),
            # Provenance
            ppm_library_version=np.array([ppm_library.__version__]),
        )

    def print_summary(self) -> None:
        """Print a summary of the analysis results."""
        print("PPM Analysis Results")
        print("=" * 40)
        print(f"Valid pixels analyzed: {self.n_valid_pixels:,}")
        print(f"Mean fiber angle: {self.mean_angle:.2f} deg")
        print(f"Std deviation: {self.std_angle:.2f} deg")
        print(f"Calibration R²: {self.calibration.r_squared:.4f}")
        print(f"Histogram bins: {len(self.histogram_counts)}")


def _load_greyscale_mask(
    path: Union[str, Path],
) -> np.ndarray:
    """Load an image and convert to greyscale if needed.

    Args:
        path: Path to the mask image

    Returns:
        2D numpy array of greyscale values (float, 0-255 range)
    """
    path = Path(path)
    img = io.imread(path)

    # Handle different image formats
    if img.ndim == 3:
        if img.shape[2] == 4:
            # RGBA - drop alpha
            img = img[:, :, :3]
        if img.shape[2] == 3:
            # RGB - convert to greyscale
            img = color.rgb2gray(img) * 255
    elif img.ndim == 2:
        # Already greyscale
        if img.dtype == np.float64 or img.dtype == np.float32:
            if img.max() <= 1.0:
                img = img * 255

    return img.astype(np.float64)


def analyze_ppm(
    calibration_input: Union[str, Path],
    ppm_image_path: Union[str, Path],
    mask_image_path: Union[str, Path],
    threshold: float,
    n_spokes: int = 16,
    histogram_bins: int = 18,
    saturation_threshold: float = 0.2,
    value_threshold: float = 0.2,
    colormap: str = "hsv",
) -> PPMAnalysisResult:
    """Analyze a PPM image using calibration and a greyscale mask.

    This function provides a complete workflow for PPM fiber orientation analysis:
    1. Load or compute calibration from a sunburst pattern
    2. Load the PPM sample image
    3. Create a binary mask from a greyscale image (with 3x3 median filter)
    4. Extract fiber angles in the masked region
    5. Compute statistics and generate visualization

    Args:
        calibration_input: Path to calibration image OR path to pre-saved
            calibration .npz file. If the path ends with ".npz", it will be
            loaded as a saved calibration. Otherwise, RadialCalibrator will
            be run on the image.
        ppm_image_path: Path to the PPM sample image to analyze.
        mask_image_path: Path to a greyscale image used to define the
            analysis region. Can be single-channel or RGB (will be converted).
        threshold: Threshold value for the greyscale mask. Pixels with values
            ABOVE this threshold will be included in the analysis.
        n_spokes: Number of spokes in the calibration pattern (default 16).
            Only used if calibration_input is an image, not a .npz file.
        histogram_bins: Number of bins for the angle histogram (default 18,
            giving 10 deg bins for the 0-180 deg range).
        saturation_threshold: Minimum HSV saturation for valid pixels (0-1).
            Pixels below this threshold are excluded from analysis.
        value_threshold: Minimum HSV value (brightness) for valid pixels (0-1).
            Pixels below this threshold are excluded from analysis.
        colormap: Matplotlib colormap name for angle visualization (default "hsv").

    Returns:
        PPMAnalysisResult containing the angle map, mask, statistics,
        histogram, and RGB visualization.

    Example:
        >>> from ppm_library import analyze_ppm
        >>>
        >>> # Analyze using a calibration image
        >>> result = analyze_ppm(
        ...     calibration_input="calibration_slide.tif",
        ...     ppm_image_path="sample.tif",
        ...     mask_image_path="sample_mask.tif",
        ...     threshold=128,  # Analyze bright regions
        ... )
        >>>
        >>> # Or use a pre-saved calibration
        >>> result = analyze_ppm(
        ...     calibration_input="calibration.npz",
        ...     ppm_image_path="sample.tif",
        ...     mask_image_path="sample_mask.tif",
        ...     threshold=128,
        ... )
        >>>
        >>> result.print_summary()
        >>> print(f"Mean angle: {result.mean_angle:.1f} deg")
    """
    calibration_input = Path(calibration_input)
    ppm_image_path = Path(ppm_image_path)
    mask_image_path = Path(mask_image_path)

    # Step 1: Load or run calibration
    if calibration_input.suffix.lower() == ".npz":
        calibration = RadialCalibrationResult.load(calibration_input)
    else:
        calibrator = RadialCalibrator(
            n_spokes=n_spokes,
            saturation_threshold=saturation_threshold,
            value_threshold=value_threshold,
        )
        calibration = calibrator.calibrate(str(calibration_input))

    # Step 2: Load PPM sample image
    ppm_image = PPMImage.load(
        str(ppm_image_path),
        saturation_threshold=saturation_threshold,
        value_threshold=value_threshold,
    )

    # Step 3: Create binary mask from greyscale image
    # Load and convert to greyscale
    mask_raw = _load_greyscale_mask(mask_image_path)

    # Verify dimensions match
    if mask_raw.shape[:2] != ppm_image.shape[:2]:
        raise ValueError(
            f"Mask dimensions {mask_raw.shape[:2]} do not match "
            f"PPM image dimensions {ppm_image.shape[:2]}"
        )

    # Apply 3x3 median filter
    mask_filtered = ndimage.median_filter(mask_raw, size=3)

    # Threshold: pixels ABOVE threshold are included
    binary_mask = mask_filtered > threshold

    # Combine with PPM image's valid mask (saturation/value thresholds)
    final_mask = binary_mask & ppm_image.valid_mask

    # Step 4: Apply calibration and extract angles
    angle_map = ppm_image.to_angle_map(calibration, roi=final_mask)

    # Step 5: Compute statistics
    valid_angles = angle_map.angles[final_mask & ~np.isnan(angle_map.angles)]
    n_valid_pixels = len(valid_angles)

    if n_valid_pixels > 0:
        mean_angle = float(np.mean(valid_angles))
        std_angle = float(np.std(valid_angles))
    else:
        mean_angle = np.nan
        std_angle = np.nan

    # Compute histogram
    histogram_counts, histogram_bin_edges = np.histogram(
        valid_angles,
        bins=histogram_bins,
        range=(0, 180),
    )

    # Step 6: Generate RGB visualization
    angle_rgb = angle_map.to_rgb_colormap(colormap)

    return PPMAnalysisResult(
        angle_map=angle_map,
        mask=final_mask,
        calibration=calibration,
        mean_angle=mean_angle,
        std_angle=std_angle,
        n_valid_pixels=n_valid_pixels,
        histogram_counts=histogram_counts,
        histogram_bin_edges=histogram_bin_edges,
        angle_rgb=angle_rgb,
    )
