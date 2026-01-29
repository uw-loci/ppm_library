"""
Sunburst Calibration Module.

This module provides tools for creating hue-to-angle linear regression models
from calibration slides containing sunburst/compass patterns with oriented rectangles.

The calibration process:
1. Load a calibration slide image (.tif or .ome.tif)
2. Segment rectangles from background (they appear as colored regions)
3. Detect the orientation angle of each rectangle
4. Extract the mode RGB/hue value for each rectangle
5. Create a linear regression mapping hue to angle (0-180 degrees)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List, Union
import warnings

import numpy as np
from scipy import ndimage, stats
from scipy.optimize import curve_fit
import cv2
from skimage import color, measure, morphology, filters
from skimage.io import imread
import matplotlib.pyplot as plt


@dataclass
class CalibrationRectangle:
    """Data for a single calibration rectangle."""

    label: int
    centroid: Tuple[float, float]  # (row, col)
    angle: float  # Orientation angle in degrees (0-180)
    area: int  # Number of pixels
    rgb_mode: Tuple[int, int, int]  # Mode RGB value
    hue_mode: float  # Mode hue value (0-1)
    hue_mean: float  # Mean hue value (0-1)
    mask: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class CalibrationResult:
    """Result of sunburst calibration."""

    # Regression coefficients: hue = slope * angle + intercept
    slope: float
    intercept: float
    r_squared: float

    # Inverse: angle = inv_slope * hue + inv_intercept
    inv_slope: float
    inv_intercept: float

    # Raw data
    angles: np.ndarray
    hue_values: np.ndarray
    rectangles: List[CalibrationRectangle]

    # Phase offset for alignment (from MATLAB: 61.875 degrees)
    phase_offset: float = 61.875

    def hue_to_angle(self, hue: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert hue value(s) to angle(s) in degrees.

        Args:
            hue: Hue value(s) in range [0, 1]

        Returns:
            Angle(s) in degrees [0, 180)
        """
        angle = self.inv_slope * hue + self.inv_intercept
        # Wrap to [0, 180)
        return np.mod(angle, 180.0)

    def angle_to_hue(self, angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert angle(s) to hue value(s).

        Args:
            angle: Angle(s) in degrees

        Returns:
            Hue value(s) in range [0, 1]
        """
        hue = self.slope * angle + self.intercept
        # Wrap to [0, 1]
        return np.mod(hue, 1.0)

    def save(self, path: Union[str, Path]) -> None:
        """Save calibration to NPZ file.

        Args:
            path: Output file path
        """
        np.savez(
            path,
            slope=self.slope,
            intercept=self.intercept,
            r_squared=self.r_squared,
            inv_slope=self.inv_slope,
            inv_intercept=self.inv_intercept,
            angles=self.angles,
            hue_values=self.hue_values,
            phase_offset=self.phase_offset,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CalibrationResult":
        """Load calibration from NPZ file.

        Args:
            path: Input file path

        Returns:
            CalibrationResult instance
        """
        data = np.load(path)
        return cls(
            slope=float(data["slope"]),
            intercept=float(data["intercept"]),
            r_squared=float(data["r_squared"]),
            inv_slope=float(data["inv_slope"]),
            inv_intercept=float(data["inv_intercept"]),
            angles=data["angles"],
            hue_values=data["hue_values"],
            rectangles=[],  # Not saved
            phase_offset=float(data.get("phase_offset", 61.875)),
        )


class SunburstCalibrator:
    """Calibrator for sunburst/compass calibration slides.

    This class segments oriented rectangles from a calibration slide image
    and creates a linear regression mapping hue values to orientation angles.

    The calibration slide should contain:
    - Background (dark or uniform color)
    - Vertical lines (ignored)
    - A set of rectangles oriented in different directions (like a compass/sunburst)

    Typical slides have 12-16 rectangles evenly spaced around 180 degrees.

    Example:
        >>> calibrator = SunburstCalibrator()
        >>> result = calibrator.calibrate("calibration_slide.tif")
        >>> print(f"R-squared: {result.r_squared:.4f}")
        >>> result.save("calibration.npz")

        # Later use:
        >>> result = CalibrationResult.load("calibration.npz")
        >>> angle = result.hue_to_angle(0.5)  # Convert hue 0.5 to angle
    """

    def __init__(
        self,
        n_expected_rectangles: int = 16,
        min_area: int = 100,
        saturation_threshold: float = 0.1,
        value_threshold: float = 0.1,
        merge_duplicates: bool = True,
        angle_tolerance: float = 5.0,
    ):
        """Initialize the calibrator.

        Args:
            n_expected_rectangles: Expected number of rectangles (default 16).
                For a full 360 deg sunburst, this would be the total count (e.g., 16),
                but since opposite directions are the same orientation (0-180 deg),
                you'll have n_expected_rectangles/2 unique angles.
            min_area: Minimum pixel area for a valid rectangle
            saturation_threshold: Minimum saturation to be considered colored (not background)
            value_threshold: Minimum value (brightness) to be considered foreground
            merge_duplicates: If True, merge rectangles with similar angles (within
                angle_tolerance). This handles sunburst patterns where opposite
                directions have the same orientation. Default True.
            angle_tolerance: Angle difference (degrees) within which rectangles
                are considered duplicates and merged. Default 5.0.
        """
        self.n_expected_rectangles = n_expected_rectangles
        self.min_area = min_area
        self.saturation_threshold = saturation_threshold
        self.value_threshold = value_threshold
        self.merge_duplicates = merge_duplicates
        self.angle_tolerance = angle_tolerance

    def calibrate(
        self,
        image_path: Union[str, Path],
        debug_plot: bool = False,
    ) -> CalibrationResult:
        """Perform calibration on a sunburst slide image.

        Args:
            image_path: Path to calibration slide image (.tif or .ome.tif)
            debug_plot: If True, show debug visualization

        Returns:
            CalibrationResult with regression model and data
        """
        # Load image
        image = self._load_image(image_path)

        # Segment rectangles
        rectangles = self._segment_rectangles(image)

        if len(rectangles) < 3:
            raise ValueError(
                f"Found only {len(rectangles)} rectangles. Need at least 3 for regression."
            )

        if len(rectangles) != self.n_expected_rectangles:
            warnings.warn(
                f"Found {len(rectangles)} rectangles, expected {self.n_expected_rectangles}. "
                "Calibration will proceed but results may be inaccurate."
            )

        # Merge duplicate angles if enabled (handles sunburst opposite directions)
        if self.merge_duplicates:
            rectangles = self._merge_duplicate_angles(rectangles)

        # Extract angles and hue values
        angles = np.array([r.angle for r in rectangles])
        hue_values = np.array([r.hue_mode for r in rectangles])

        # Fit linear regression
        result = self._fit_regression(angles, hue_values, rectangles)

        if debug_plot:
            self._plot_debug(image, rectangles, result)

        return result

    def _load_image(self, path: Union[str, Path]) -> np.ndarray:
        """Load image from file.

        Args:
            path: Path to image file

        Returns:
            RGB image as numpy array (H, W, 3), uint8
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        # Try tifffile first for OME-TIFF support
        try:
            import tifffile
            image = tifffile.imread(str(path))
        except Exception:
            # Fall back to skimage
            image = imread(str(path))

        # Handle multi-channel TIFF (channels might be first dimension)
        if image.ndim == 3:
            if image.shape[0] == 3 and image.shape[2] != 3:
                # Channels first: (3, H, W) -> (H, W, 3)
                image = np.moveaxis(image, 0, -1)
            elif image.shape[2] == 4:
                # RGBA -> RGB
                image = image[:, :, :3]
        elif image.ndim == 2:
            # Grayscale -> RGB
            image = np.stack([image] * 3, axis=-1)

        # Ensure uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        return image

    def _segment_rectangles(self, image: np.ndarray) -> List[CalibrationRectangle]:
        """Segment oriented rectangles from the calibration image.

        Args:
            image: RGB image (H, W, 3)

        Returns:
            List of CalibrationRectangle objects
        """
        # Convert to HSV
        hsv = color.rgb2hsv(image)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        # Create foreground mask: pixels with sufficient saturation and value
        # This removes background and dark regions
        foreground_mask = (saturation > self.saturation_threshold) & (value > self.value_threshold)

        # Clean up mask with morphological operations
        foreground_mask = morphology.remove_small_objects(foreground_mask, min_size=self.min_area)
        foreground_mask = morphology.remove_small_holes(foreground_mask, area_threshold=500)

        # Optional: Apply median filter to reduce noise
        foreground_mask = ndimage.median_filter(foreground_mask.astype(np.uint8), size=5).astype(bool)

        # Label connected components
        labels = measure.label(foreground_mask)
        regions = measure.regionprops(labels, intensity_image=hue)

        rectangles = []

        for region in regions:
            # Skip small regions
            if region.area < self.min_area:
                continue

            # Get the mask for this region
            region_mask = labels == region.label

            # Calculate orientation angle
            angle = self._calculate_rectangle_angle(region, region_mask)

            # Extract RGB values for this region
            rgb_pixels = image[region_mask]
            rgb_mode = self._calculate_mode_rgb(rgb_pixels)

            # Extract hue values
            hue_pixels = hue[region_mask]
            hue_mode = self._calculate_mode_hue(hue_pixels)
            hue_mean = np.mean(hue_pixels)

            rect = CalibrationRectangle(
                label=region.label,
                centroid=region.centroid,
                angle=angle,
                area=region.area,
                rgb_mode=rgb_mode,
                hue_mode=hue_mode,
                hue_mean=hue_mean,
                mask=region_mask,
            )
            rectangles.append(rect)

        # Sort by angle for consistent ordering
        rectangles.sort(key=lambda r: r.angle)

        return rectangles

    def _merge_duplicate_angles(
        self,
        rectangles: List[CalibrationRectangle],
    ) -> List[CalibrationRectangle]:
        """Merge rectangles with similar angles (e.g., opposite directions in sunburst).

        In a 360 deg sunburst pattern, rectangles pointing in opposite directions
        (e.g., up vs down) have the same orientation in the 0-180 deg range.
        This function merges such duplicates by averaging their hue values.

        Args:
            rectangles: List of detected rectangles, sorted by angle

        Returns:
            List of merged rectangles with unique angles
        """
        if not rectangles:
            return rectangles

        merged = []
        used = set()

        for i, rect in enumerate(rectangles):
            if i in used:
                continue

            # Find all rectangles with similar angles
            group = [rect]
            group_indices = [i]

            for j, other in enumerate(rectangles[i + 1:], start=i + 1):
                if j in used:
                    continue

                # Check if angles are within tolerance
                angle_diff = abs(rect.angle - other.angle)
                # Also check wrap-around (e.g., 179 deg and 1 deg are close)
                angle_diff = min(angle_diff, 180 - angle_diff)

                if angle_diff <= self.angle_tolerance:
                    group.append(other)
                    group_indices.append(j)

            # Mark all in group as used
            used.update(group_indices)

            if len(group) == 1:
                # No duplicates, keep as-is
                merged.append(rect)
            else:
                # Merge duplicates by averaging
                avg_angle = np.mean([r.angle for r in group])
                avg_hue_mode = np.mean([r.hue_mode for r in group])
                avg_hue_mean = np.mean([r.hue_mean for r in group])
                total_area = sum(r.area for r in group)

                # Use RGB mode from the rectangle with largest area
                largest = max(group, key=lambda r: r.area)

                merged_rect = CalibrationRectangle(
                    label=rect.label,  # Keep first label
                    centroid=rect.centroid,  # Keep first centroid
                    angle=avg_angle,
                    area=total_area,
                    rgb_mode=largest.rgb_mode,
                    hue_mode=avg_hue_mode,
                    hue_mean=avg_hue_mean,
                    mask=None,  # Don't merge masks
                )
                merged.append(merged_rect)

        # Re-sort by angle
        merged.sort(key=lambda r: r.angle)

        return merged

    def _calculate_rectangle_angle(
        self,
        region,  # skimage.measure._regionprops.RegionProperties
        mask: np.ndarray
    ) -> float:
        """Calculate the orientation angle of a rectangle region.

        Uses the region's orientation property (from moments) or
        minimum bounding rectangle as fallback.

        Args:
            region: Region properties from skimage
            mask: Binary mask for the region

        Returns:
            Angle in degrees [0, 180)
        """
        # Method 1: Use skimage orientation (from image moments)
        # This gives angle in radians from -pi/2 to pi/2
        orientation_rad = region.orientation

        # Convert to degrees and adjust to [0, 180) range
        # skimage orientation is measured counter-clockwise from horizontal
        angle = np.degrees(-orientation_rad)  # Negate for standard convention

        # Wrap to [0, 180)
        angle = np.mod(angle, 180.0)

        return angle

    def _calculate_mode_rgb(self, rgb_pixels: np.ndarray) -> Tuple[int, int, int]:
        """Calculate the mode (most common) RGB value.

        Args:
            rgb_pixels: Array of RGB values (N, 3)

        Returns:
            Tuple of (R, G, B) mode values
        """
        # Use histogram to find mode for each channel
        r_mode = int(stats.mode(rgb_pixels[:, 0], keepdims=False).mode)
        g_mode = int(stats.mode(rgb_pixels[:, 1], keepdims=False).mode)
        b_mode = int(stats.mode(rgb_pixels[:, 2], keepdims=False).mode)

        return (r_mode, g_mode, b_mode)

    def _calculate_mode_hue(self, hue_pixels: np.ndarray) -> float:
        """Calculate the mode hue value.

        Since hue is circular, we use histogram binning.

        Args:
            hue_pixels: Array of hue values (N,) in range [0, 1]

        Returns:
            Mode hue value in [0, 1]
        """
        # Bin hue values (256 bins to match standard image representation)
        hist, bin_edges = np.histogram(hue_pixels, bins=256, range=(0, 1))

        # Find bin with maximum count
        mode_bin = np.argmax(hist)

        # Return center of that bin
        mode_hue = (bin_edges[mode_bin] + bin_edges[mode_bin + 1]) / 2

        return float(mode_hue)

    def _fit_regression(
        self,
        angles: np.ndarray,
        hue_values: np.ndarray,
        rectangles: List[CalibrationRectangle],
    ) -> CalibrationResult:
        """Fit linear regression between angles and hue values.

        Since both angle and hue are circular, we need to handle wrap-around.
        The regression maps hue -> angle for the 0-180 degree range.

        Args:
            angles: Array of angles in degrees
            hue_values: Array of hue values [0, 1]
            rectangles: List of calibration rectangles

        Returns:
            CalibrationResult with regression parameters
        """
        # Scale hue to 0-180 to match angle range for fitting
        hue_scaled = hue_values * 180.0

        # Handle circular nature of data by checking for wrap-around
        # If there's a big jump, we might need to unwrap
        hue_sorted_idx = np.argsort(hue_scaled)
        hue_sorted = hue_scaled[hue_sorted_idx]
        angles_sorted = angles[hue_sorted_idx]

        # Check for wrap-around in angles (large jumps)
        angle_diffs = np.diff(angles_sorted)
        if np.any(np.abs(angle_diffs) > 90):
            # There's wrap-around; adjust angles to be continuous
            angles_unwrapped = angles_sorted.copy()
            for i in range(1, len(angles_unwrapped)):
                if angles_unwrapped[i] - angles_unwrapped[i-1] < -90:
                    angles_unwrapped[i:] += 180
                elif angles_unwrapped[i] - angles_unwrapped[i-1] > 90:
                    angles_unwrapped[i:] -= 180
            angles_sorted = angles_unwrapped

        # Fit forward regression: angle = m * hue_scaled + b
        coeffs = np.polyfit(hue_sorted, angles_sorted, 1)
        inv_slope = coeffs[0]  # This maps hue (scaled) to angle
        inv_intercept = coeffs[1]

        # Calculate R-squared
        predicted = np.polyval(coeffs, hue_sorted)
        ss_res = np.sum((angles_sorted - predicted) ** 2)
        ss_tot = np.sum((angles_sorted - np.mean(angles_sorted)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Fit inverse regression: hue_scaled = m * angle + b
        inv_coeffs = np.polyfit(angles_sorted, hue_sorted, 1)
        slope = inv_coeffs[0] / 180.0  # Scale back to hue [0,1] per degree
        intercept = inv_coeffs[1] / 180.0

        # Scale inv_slope/intercept for hue in [0,1]
        # angle = inv_slope * (hue * 180) + inv_intercept
        # angle = (inv_slope * 180) * hue + inv_intercept

        return CalibrationResult(
            slope=slope,
            intercept=intercept,
            r_squared=r_squared,
            inv_slope=inv_slope * 180.0,  # Now takes hue [0,1] directly
            inv_intercept=inv_intercept,
            angles=angles,
            hue_values=hue_values,
            rectangles=rectangles,
        )

    def _plot_debug(
        self,
        image: np.ndarray,
        rectangles: List[CalibrationRectangle],
        result: CalibrationResult,
    ) -> None:
        """Create debug visualization.

        Args:
            image: Original RGB image
            rectangles: Detected rectangles
            result: Calibration result
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Plot 1: Original image with detected rectangles
        ax1 = axes[0, 0]
        ax1.imshow(image)
        for rect in rectangles:
            cy, cx = rect.centroid
            ax1.plot(cx, cy, 'ro', markersize=8)
            ax1.annotate(
                f"{rect.angle:.1f}",
                (cx, cy),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7)
            )
        ax1.set_title("Detected Rectangles with Angles")
        ax1.axis('off')

        # Plot 2: Black and white segmentation mask
        ax2 = axes[0, 1]
        mask_combined = np.zeros(image.shape[:2], dtype=np.float32)
        for rect in rectangles:
            if rect.mask is not None:
                mask_combined[rect.mask] = 1.0
        ax2.imshow(mask_combined, cmap='gray', vmin=0, vmax=1)
        ax2.set_title("Segmentation Mask (white = detected)")
        ax2.axis('off')

        # Plot 3: Scatter plot of hue vs angle
        ax3 = axes[1, 0]
        ax3.scatter(result.hue_values, result.angles, s=100, c='blue', edgecolors='black')

        # Plot regression line
        hue_range = np.linspace(0, 1, 100)
        predicted_angles = result.hue_to_angle(hue_range)
        ax3.plot(hue_range, predicted_angles, 'r-', linewidth=2, label=f'Regression (R²={result.r_squared:.4f})')

        ax3.set_xlabel("Hue Value [0-1]")
        ax3.set_ylabel("Angle (degrees)")
        ax3.set_title("Hue to Angle Calibration")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 180)

        # Plot 4: Color wheel visualization
        ax4 = axes[1, 1]
        theta = np.linspace(0, np.pi, 100)  # 0-180 degrees
        r = np.ones_like(theta)

        # Create polar subplot
        ax4.remove()
        ax4 = fig.add_subplot(2, 2, 4, projection='polar')

        # Plot expected positions based on calibration
        for rect in rectangles:
            angle_rad = np.radians(rect.angle)
            # Color based on hue
            rgb = color.hsv2rgb([[[rect.hue_mode, 1.0, 1.0]]])[0, 0]
            ax4.scatter([angle_rad], [1], s=200, c=[rgb], edgecolors='black', linewidths=2)

        ax4.set_theta_zero_location('E')
        ax4.set_theta_direction(-1)
        ax4.set_thetamin(0)
        ax4.set_thetamax(180)
        ax4.set_title("Rectangle Positions (polar)")

        plt.tight_layout()
        plt.show()


def calibrate_from_image(
    image_path: Union[str, Path],
    n_rectangles: int = 16,
    debug: bool = False,
) -> CalibrationResult:
    """Convenience function to perform calibration.

    Args:
        image_path: Path to calibration slide image
        n_rectangles: Expected number of rectangles
        debug: Show debug plot

    Returns:
        CalibrationResult
    """
    calibrator = SunburstCalibrator(n_expected_rectangles=n_rectangles)
    return calibrator.calibrate(image_path, debug_plot=debug)
