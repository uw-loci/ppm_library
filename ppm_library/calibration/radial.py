"""
Radial Calibration Module.

This module provides tools for creating hue-to-angle linear regression models
using radial sampling from calibration slides with sunburst/fan patterns.

This approach samples hue values along radial lines (spokes) from the center
of the pattern, fitting a linear regression between hue and orientation angle.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union

import numpy as np
from scipy import ndimage, stats
from skimage import color, morphology, measure
from skimage.io import imread
import matplotlib.colors as mcolors

# ROYGBIV color positions in hue space (0-1)
# Red=0, Orange≈0.08, Yellow≈0.17, Green≈0.33, Blue≈0.58, Indigo≈0.67, Violet≈0.75, Red=1.0
ROYGBIV_COLORS = [
    (0.000, "R", "red"),
    (0.083, "O", "orange"),
    (0.167, "Y", "yellow"),
    (0.333, "G", "green"),
    (0.500, "C", "cyan"),
    (0.583, "B", "blue"),
    (0.667, "I", "indigo"),
    (0.750, "V", "violet"),
    (0.917, "M", "magenta"),
]


def add_hue_colorbar(ax, orientation="horizontal"):
    """Add a rainbow colorbar to indicate hue values.

    Args:
        ax: Matplotlib axes
        orientation: 'horizontal' or 'vertical'
    """
    # Create a gradient of hue values
    if orientation == "horizontal":
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
    else:
        gradient = np.linspace(0, 1, 256).reshape(-1, 1)

    # Convert hue to RGB
    hsv = np.zeros((*gradient.shape, 3))
    hsv[:, :, 0] = gradient
    hsv[:, :, 1] = 1.0  # Full saturation
    hsv[:, :, 2] = 1.0  # Full value
    rgb = mcolors.hsv_to_rgb(hsv)

    return rgb


def add_roygbiv_labels(ax, y_position=-0.12):
    """Add ROYGBIV letter labels below the x-axis.

    Args:
        ax: Matplotlib axes (assumes x-axis is hue 0-1)
        y_position: Relative y position for labels (in axes coordinates)
    """
    for hue, letter, color_name in ROYGBIV_COLORS:
        ax.annotate(
            letter,
            xy=(hue, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -25),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color=color_name,
        )


def create_hue_axis_colorbar(ax):
    """Add a thin rainbow bar below the x-axis to show hue colors.

    Args:
        ax: Matplotlib axes (assumes x-axis is hue 0-1)
    """
    # Get axis position
    pos = ax.get_position()

    # Create a new axes for the colorbar below the plot
    cbar_ax = ax.figure.add_axes([pos.x0, pos.y0 - 0.05, pos.width, 0.02])

    # Create hue gradient
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    hsv = np.zeros((1, 256, 3))
    hsv[0, :, 0] = gradient
    hsv[0, :, 1] = 1.0
    hsv[0, :, 2] = 1.0
    rgb = mcolors.hsv_to_rgb(hsv)

    cbar_ax.imshow(rgb, aspect="auto", extent=[0, 1, 0, 1])
    cbar_ax.set_xlim(0, 1)
    cbar_ax.set_xticks([])
    cbar_ax.set_yticks([])

    # Add ROYGBIV labels
    for hue, letter, color_name in ROYGBIV_COLORS:
        cbar_ax.text(
            hue, -0.5, letter, ha="center", va="top", fontsize=9, fontweight="bold", color="black"
        )

    return cbar_ax


def add_shifted_hue_colorbar(ax, hue_offset: float) -> None:
    """Add a rainbow colorbar shifted by hue_offset below the given axes.

    The colorbar shows the full hue spectrum shifted so that the x-axis
    positions correspond to the shifted hue values used in calibration
    scatter plots.

    Args:
        ax: Matplotlib axes
        hue_offset: Hue offset used when shifting hue values
    """
    pos = ax.get_position()

    # Create a new axes for the colorbar below the plot
    cbar_ax = ax.figure.add_axes([pos.x0, pos.y0 - 0.06, pos.width, 0.02])

    # Create SHIFTED hue gradient (matching the X axis which shows shifted values)
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    hsv = np.zeros((1, 256, 3))
    # Shift the hue values back by offset to show actual colors at shifted positions
    hsv[0, :, 0] = (gradient + hue_offset) % 1.0
    hsv[0, :, 1] = 1.0
    hsv[0, :, 2] = 1.0
    rgb = mcolors.hsv_to_rgb(hsv)

    cbar_ax.imshow(rgb, aspect="auto", extent=[0, 1, 0, 1])
    cbar_ax.set_xlim(0, 1)
    cbar_ax.set_xticks([])
    cbar_ax.set_yticks([])

    # Add shifted ROYGBIV labels
    for hue, letter, color_name in ROYGBIV_COLORS:
        shifted_pos = (hue - hue_offset) % 1.0
        cbar_ax.text(
            shifted_pos,
            -0.5,
            letter,
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            color="black",
        )


def _circular_hue_mean(hue_values: np.ndarray) -> float:
    """Compute circular mean for hue values (0-1 range, wraps at 0/1).

    Hue is circular: 0.99 and 0.01 are close together (both near red).
    Regular np.mean gives incorrect results near the wrapping boundary.
    """
    theta = np.asarray(hue_values) * 2.0 * np.pi
    mean_angle = np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta)))
    if mean_angle < 0:
        mean_angle += 2.0 * np.pi
    return float(mean_angle / (2.0 * np.pi))


def _circular_hue_std(hue_values: np.ndarray) -> float:
    """Compute circular standard deviation for hue values (0-1 range).

    Returns a value in [0, ~0.37] where 0 = perfectly consistent and
    higher values indicate more dispersion. Completely uniform distribution
    gives ~0.37. Values above 0.15 typically indicate non-spoke pixels.
    """
    theta = np.asarray(hue_values) * 2.0 * np.pi
    sin_mean = np.mean(np.sin(theta))
    cos_mean = np.mean(np.cos(theta))
    R = np.sqrt(sin_mean**2 + cos_mean**2)
    R = min(R, 1.0)  # numerical safety
    if R < 1e-10:
        return 1.0  # completely dispersed
    return float(np.sqrt(-2.0 * np.log(R)) / (2.0 * np.pi))


@dataclass
class RadialSample:
    """Data for a single radial sample."""

    angle: float  # Angle in degrees (0-180)
    hue_mean: float  # Circular mean hue value along the radial line
    hue_std: float  # Circular standard deviation of hue
    n_samples: int  # Number of pixels sampled


@dataclass
class RadialCalibrationResult:
    """Result of radial calibration."""

    # Regression coefficients: angle = inv_slope * hue_shifted + inv_intercept
    slope: float  # hue_shifted = slope * angle + intercept
    intercept: float
    inv_slope: float  # angle = inv_slope * hue_shifted + inv_intercept
    inv_intercept: float
    r_squared: float

    # Hue offset for unwrapping (hue_shifted = (hue_raw - hue_offset) % 1.0)
    hue_offset: float

    # Raw data (hue_values are shifted)
    angles: np.ndarray
    hue_values: np.ndarray  # Shifted hue values
    samples: List[RadialSample]

    # Center detection
    center: Tuple[int, int]  # (y, x) pixel coordinates

    # Diagnostics
    warnings: List[str] = None  # List of warning messages
    rotation: float = 0.0  # Optimal rotation offset found

    def hue_to_angle(self, hue: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert raw hue value(s) to angle(s) in degrees.

        Args:
            hue: Raw hue value(s) in range [0, 1]

        Returns:
            Angle(s) in degrees [0, 180)
        """
        # Apply hue offset to unwrap
        hue_shifted = (np.asarray(hue) - self.hue_offset) % 1.0
        angle = self.inv_slope * hue_shifted + self.inv_intercept
        return np.mod(angle, 180.0)

    def angle_to_hue(self, angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert angle(s) to raw hue value(s).

        Args:
            angle: Angle(s) in degrees

        Returns:
            Raw hue value(s) in range [0, 1]
        """
        # Compute shifted hue, then reverse the offset
        hue_shifted = self.slope * np.asarray(angle) + self.intercept
        hue_raw = (hue_shifted + self.hue_offset) % 1.0
        return hue_raw

    def save(self, path: Union[str, Path]) -> None:
        """Save calibration to NPZ file."""
        np.savez(
            path,
            slope=self.slope,
            intercept=self.intercept,
            inv_slope=self.inv_slope,
            inv_intercept=self.inv_intercept,
            r_squared=self.r_squared,
            hue_offset=self.hue_offset,
            angles=self.angles,
            hue_values=self.hue_values,
            center=np.array(self.center),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RadialCalibrationResult":
        """Load calibration from NPZ file."""
        data = np.load(path)
        return cls(
            slope=float(data["slope"]),
            intercept=float(data["intercept"]),
            inv_slope=float(data["inv_slope"]),
            inv_intercept=float(data["inv_intercept"]),
            r_squared=float(data["r_squared"]),
            hue_offset=float(data["hue_offset"]),
            angles=data["angles"],
            hue_values=data["hue_values"],
            samples=[],
            center=tuple(data["center"]),
        )

    def check_quality(
        self,
        expected_spokes: int,
        min_r_squared: float = 0.95,
    ) -> List[str]:
        """Check calibration quality and return warning strings.

        The caller decides what to do with warnings (log them, display them,
        abort, etc.). This method only identifies issues.

        Args:
            expected_spokes: Expected number of spokes in the sunburst pattern
            min_r_squared: Minimum acceptable R-squared value (default 0.95)

        Returns:
            List of warning strings (empty if quality is acceptable)
        """
        warnings = []
        spokes_detected = len(self.samples)

        if spokes_detected < expected_spokes:
            warnings.append(
                f"Expected {expected_spokes} spokes but found {spokes_detected}. "
                "Consider repositioning slide or adjusting detection thresholds."
            )

        if self.r_squared < min_r_squared:
            warnings.append(
                f"R-squared ({self.r_squared:.4f}) is below {min_r_squared}. "
                "Calibration may be inaccurate. Check for slide positioning "
                "or detection issues."
            )

        return warnings

    def save_plot(
        self,
        output_path: Union[str, Path, None],
        image: np.ndarray,
        calibrator: "RadialCalibrator",
        extra_info: Optional[Dict[str, str]] = None,
        dpi: int = 150,
    ) -> None:
        """Create and save a calibration visualization plot.

        Creates a 2x2 plot showing:
        - Original image with center crosshair and radial sampling lines
        - Foreground mask (B&W threshold visualization)
        - Color-coded scatter plot with regression line and hue colorbar
        - Calibration info text with optional extra metadata

        When output_path is None, displays the plot interactively using
        plt.show(). When a path is given, saves to file using the Agg
        backend.

        Args:
            output_path: Path to save the plot, or None for interactive display
            image: Original calibration image (RGB, uint8 or float)
            calibrator: RadialCalibrator instance (used for threshold and
                radius parameters)
            extra_info: Optional dict of display-only metadata to show in
                the info panel (e.g. {"Exposure R": "50 ms"})
            dpi: DPI for saved images (default 150, ignored for interactive)
        """
        import matplotlib

        if output_path is not None:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt_local
        import matplotlib.colors as mcolors_local
        from skimage import color as skcolor

        fig, axes = plt_local.subplots(2, 2, figsize=(14, 14))

        # Normalize image to uint8 for display and HSV conversion
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                img_uint8 = (image * 255).astype(np.uint8)
            else:
                img_uint8 = image.astype(np.uint8)
        else:
            img_uint8 = image

        # --- Plot 1: Original image with center crosshair and sampling lines ---
        ax1 = axes[0, 0]
        ax1.imshow(img_uint8)
        cy, cx = self.center
        ax1.plot(cx, cy, "w+", markersize=20, markeredgewidth=3)
        for sample in self.samples:
            angle_rad = np.radians(sample.angle)
            x_inner = cx + calibrator.radius_inner * np.cos(angle_rad)
            y_inner = cy - calibrator.radius_inner * np.sin(angle_rad)
            x_outer = cx + calibrator.radius_outer * np.cos(angle_rad)
            y_outer = cy - calibrator.radius_outer * np.sin(angle_rad)
            ax1.plot([x_inner, x_outer], [y_inner, y_outer], "w-", alpha=0.5, linewidth=1)
        rot_text = f", rot={self.rotation:.1f} deg" if self.rotation != 0 else ""
        ax1.set_title(f"Radial Sampling ({len(self.samples)} spokes{rot_text})")
        ax1.axis("off")

        # --- Plot 2: Foreground mask ---
        ax2 = axes[0, 1]
        hsv_img = skcolor.rgb2hsv(img_uint8)
        foreground_mask = (hsv_img[:, :, 1] > calibrator.saturation_threshold) & (
            hsv_img[:, :, 2] > calibrator.value_threshold
        )
        ax2.imshow(foreground_mask, cmap="gray")
        ax2.plot(cx, cy, "r+", markersize=15, markeredgewidth=2)
        ax2.set_title(
            f"Foreground Mask (sat>{calibrator.saturation_threshold}, "
            f"val>{calibrator.value_threshold})"
        )
        ax2.axis("off")

        # --- Plot 3: Color-coded scatter plot ---
        ax3 = axes[1, 0]
        raw_hue_values = np.array([s.hue_mean for s in self.samples])

        # Check for saturation warnings
        saturated_angles = set()
        if self.warnings:
            for warning in self.warnings:
                if "SATURATION" in warning.upper():
                    for sample in self.samples:
                        saturated_angles.add(sample.angle)

        for i, (shifted_hue, angle) in enumerate(zip(self.hue_values, self.angles)):
            raw_hue = raw_hue_values[i]
            hsv_color = np.array([[[raw_hue, 1.0, 1.0]]])
            rgb_color = mcolors_local.hsv_to_rgb(hsv_color)[0, 0]
            marker = "X" if self.samples[i].angle in saturated_angles else "o"
            edge_color = "red" if self.samples[i].angle in saturated_angles else "black"
            ax3.scatter(
                shifted_hue,
                angle,
                s=100,
                c=[rgb_color],
                edgecolors=edge_color,
                linewidths=2,
                marker=marker,
                zorder=5,
            )

        # Regression line
        hue_line = np.linspace(0, 1, 100)
        predicted_angles = self.inv_slope * hue_line + self.inv_intercept
        ax3.plot(hue_line, predicted_angles, "r-", linewidth=2, label=f"R^2={self.r_squared:.4f}")
        ax3.set_xlabel("Shifted Hue Value")
        ax3.set_ylabel("Angle (degrees)")
        ax3.set_title("Hue to Angle Calibration")
        ax3.legend(loc="best")
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 180)

        # --- Plot 4: Calibration info text ---
        ax4 = axes[1, 1]
        ax4.axis("off")
        info_text = (
            f"Radial Calibration Results\n"
            f"{'=' * 35}\n\n"
            f"R-squared: {self.r_squared:.6f}\n"
            f"Spokes detected: {len(self.samples)}\n"
            f"Center: y={self.center[0]}, x={self.center[1]}\n"
            f"Hue offset: {self.hue_offset:.4f}\n\n"
            f"Regression (hue -> angle):\n"
            f"  angle = {self.inv_slope:.4f} * hue + "
            f"{self.inv_intercept:.4f}\n\n"
            f"Regression (angle -> hue):\n"
            f"  hue = {self.slope:.6f} * angle + "
            f"{self.intercept:.4f}\n\n"
            f"Sampling: r_inner={calibrator.radius_inner}, "
            f"r_outer={calibrator.radius_outer}\n"
        )

        if extra_info:
            info_text += "\nImaging Settings:\n"
            for key, value in extra_info.items():
                info_text += f"  {key}: {value}\n"

        if self.warnings:
            info_text += f"\nWarnings: {len(self.warnings)}\n"
            for w in self.warnings:
                info_text += f"  - {w}\n"

        info_text += "\nCalibration file saved for use in PPM analysis."

        ax4.text(
            0.1,
            0.9,
            info_text,
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Finalize layout BEFORE adding the colorbar so get_position()
        # returns accurate coordinates
        plt_local.tight_layout()
        plt_local.subplots_adjust(bottom=0.08)

        # Add shifted rainbow colorbar below the scatter plot
        add_shifted_hue_colorbar(ax3, self.hue_offset)

        if output_path is not None:
            plt_local.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
            plt_local.close(fig)
        else:
            plt_local.show()


class RadialCalibrator:
    """Calibrator using radial sampling for connected sunburst patterns.

    This approach samples hue values along radial lines from the center of
    the sunburst pattern, which works better than region segmentation when
    spokes connect at the center.

    Example:
        >>> calibrator = RadialCalibrator(n_spokes=17)
        >>> result = calibrator.calibrate("calibration_slide.tif")
        >>> print(f"R-squared: {result.r_squared:.4f}")
        >>> result.save("calibration.npz")
    """

    def __init__(
        self,
        n_spokes: int = 16,
        radius_inner: int = 30,
        radius_outer: int = 150,
        saturation_threshold: float = 0.2,
        value_threshold: float = 0.2,
        min_samples_per_angle: int = 5,
        rotation_search_degrees: float = 5.0,
    ):
        """Initialize the radial calibrator.

        Args:
            n_spokes: Number of unique orientations in the sunburst (default 16).
                      This gives 17 spokes from horizontal to horizontal INCLUDING
                      both endpoints. Angles sampled at 180/n_spokes = 11.25 deg intervals.
            radius_inner: Inner radius to start sampling (pixels from center)
            radius_outer: Outer radius to stop sampling
            saturation_threshold: Minimum saturation to sample a pixel
            value_threshold: Minimum value (brightness) to sample a pixel
            min_samples_per_angle: Minimum samples needed for a valid angle
            rotation_search_degrees: Search range (+/-) in degrees to find spoke centers
        """
        self.n_spokes = n_spokes
        self.radius_inner = radius_inner
        self.radius_outer = radius_outer
        self.saturation_threshold = saturation_threshold
        self.value_threshold = value_threshold
        self.min_samples_per_angle = min_samples_per_angle
        self.rotation_search_degrees = rotation_search_degrees

    def save_detection_mask(
        self,
        image: np.ndarray,
        output_path: Union[str, Path],
        dpi: int = 150,
    ) -> None:
        """Save a debug visualization of the foreground detection mask.

        Creates a 2x2 figure showing the original image, raw foreground mask,
        cleaned connected-component labels, and an overlay of detected regions
        on the original. Useful for troubleshooting when spoke detection fails.

        Uses the same HSV saturation and value thresholds configured on this
        calibrator instance.

        Args:
            image: Original RGB image (uint8 or float)
            output_path: Path to save the mask visualization PNG
            dpi: DPI for the saved image (default 150)
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt_local
        from skimage import color as skcolor, morphology as skmorph, measure
        from scipy import ndimage as ndi

        # Normalize to uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                img_uint8 = (image * 255).astype(np.uint8)
            else:
                img_uint8 = image.astype(np.uint8)
        else:
            img_uint8 = image

        hsv = skcolor.rgb2hsv(img_uint8)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        # Create foreground mask using same thresholds as calibration
        foreground_mask = (sat > self.saturation_threshold) & (val > self.value_threshold)

        # Clean up mask
        try:
            foreground_clean = skmorph.remove_small_objects(foreground_mask, min_size=100)
            foreground_clean = skmorph.remove_small_holes(foreground_clean, area_threshold=500)
            foreground_clean = ndi.median_filter(foreground_clean.astype(np.uint8), size=5).astype(
                bool
            )
        except Exception:
            foreground_clean = foreground_mask

        labels = measure.label(foreground_clean)
        n_regions = labels.max()

        fig, axes = plt_local.subplots(2, 2, figsize=(14, 12))

        axes[0, 0].imshow(img_uint8)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(foreground_mask, cmap="gray")
        axes[0, 1].set_title(
            f"Foreground Mask (sat>{self.saturation_threshold}, " f"val>{self.value_threshold})"
        )
        axes[0, 1].axis("off")

        axes[1, 0].imshow(labels, cmap="nipy_spectral")
        axes[1, 0].set_title(f"Detected Regions: {n_regions} found")
        axes[1, 0].axis("off")

        overlay = img_uint8.copy()
        overlay[foreground_clean, 1] = np.minimum(255, overlay[foreground_clean, 1] + 100)
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title("Overlay (detected regions highlighted)")
        axes[1, 1].axis("off")

        fig.suptitle(
            f"Detection Debug - Saturation threshold: "
            f"{self.saturation_threshold}, "
            f"Value threshold: {self.value_threshold}\n"
            f"Regions found: {n_regions} "
            f"(need at least 3 for calibration)",
            fontsize=12,
        )

        plt_local.tight_layout()
        plt_local.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
        plt_local.close(fig)

    def calibrate(
        self,
        image_path: Union[str, Path],
        center: Optional[Tuple[int, int]] = None,
        debug_plot: bool = False,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> RadialCalibrationResult:
        """Perform radial calibration on a sunburst slide image.

        Args:
            image_path: Path to calibration slide image
            center: Optional (y, x) center coordinates. If None, auto-detected.
            debug_plot: If True, show debug visualization
            roi: Optional region of interest (y1, y2, x1, x2) to restrict search

        Returns:
            RadialCalibrationResult with regression model and data
        """
        # Load image (normalized to uint8)
        image = self._load_image(image_path)

        # Also load raw image for saturation checking
        raw_image = self._load_raw_image(image_path)

        # Convert to HSV
        hsv = color.rgb2hsv(image)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        # Auto-detect center if not provided
        if center is None:
            center = self._find_center(saturation, value, roi, image=image)

        # Refine center by testing nearby positions and picking the one
        # that gives the most consistent spoke hue readings
        center = self._refine_center(hue, saturation, value, center)

        # Sample along radial lines (finds optimal rotation to hit spoke centers)
        samples, rotation = self._radial_sample(hue, saturation, value, center)

        if len(samples) < 3:
            raise ValueError(
                f"Only {len(samples)} valid angles found. Need at least 3 for regression."
            )

        # Check for saturation and other issues
        warnings = self._check_saturation(raw_image, samples, center, rotation)

        # Fit regression
        result = self._fit_regression(samples, center, rotation, warnings)

        if debug_plot:
            self._plot_debug(image, hue, samples, result, center, rotation)

        return result

    def _load_image(self, path: Union[str, Path]) -> np.ndarray:
        """Load image from file."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        try:
            import tifffile

            image = tifffile.imread(str(path))
        except Exception:
            image = imread(str(path))

        # Handle multi-channel TIFF
        if image.ndim == 3:
            if image.shape[0] == 3 and image.shape[2] != 3:
                image = np.moveaxis(image, 0, -1)
            elif image.shape[2] == 4:
                image = image[:, :, :3]
        elif image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        return image

    def _load_raw_image(self, path: Union[str, Path]) -> np.ndarray:
        """Load raw image without normalization for saturation checking."""
        path = Path(path)

        try:
            import tifffile

            image = tifffile.imread(str(path))
        except Exception:
            image = imread(str(path))

        # Handle multi-channel TIFF
        if image.ndim == 3:
            if image.shape[0] == 3 and image.shape[2] != 3:
                image = np.moveaxis(image, 0, -1)
            elif image.shape[2] == 4:
                image = image[:, :, :3]
        elif image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        return image

    def _check_saturation(
        self,
        raw_image: np.ndarray,
        samples: List[RadialSample],
        center: Tuple[int, int],
        rotation: float,
    ) -> List[str]:
        """Check for pixel saturation and other issues along sampling lines.

        Returns:
            List of warning messages
        """
        warnings = []
        cy, cx = center
        h, w = raw_image.shape[:2]

        # Determine max value based on dtype
        if raw_image.dtype == np.uint8:
            max_val = 255
        elif raw_image.dtype == np.uint16:
            max_val = 65535
        else:
            max_val = raw_image.max()

        total_sampled = 0
        saturated_count = 0
        saturated_by_channel = {"R": 0, "G": 0, "B": 0}

        base_angles = np.linspace(0, 180, self.n_spokes, endpoint=False)

        for angle in base_angles:
            rotated_angle = angle + rotation
            angle_rad = np.radians(rotated_angle)

            for r in range(self.radius_inner, self.radius_outer, 2):
                x = int(cx + r * np.cos(angle_rad))
                y = int(cy - r * np.sin(angle_rad))

                if 0 <= y < h and 0 <= x < w:
                    total_sampled += 1
                    pixel = raw_image[y, x]

                    if pixel[0] >= max_val:
                        saturated_by_channel["R"] += 1
                    if pixel[1] >= max_val:
                        saturated_by_channel["G"] += 1
                    if pixel[2] >= max_val:
                        saturated_by_channel["B"] += 1

                    if np.any(pixel >= max_val):
                        saturated_count += 1

        # Add saturation warnings
        if saturated_count > 0:
            pct = 100 * saturated_count / total_sampled
            warnings.append(
                f"SATURATION: {saturated_count}/{total_sampled} pixels ({pct:.1f}%) "
                f"are saturated (R:{saturated_by_channel['R']}, "
                f"G:{saturated_by_channel['G']}, B:{saturated_by_channel['B']})"
            )

        # Check for spokes with low sample count or high variance
        expected_samples = (self.radius_outer - self.radius_inner) // 2
        for sample in samples:
            if sample.n_samples < expected_samples * 0.6:
                warnings.append(
                    f"LOW_SAMPLES: Angle {sample.angle:.1f} deg has only {sample.n_samples} "
                    f"samples (expected ~{expected_samples}). Spoke may be faded/missing."
                )
            if sample.hue_std > 0.1:
                warnings.append(
                    f"HIGH_VARIANCE: Angle {sample.angle:.1f} deg has hue std={sample.hue_std:.3f}. "
                    f"Measurement may be unreliable."
                )

        return warnings

    def _find_center(
        self,
        saturation: np.ndarray,
        value: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
        image: Optional[np.ndarray] = None,
    ) -> Tuple[int, int]:
        """Auto-detect the center of the sunburst pattern.

        Uses mode-based background detection: finds the most common pixel color
        (background), then identifies foreground pixels as those differing
        significantly from the background. The sunburst is identified as the
        leftmost large connected component.

        Args:
            saturation: Saturation channel
            value: Value channel
            roi: Optional (y1, y2, x1, x2) to restrict search
            image: Optional RGB image for mode-based detection (preferred)

        Returns:
            (y, x) center coordinates
        """
        # Try mode-based detection if we have the RGB image
        if image is not None:
            center = self._find_center_mode_based(image, roi)
            if center is not None:
                return center

        # Fall back to saturation-based detection
        return self._find_center_saturation_based(saturation, value, roi)

    def _find_center_mode_based(
        self,
        image: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[Tuple[int, int]]:
        """Find center using mode-based background detection.

        Detects foreground pixels by their Euclidean distance from the
        background color (estimated via per-channel mode). At higher
        magnifications the individual spokes may not be connected, so a
        morphological dilation is applied to merge nearby spoke regions
        before selecting the largest connected component.

        Args:
            image: RGB image
            roi: Optional region of interest

        Returns:
            (y, x) center coordinates, or None if detection fails
        """
        # Find background color using mode (most common pixel value per channel)
        r_mode = stats.mode(image[:, :, 0].flatten(), keepdims=False).mode
        g_mode = stats.mode(image[:, :, 1].flatten(), keepdims=False).mode
        b_mode = stats.mode(image[:, :, 2].flatten(), keepdims=False).mode
        background = np.array([r_mode, g_mode, b_mode], dtype=np.float64)

        # Calculate distance from background for each pixel
        diff = np.sqrt(np.sum((image.astype(np.float64) - background) ** 2, axis=2))

        # Threshold to find non-background pixels (30 is a reasonable default)
        foreground_threshold = 30
        foreground = diff > foreground_threshold

        # Apply ROI if provided
        if roi is not None:
            y1, y2, x1, x2 = roi
            roi_mask = np.zeros_like(foreground)
            roi_mask[y1:y2, x1:x2] = foreground[y1:y2, x1:x2]
            foreground = roi_mask

        # Check if we have enough foreground
        if np.sum(foreground) < 1000:
            return None

        # Dilate the foreground mask to merge nearby spoke regions.
        # At higher magnifications spokes are separate connected components;
        # dilation bridges the gaps so they form a single cluster.
        selem = morphology.disk(10)
        foreground_merged = ndimage.binary_dilation(foreground, structure=selem)

        # Find connected components on the merged mask
        labeled, n_labels = ndimage.label(foreground_merged)
        if n_labels == 0:
            return None

        # Select the largest connected component (the sunburst cluster)
        regions = measure.regionprops(labeled)
        sunburst = max(regions, key=lambda r: r.area)

        # Use the bounding box center of the largest merged region
        y_min, x_min, y_max, x_max = sunburst.bbox
        bbox_height = y_max - y_min
        bbox_width = x_max - x_min

        # Vertical center is the middle of the bounding box
        center_y = (y_min + y_max) // 2

        # If the region is much wider than tall, gratings may extend to the
        # right -- the hub is at the left side of the circular portion
        if bbox_width > bbox_height * 1.2:
            center_x = x_min + bbox_height // 2
        else:
            center_x = (x_min + x_max) // 2

        return (int(center_y), int(center_x))

    def _find_center_saturation_based(
        self,
        saturation: np.ndarray,
        value: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[int, int]:
        """Find center using saturation-based detection (fallback method).

        Args:
            saturation: Saturation channel
            value: Value channel
            roi: Optional region of interest

        Returns:
            (y, x) center coordinates
        """
        # Create foreground mask
        mask = (saturation > self.saturation_threshold) & (value > self.value_threshold)

        # Apply ROI if provided
        if roi is not None:
            y1, y2, x1, x2 = roi
            roi_mask = np.zeros_like(mask)
            roi_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
            mask = roi_mask

        # Check if we have any foreground
        if not mask.any():
            # Fall back to image center
            return (saturation.shape[0] // 2, saturation.shape[1] // 4)

        # Dilate to merge nearby spoke regions (same as mode-based method)
        selem = morphology.disk(10)
        mask_merged = ndimage.binary_dilation(mask, structure=selem)

        # Label connected components on the merged mask
        labeled, num_features = ndimage.label(mask_merged)

        if num_features == 0:
            return (saturation.shape[0] // 2, saturation.shape[1] // 4)

        # Select the largest connected component (the sunburst cluster)
        regions = measure.regionprops(labeled)
        sunburst = max(regions, key=lambda r: r.area)

        # Use bounding box center
        y_min, x_min, y_max, x_max = sunburst.bbox
        bbox_height = y_max - y_min
        bbox_width = x_max - x_min

        # Center y is the middle of the vertical extent
        center_y = (y_min + y_max) // 2

        # Center x: if width >> height, gratings are included on the right
        if bbox_width > bbox_height * 1.2:
            center_x = x_min + bbox_height // 2
        else:
            center_x = (x_min + x_max) // 2

        return (int(center_y), int(center_x))

    def _refine_center(
        self,
        hue: np.ndarray,
        saturation: np.ndarray,
        value: np.ndarray,
        initial_center: Tuple[int, int],
        search_radius: int = 6,
        step: int = 2,
    ) -> Tuple[int, int]:
        """Refine center position by testing nearby candidates.

        Tests a grid of center positions around the initial estimate,
        running the full radial sampling and regression for each candidate.
        Selects the center that produces the highest R-squared.

        The search grid is small (default +/- 6 pixels, step 2) since
        the initial center from auto-detection is usually close. Each
        candidate runs the full sampling pipeline including per-spoke
        local refinement, so results are reliable.

        Args:
            hue: Hue channel (0-1)
            saturation: Saturation channel
            value: Value channel
            initial_center: (y, x) initial center estimate
            search_radius: Pixels to search in each direction (default 6)
            step: Step size in pixels (default 2)

        Returns:
            (y, x) refined center coordinates
        """
        h, w = hue.shape
        cy0, cx0 = initial_center

        best_r_squared = -1.0
        best_center = initial_center

        for dy in range(-search_radius, search_radius + 1, step):
            for dx in range(-search_radius, search_radius + 1, step):
                cy = cy0 + dy
                cx = cx0 + dx

                if cy < 0 or cy >= h or cx < 0 or cx >= w:
                    continue

                # Run full sampling pipeline for this candidate center
                samples, rotation = self._radial_sample(hue, saturation, value, (cy, cx))

                if len(samples) < 3:
                    continue

                # Fit regression and score by R-squared
                result = self._fit_regression(samples, (cy, cx), rotation)

                if result.r_squared > best_r_squared:
                    best_r_squared = result.r_squared
                    best_center = (cy, cx)

        return best_center

    def _radial_sample(
        self,
        hue: np.ndarray,
        saturation: np.ndarray,
        value: np.ndarray,
        center: Tuple[int, int],
    ) -> Tuple[List[RadialSample], float]:
        """Sample hue values along radial lines at spoke centers.

        Two-stage alignment:
        1. Global rotation search to roughly align sampling grid with spokes
        2. Per-spoke local refinement to find each spoke's actual center

        The local refinement is necessary because real sunburst patterns have
        slight non-uniformities in spoke spacing. A global rotation that works
        well on average can still miss individual spokes by 2-3 degrees, which
        is enough to land in the gap between spokes at higher magnifications.

        Args:
            hue: Hue channel (0-1)
            saturation: Saturation channel
            value: Value channel
            center: (y, x) center coordinates

        Returns:
            Tuple of (List of RadialSample objects, global rotation in degrees)
        """
        center_y, center_x = center
        h, w = hue.shape

        # Base angles at spoke centers (180/n_spokes spacing)
        base_angles = np.linspace(0, 180, self.n_spokes, endpoint=False)
        spoke_half_width = (180.0 / self.n_spokes) / 2.0

        # Stage 1: Global rotation search (coarse alignment)
        best_rotation = 0.0
        best_total_saturation = -1

        rotation_steps = np.linspace(
            -self.rotation_search_degrees,
            self.rotation_search_degrees,
            21,  # Test 21 rotations within range
        )

        for rotation in rotation_steps:
            total_saturation = 0
            for angle in base_angles:
                rotated_angle = angle + rotation
                angle_rad = np.radians(rotated_angle)

                for r in range(self.radius_inner, self.radius_outer, 2):
                    x = int(center_x + r * np.cos(angle_rad))
                    y = int(center_y - r * np.sin(angle_rad))

                    if 0 <= y < h and 0 <= x < w:
                        if value[y, x] > self.value_threshold:
                            total_saturation += saturation[y, x]

            if total_saturation > best_total_saturation:
                best_total_saturation = total_saturation
                best_rotation = rotation

        # Stage 2: Per-spoke local refinement and sampling
        # For each spoke, search a small range around the globally rotated
        # position to find the actual spoke center.
        #
        # Refinement criterion: minimum circular hue standard deviation
        # among foreground pixels, with a minimum sample count floor.
        # A line through the center of a spoke has consistent hue (low
        # circular std), while a line in the gap between spokes samples
        # dark pixels with random/noisy hue values and high circular std.
        #
        # The search range is limited to +/- 3 degrees to prevent jumping
        # to an adjacent spoke. Real non-uniformities are typically <3 deg.
        local_search_range = min(3.0, spoke_half_width)
        expected_samples = (self.radius_outer - self.radius_inner) // 2
        min_local_samples = max(self.min_samples_per_angle, expected_samples // 2)

        samples = []
        for angle in base_angles:
            globally_rotated = angle + best_rotation

            # Fine search around globally rotated position
            best_local_std = float("inf")
            best_local_angle = globally_rotated
            best_local_hue_samples = None
            local_steps = np.linspace(-local_search_range, local_search_range, 21)

            for local_offset in local_steps:
                test_angle = globally_rotated + local_offset
                angle_rad = np.radians(test_angle)
                local_hues = []

                for r in range(self.radius_inner, self.radius_outer, 2):
                    x = int(center_x + r * np.cos(angle_rad))
                    y = int(center_y - r * np.sin(angle_rad))

                    if 0 <= y < h and 0 <= x < w:
                        if (
                            saturation[y, x] > self.saturation_threshold
                            and value[y, x] > self.value_threshold
                        ):
                            local_hues.append(hue[y, x])

                # Require enough samples to get a reliable circular std
                if len(local_hues) >= min_local_samples:
                    local_std = _circular_hue_std(local_hues)
                    if local_std < best_local_std:
                        best_local_std = local_std
                        best_local_angle = test_angle
                        best_local_hue_samples = local_hues

            # Fall back to global rotation position if no candidate met
            # the sample count requirement
            if best_local_hue_samples is None:
                angle_rad = np.radians(globally_rotated)
                fallback_hues = []
                for r in range(self.radius_inner, self.radius_outer, 2):
                    x = int(center_x + r * np.cos(angle_rad))
                    y = int(center_y - r * np.sin(angle_rad))
                    if 0 <= y < h and 0 <= x < w:
                        if (
                            saturation[y, x] > self.saturation_threshold
                            and value[y, x] > self.value_threshold
                        ):
                            fallback_hues.append(hue[y, x])
                if len(fallback_hues) >= self.min_samples_per_angle:
                    best_local_hue_samples = fallback_hues
                    best_local_angle = globally_rotated
                    best_local_std = _circular_hue_std(fallback_hues)

            if best_local_hue_samples is not None:
                sample = RadialSample(
                    angle=best_local_angle,
                    hue_mean=_circular_hue_mean(best_local_hue_samples),
                    hue_std=best_local_std,
                    n_samples=len(best_local_hue_samples),
                )
                samples.append(sample)

        return samples, best_rotation

    def _fit_regression(
        self,
        samples: List[RadialSample],
        center: Tuple[int, int],
        rotation: float = 0.0,
        warnings: List[str] = None,
    ) -> RadialCalibrationResult:
        """Fit linear regression between angles and hue values.

        Handles hue wrapping by finding the optimal offset that produces
        the best linear fit. Since hue wraps at 1.0, data may be split
        into two groups that need to be unwrapped for proper fitting.
        """
        if warnings is None:
            warnings = []
        angles = np.array([s.angle for s in samples])
        hue_values_raw = np.array([s.hue_mean for s in samples])

        # Find optimal hue offset by testing multiple values
        best_r_squared = -1
        best_offset = 0
        best_coeffs = None

        for offset in np.linspace(0, 1, 50, endpoint=False):
            # Shift hue values by offset and wrap to [0, 1]
            hue_shifted = (hue_values_raw - offset) % 1.0

            # Fit: angle = m * hue + b
            coeffs = np.polyfit(hue_shifted, angles, 1)

            # Calculate R-squared
            predicted = np.polyval(coeffs, hue_shifted)
            ss_res = np.sum((angles - predicted) ** 2)
            ss_tot = np.sum((angles - np.mean(angles)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            if r_squared > best_r_squared:
                best_r_squared = r_squared
                best_offset = offset
                best_coeffs = coeffs

        # Apply best offset
        hue_values = (hue_values_raw - best_offset) % 1.0
        inv_slope = best_coeffs[0]
        inv_intercept = best_coeffs[1]
        r_squared = best_r_squared

        # Fit inverse: hue = m * angle + b (using shifted hue values)
        inv_coeffs = np.polyfit(angles, hue_values, 1)
        slope = inv_coeffs[0]
        intercept = inv_coeffs[1]

        return RadialCalibrationResult(
            slope=slope,
            intercept=intercept,
            inv_slope=inv_slope,
            inv_intercept=inv_intercept,
            r_squared=r_squared,
            hue_offset=best_offset,
            angles=angles,
            hue_values=hue_values,
            samples=samples,
            center=center,
            warnings=warnings,
            rotation=rotation,
        )

    def _plot_debug(
        self,
        image: np.ndarray,
        hue: np.ndarray,
        samples: List[RadialSample],
        result: RadialCalibrationResult,
        center: Tuple[int, int],
        rotation: float = 0.0,
    ) -> None:
        """Create debug visualization with ROYGBIV color indicators.

        Delegates to result.save_plot() with output_path=None for
        interactive display.
        """
        result.save_plot(output_path=None, image=image, calibrator=self)

    def _add_shifted_hue_colorbar(self, ax, hue_offset: float) -> None:
        """Add a rainbow colorbar shifted by hue_offset.

        Delegates to the module-level function.

        Args:
            ax: Matplotlib axes
            hue_offset: Hue offset to shift the colorbar
        """
        add_shifted_hue_colorbar(ax, hue_offset)


def calibrate_radial(
    image_path: Union[str, Path],
    n_spokes: int = 17,
    debug: bool = False,
) -> RadialCalibrationResult:
    """Convenience function for radial calibration.

    Args:
        image_path: Path to calibration slide image
        n_spokes: Number of spokes in the sunburst pattern
        debug: Show debug plot

    Returns:
        RadialCalibrationResult
    """
    calibrator = RadialCalibrator(n_spokes=n_spokes)
    return calibrator.calibrate(image_path, debug_plot=debug)
