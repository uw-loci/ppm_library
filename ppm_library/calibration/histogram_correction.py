"""
Histogram anisotropy correction for PPM images.

Provides tools to correct for optical anisotropy in the PPM system using
calibration data from circular and perpendicular calibration patterns.

The correction addresses systematic variations in hue histogram intensity
that arise from the optical properties of the polarization microscopy system.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy import interpolate, signal


@dataclass
class HistogramCalibration:
    """Calibration data for histogram anisotropy correction.

    This class holds the correction curve and phase shift derived from
    calibration patterns, and provides methods to apply corrections to
    measured hue histograms.

    Attributes:
        correction_curve: 256-element array of correction factors (normalized to max=1)
        phase_shift: Number of bins to shift (positive = shift right)
        phase_direction: Direction of phase shift (+1 or -1)
        peak_locations: Bin locations of detected peaks in circular pattern
        peak_heights: Heights of detected peaks

    Example:
        >>> from ppm_library.calibration import HistogramCalibration
        >>>
        >>> # Create from calibration data files
        >>> cal = HistogramCalibration.from_histogram_files(
        ...     circular_histogram_path="circular_pattern.csv",
        ...     perpendicular_histogram_path="perpendicular_pattern.csv"
        ... )
        >>>
        >>> # Apply correction to a measured histogram
        >>> corrected = cal.correct_histogram(raw_histogram)
        >>>
        >>> # Save for later use
        >>> cal.save("histogram_calibration.npz")
    """

    correction_curve: np.ndarray  # Shape (256,), normalized correction factors
    phase_shift: int  # Bins to shift
    phase_direction: int  # +1 or -1
    peak_locations: np.ndarray = field(default_factory=lambda: np.array([]))
    peak_heights: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self) -> None:
        """Validate calibration data."""
        if len(self.correction_curve) != 256:
            raise ValueError(
                f"Correction curve must have 256 elements, got {len(self.correction_curve)}"
            )
        if self.phase_direction not in (-1, 0, 1):
            raise ValueError(f"Phase direction must be -1, 0, or 1, got {self.phase_direction}")

    @classmethod
    def from_circular_histogram(
        cls,
        histogram: np.ndarray,
        n_peaks: int = 16,
        min_prominence: Optional[float] = None,
        reference_bin: int = 128,
    ) -> "HistogramCalibration":
        """Create calibration from a circular pattern histogram.

        The circular pattern contains all fiber orientations equally, so the
        histogram should have equal peak heights. Variations in peak height
        are due to optical anisotropy and should be corrected.

        Args:
            histogram: 256-bin hue histogram from circular calibration pattern
            n_peaks: Expected number of peaks (default 16 for standard pattern)
            min_prominence: Minimum peak prominence for detection.
                If None, auto-tunes to find exactly n_peaks.
            reference_bin: Expected bin for 90-degree orientation (default 128)

        Returns:
            HistogramCalibration instance
        """
        histogram = np.asarray(histogram).flatten()
        if len(histogram) != 256:
            raise ValueError(f"Histogram must have 256 bins, got {len(histogram)}")

        # Find peaks with auto-tuning if needed
        peak_locs, peak_heights = _find_n_peaks(histogram, n_peaks, min_prominence)

        # Create correction curve by interpolating between peaks
        correction_curve = _interpolate_peaks(peak_locs, peak_heights, n_bins=256)

        # Normalize
        correction_curve = correction_curve / np.max(correction_curve)

        # For circular pattern alone, phase shift is 0 (need perpendicular pattern)
        phase_shift = 0
        phase_direction = 1

        return cls(
            correction_curve=correction_curve,
            phase_shift=phase_shift,
            phase_direction=phase_direction,
            peak_locations=peak_locs,
            peak_heights=peak_heights,
        )

    @classmethod
    def from_histogram_files(
        cls,
        circular_histogram_path: Union[str, Path],
        perpendicular_histogram_path: Optional[Union[str, Path]] = None,
        n_peaks: int = 16,
        reference_bin: int = 128,
    ) -> "HistogramCalibration":
        """Create calibration from histogram CSV files.

        Args:
            circular_histogram_path: Path to CSV with circular pattern histogram.
                Expected format: two columns (bin, count) or single column (count).
            perpendicular_histogram_path: Path to CSV with perpendicular pattern histogram.
                Used to determine phase shift. If None, phase_shift=0.
            n_peaks: Expected number of peaks in circular pattern (default 16)
            reference_bin: Expected bin for 90-degree orientation (default 128)

        Returns:
            HistogramCalibration instance
        """
        # Load circular pattern
        circular_data = np.loadtxt(circular_histogram_path, delimiter=",")
        if circular_data.ndim == 2:
            circular_hist = circular_data[:, 1] if circular_data.shape[1] >= 2 else circular_data[:, 0]
        else:
            circular_hist = circular_data

        # Create base calibration from circular pattern
        cal = cls.from_circular_histogram(circular_hist, n_peaks=n_peaks)

        # Determine phase shift from perpendicular pattern if provided
        if perpendicular_histogram_path is not None:
            perp_data = np.loadtxt(perpendicular_histogram_path, delimiter=",")
            if perp_data.ndim == 2:
                perp_hist = perp_data[:, 1] if perp_data.shape[1] >= 2 else perp_data[:, 0]
            else:
                perp_hist = perp_data

            # Find single peak in perpendicular pattern
            perp_locs, _ = _find_n_peaks(perp_hist, n_peaks=1, min_prominence=None)

            if len(perp_locs) > 0:
                perp_peak = perp_locs[0]
                # Phase shift is difference from expected 90-degree bin
                phase_shift = int(abs(reference_bin - perp_peak))
                phase_direction = int(np.sign(reference_bin - perp_peak))
                if phase_direction == 0:
                    phase_direction = 1

                cal = cls(
                    correction_curve=cal.correction_curve,
                    phase_shift=phase_shift,
                    phase_direction=phase_direction,
                    peak_locations=cal.peak_locations,
                    peak_heights=cal.peak_heights,
                )

        return cal

    def correct_histogram(
        self,
        histogram: np.ndarray,
        apply_phase_shift: bool = True,
        remove_background: bool = True,
    ) -> np.ndarray:
        """Apply anisotropy correction to a histogram.

        Args:
            histogram: 256-bin hue histogram to correct
            apply_phase_shift: Whether to apply phase shift correction (default True)
            remove_background: Whether to remove bin 0 (background pixels) (default True)

        Returns:
            Corrected histogram (256 bins, normalized to max=1)
        """
        histogram = np.asarray(histogram).flatten()
        if len(histogram) != 256:
            raise ValueError(f"Histogram must have 256 bins, got {len(histogram)}")

        # Make a copy
        corrected = histogram.astype(np.float64).copy()

        # Remove background (bin 0) if requested
        if remove_background:
            corrected[0] = 0

        # Divide by correction curve (avoid division by zero)
        safe_curve = np.where(self.correction_curve > 0.01, self.correction_curve, 0.01)
        corrected = corrected / safe_curve

        # Apply phase shift
        if apply_phase_shift and self.phase_shift != 0:
            shift = self.phase_shift * self.phase_direction
            corrected = np.roll(corrected, shift)

        # Normalize
        if np.max(corrected) > 0:
            corrected = corrected / np.max(corrected)

        return corrected

    def correct_hue_image(
        self,
        hue_image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply pixel-wise anisotropy correction to a hue image.

        This applies the inverse of the correction curve to each pixel's
        hue value, effectively pre-correcting the image so that subsequent
        histogram analysis doesn't need correction.

        Args:
            hue_image: Hue values as float (0-1) or uint8 (0-255)
            mask: Optional boolean mask (only correct where True)

        Returns:
            Corrected hue image (same dtype as input)
        """
        # Determine input format
        is_uint8 = hue_image.dtype == np.uint8
        if is_uint8:
            hue = hue_image.astype(np.float64)
        else:
            hue = hue_image.astype(np.float64)
            if hue.max() <= 1.0:
                hue = hue * 255.0

        # Create corrected image
        corrected = hue.copy()

        # Apply correction curve (divide by curve value at each hue bin)
        # This is pixel-wise, so we look up each pixel's correction factor
        if mask is not None:
            pixels_to_correct = mask
        else:
            pixels_to_correct = hue > 0  # Skip background

        for i in range(256):
            bin_mask = (np.floor(hue) == i) & pixels_to_correct
            if np.any(bin_mask):
                correction = self.correction_curve[i] if self.correction_curve[i] > 0.01 else 0.01
                corrected[bin_mask] = hue[bin_mask] / correction

        # Apply phase shift
        if self.phase_shift != 0:
            shift_hue = (self.phase_shift * self.phase_direction) / 255.0 * 180.0
            # Convert shift to hue units (0-255 = 0-180 degrees)
            shift_bins = (self.phase_shift * self.phase_direction)
            corrected = np.mod(corrected + shift_bins, 255)

        # Normalize back to 0-255 range
        corrected = np.clip(corrected, 0, 255)

        # Convert back to original format
        if is_uint8:
            return corrected.astype(np.uint8)
        else:
            if hue_image.max() <= 1.0:
                return corrected / 255.0
            return corrected

    def save(self, path: Union[str, Path]) -> None:
        """Save calibration to a .npz file.

        Args:
            path: Output file path (should end in .npz)
        """
        path = Path(path)
        np.savez(
            path,
            correction_curve=self.correction_curve,
            phase_shift=np.array([self.phase_shift]),
            phase_direction=np.array([self.phase_direction]),
            peak_locations=self.peak_locations,
            peak_heights=self.peak_heights,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "HistogramCalibration":
        """Load calibration from a .npz file.

        Args:
            path: Path to .npz file

        Returns:
            HistogramCalibration instance
        """
        path = Path(path)
        data = np.load(path)
        return cls(
            correction_curve=data["correction_curve"],
            phase_shift=int(data["phase_shift"][0]),
            phase_direction=int(data["phase_direction"][0]),
            peak_locations=data.get("peak_locations", np.array([])),
            peak_heights=data.get("peak_heights", np.array([])),
        )

    def plot_correction_curve(
        self,
        ax=None,
        show_peaks: bool = True,
    ):
        """Plot the correction curve.

        Args:
            ax: Matplotlib axes (if None, creates new figure)
            show_peaks: Whether to mark detected peaks

        Returns:
            Matplotlib axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        # Plot correction curve
        bins = np.arange(256)
        angles = bins / 256 * 180  # Convert to degrees

        ax.plot(angles, self.correction_curve, "b-", linewidth=1.5, label="Correction curve")

        if show_peaks and len(self.peak_locations) > 0:
            peak_angles = self.peak_locations / 256 * 180
            ax.scatter(
                peak_angles,
                self.peak_heights / np.max(self.peak_heights),
                c="red",
                s=50,
                zorder=5,
                label=f"Detected peaks ({len(self.peak_locations)})",
            )

        ax.set_xlabel("Fiber Angle (degrees)")
        ax.set_ylabel("Correction Factor (normalized)")
        ax.set_xlim(0, 180)
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.set_title(f"Histogram Anisotropy Correction (phase shift: {self.phase_shift} bins)")

        return ax


def _find_n_peaks(
    histogram: np.ndarray,
    n_peaks: int,
    min_prominence: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find exactly n peaks in a histogram using auto-tuned prominence.

    Args:
        histogram: Input histogram
        n_peaks: Target number of peaks
        min_prominence: Starting prominence value. If None, auto-tunes.

    Returns:
        Tuple of (peak_locations, peak_heights)
    """
    histogram = np.asarray(histogram).flatten()

    if min_prominence is not None:
        peaks, properties = signal.find_peaks(histogram, prominence=min_prominence)
        return peaks, histogram[peaks]

    # Auto-tune prominence to find exactly n_peaks
    prominence = 1.0
    max_iterations = 1000

    for _ in range(max_iterations):
        peaks, properties = signal.find_peaks(histogram, prominence=prominence)

        if len(peaks) == n_peaks:
            return peaks, histogram[peaks]
        elif len(peaks) < n_peaks:
            prominence -= 0.5
            if prominence < 0.1:
                # Can't find enough peaks, return what we have
                break
        else:
            prominence += 1.0

    # Return best effort
    peaks, _ = signal.find_peaks(histogram, prominence=max(prominence, 0.1))
    return peaks, histogram[peaks]


def _interpolate_peaks(
    peak_locations: np.ndarray,
    peak_heights: np.ndarray,
    n_bins: int = 256,
) -> np.ndarray:
    """Interpolate between peaks to create a smooth curve.

    Uses Akima spline (similar to MATLAB's 'makima') for smooth interpolation.

    Args:
        peak_locations: Bin indices of peaks
        peak_heights: Heights of peaks
        n_bins: Number of bins in output (default 256)

    Returns:
        Interpolated curve of length n_bins
    """
    if len(peak_locations) < 2:
        # Not enough peaks for interpolation
        return np.ones(n_bins)

    # Sort by location
    sort_idx = np.argsort(peak_locations)
    locs = peak_locations[sort_idx]
    heights = peak_heights[sort_idx]

    # Use Akima interpolation (similar to MATLAB's 'makima')
    try:
        interp_func = interpolate.Akima1DInterpolator(locs, heights)
        curve = interp_func(np.arange(n_bins))
    except Exception:
        # Fall back to linear interpolation
        curve = np.interp(np.arange(n_bins), locs, heights)

    # Handle NaN values at edges (extrapolation)
    curve = np.nan_to_num(curve, nan=np.nanmean(curve))

    # Ensure positive values
    curve = np.maximum(curve, 0.01)

    return curve


def compute_hue_histogram(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    n_bins: int = 256,
    saturation_threshold: float = 0.2,
    value_threshold: float = 0.2,
) -> np.ndarray:
    """Compute hue histogram from an RGB image.

    Args:
        image: RGB image as numpy array (H, W, 3)
        mask: Optional boolean mask (only include where True)
        n_bins: Number of histogram bins (default 256)
        saturation_threshold: Minimum saturation for valid pixels
        value_threshold: Minimum value/brightness for valid pixels

    Returns:
        Histogram of hue values (n_bins elements)
    """
    from skimage import color

    # Convert to HSV
    if image.dtype == np.uint8:
        image = image.astype(np.float64) / 255.0

    hsv = color.rgb2hsv(image)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Create valid pixel mask
    valid = (saturation > saturation_threshold) & (value > value_threshold)
    if mask is not None:
        valid = valid & mask

    # Extract valid hue values and convert to bins
    hue_values = hue[valid]
    hue_bins = (hue_values * (n_bins - 1)).astype(int)
    hue_bins = np.clip(hue_bins, 0, n_bins - 1)

    # Compute histogram
    histogram, _ = np.histogram(hue_bins, bins=n_bins, range=(0, n_bins))

    return histogram.astype(np.float64)
