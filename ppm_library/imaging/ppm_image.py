"""
PPM Image processing module.

Provides classes for loading PPM images and converting hue values to fiber angles
using calibration data.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from skimage import color
from skimage.io import imread


@dataclass
class AngleMap:
    """Result of applying calibration to extract fiber angles from a PPM image.

    Attributes:
        angles: Fiber orientation angles in degrees (0-180). NaN where invalid.
        valid_mask: Boolean mask indicating valid measurements (sufficient saturation/value).
        hue: Original hue values (0-1).
        saturation: Original saturation values (0-1).
        value: Original value/brightness values (0-1).
    """

    angles: np.ndarray  # Shape (H, W), float, 0-180 degrees, NaN where invalid
    valid_mask: np.ndarray  # Shape (H, W), bool
    hue: np.ndarray  # Shape (H, W), float 0-1
    saturation: np.ndarray  # Shape (H, W), float 0-1
    value: np.ndarray  # Shape (H, W), float 0-1

    @property
    def shape(self) -> Tuple[int, int]:
        """Image dimensions (height, width)."""
        return self.angles.shape

    def get_angles_in_roi(self, mask: np.ndarray) -> np.ndarray:
        """Get valid angle values within a region of interest.

        Args:
            mask: Boolean mask defining the ROI

        Returns:
            1D array of valid angle values within the ROI
        """
        combined_mask = mask & self.valid_mask
        return self.angles[combined_mask]

    def get_mean_angle_in_roi(self, mask: np.ndarray) -> float:
        """Get mean fiber angle within a region of interest.

        Args:
            mask: Boolean mask defining the ROI

        Returns:
            Mean angle in degrees, or NaN if no valid pixels
        """
        angles = self.get_angles_in_roi(mask)
        if len(angles) == 0:
            return np.nan
        return np.mean(angles)

    def get_angle_histogram(
        self,
        mask: Optional[np.ndarray] = None,
        bins: int = 18,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get histogram of fiber angles.

        Args:
            mask: Optional ROI mask. If None, uses all valid pixels.
            bins: Number of histogram bins (default 18 = 10-degree bins)

        Returns:
            Tuple of (counts, bin_edges)
        """
        if mask is not None:
            angles = self.get_angles_in_roi(mask)
        else:
            angles = self.angles[self.valid_mask]

        return np.histogram(angles, bins=bins, range=(0, 180))

    def to_rgb_colormap(self, colormap: str = "hsv") -> np.ndarray:
        """Convert angle map to RGB visualization using a colormap.

        Args:
            colormap: Matplotlib colormap name (default 'hsv' for rainbow)

        Returns:
            RGB image (H, W, 3) as uint8
        """
        import matplotlib.pyplot as plt

        # Normalize angles to 0-1 range
        normalized = self.angles / 180.0

        # Apply colormap
        cmap = plt.get_cmap(colormap)
        rgb = cmap(normalized)[:, :, :3]  # Drop alpha channel

        # Set invalid pixels to gray
        rgb[~self.valid_mask] = 0.5

        return (rgb * 255).astype(np.uint8)


class PPMImage:
    """Container for PPM (Polychromatic Polarization Microscopy) image data.

    Provides methods for extracting hue values and converting to fiber angles
    using calibration data.

    Example:
        >>> from ppm_library.imaging import PPMImage
        >>> from ppm_library.calibration.radial import RadialCalibrationResult
        >>>
        >>> # Load image and calibration
        >>> ppm = PPMImage.load("sample.tif")
        >>> calibration = RadialCalibrationResult.load("calibration.npz")
        >>>
        >>> # Convert to angle map
        >>> angle_map = ppm.to_angle_map(calibration)
        >>> print(f"Mean angle: {angle_map.get_mean_angle_in_roi(roi_mask):.1f} deg")
    """

    def __init__(
        self,
        image: np.ndarray,
        saturation_threshold: float = 0.2,
        value_threshold: float = 0.2,
    ):
        """Initialize PPMImage from RGB array.

        Args:
            image: RGB image as numpy array (H, W, 3), uint8
            saturation_threshold: Minimum HSV saturation for valid measurements
            value_threshold: Minimum HSV value (brightness) for valid measurements
        """
        self._validate_image(image)
        self.image = image
        self.saturation_threshold = saturation_threshold
        self.value_threshold = value_threshold

        # Convert to HSV
        self._hsv = color.rgb2hsv(image)

    def _validate_image(self, image: np.ndarray) -> None:
        """Validate input image format."""
        if image.ndim != 3:
            raise ValueError(f"Expected 3D array (H, W, 3), got {image.ndim}D")
        if image.shape[2] != 3:
            raise ValueError(f"Expected 3 channels, got {image.shape[2]}")

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        saturation_threshold: float = 0.2,
        value_threshold: float = 0.2,
    ) -> "PPMImage":
        """Load PPM image from file.

        Args:
            path: Path to image file (TIFF, PNG, etc.)
            saturation_threshold: Minimum saturation for valid measurements
            value_threshold: Minimum value for valid measurements

        Returns:
            PPMImage instance
        """
        image = _load_image(path)
        return cls(image, saturation_threshold, value_threshold)

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Image dimensions (height, width, channels)."""
        return self.image.shape

    @property
    def hue(self) -> np.ndarray:
        """Hue channel (0-1)."""
        return self._hsv[:, :, 0]

    @property
    def saturation(self) -> np.ndarray:
        """Saturation channel (0-1)."""
        return self._hsv[:, :, 1]

    @property
    def value(self) -> np.ndarray:
        """Value/brightness channel (0-1)."""
        return self._hsv[:, :, 2]

    @property
    def valid_mask(self) -> np.ndarray:
        """Boolean mask of pixels with sufficient saturation and value."""
        return (self.saturation > self.saturation_threshold) & (self.value > self.value_threshold)

    def to_angle_map(
        self,
        calibration: "RadialCalibrationResult",
        roi: Optional[np.ndarray] = None,
    ) -> AngleMap:
        """Convert hue values to fiber angles using calibration.

        Args:
            calibration: RadialCalibrationResult with hue-to-angle mapping
            roi: Optional boolean mask to restrict processing to a region

        Returns:
            AngleMap with fiber orientation angles
        """
        # Import here to avoid circular dependency
        from ppm_library.calibration.radial import RadialCalibrationResult

        if not isinstance(calibration, RadialCalibrationResult):
            raise TypeError(f"Expected RadialCalibrationResult, got {type(calibration).__name__}")

        # Determine valid pixels
        valid = self.valid_mask.copy()
        if roi is not None:
            valid = valid & roi

        # Convert hue to angle for all pixels
        angles = np.full(self.hue.shape, np.nan, dtype=np.float64)
        angles[valid] = calibration.hue_to_angle(self.hue[valid])

        return AngleMap(
            angles=angles,
            valid_mask=valid,
            hue=self.hue.copy(),
            saturation=self.saturation.copy(),
            value=self.value.copy(),
        )

    def get_hue_in_roi(self, mask: np.ndarray) -> np.ndarray:
        """Get valid hue values within a region of interest.

        Args:
            mask: Boolean mask defining the ROI

        Returns:
            1D array of valid hue values within the ROI
        """
        combined_mask = mask & self.valid_mask
        return self.hue[combined_mask]

    def get_mean_hue_in_roi(self, mask: np.ndarray) -> float:
        """Get mean hue value within a region of interest.

        Args:
            mask: Boolean mask defining the ROI

        Returns:
            Mean hue (0-1), or NaN if no valid pixels

        Note:
            This computes a simple arithmetic mean, which may not be appropriate
            for circular hue values near the 0/1 boundary.
        """
        hues = self.get_hue_in_roi(mask)
        if len(hues) == 0:
            return np.nan
        return np.mean(hues)


def load_ppm_image(
    path: Union[str, Path],
    saturation_threshold: float = 0.2,
    value_threshold: float = 0.2,
) -> PPMImage:
    """Convenience function to load a PPM image.

    Args:
        path: Path to image file
        saturation_threshold: Minimum saturation for valid measurements
        value_threshold: Minimum value for valid measurements

    Returns:
        PPMImage instance
    """
    return PPMImage.load(path, saturation_threshold, value_threshold)


def _load_image(path: Union[str, Path]) -> np.ndarray:
    """Load image from file, handling various formats.

    Args:
        path: Path to image file

    Returns:
        RGB image as uint8 array (H, W, 3)
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    # Try tifffile first for TIFF images
    try:
        import tifffile

        image = tifffile.imread(str(path))
    except Exception:
        image = imread(str(path))

    # Handle different channel orderings
    if image.ndim == 3:
        if image.shape[0] == 3 and image.shape[2] != 3:
            # Channels-first format (3, H, W) -> (H, W, 3)
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
