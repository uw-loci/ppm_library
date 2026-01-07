"""
TIFF writing utilities for microscopy image files.

This module contains utilities for writing OME-TIFF files and performing
image processing operations such as birefringence calculations for PPM imaging.
"""

import pathlib
import shutil
from typing import Optional, Dict
import numpy as np
import tifffile as tf
import logging

logger = logging.getLogger(__name__)


class TifWriterUtils:
    """Utilities for writing TIFF files and image processing."""

    def __init__(self):
        pass

    @staticmethod
    def ome_writer(filename: str, pixel_size_um: float, data: np.ndarray):
        """
        Write OME-TIFF file with metadata.

        Args:
            filename: Output filename
            pixel_size_um: Pixel size in micrometers
            data: Image data array
        """
        with tf.TiffWriter(filename) as tif:
            # Use lossless compression for scientific imaging
            # LZW works well for all bit depths and is widely compatible
            # NEVER use JPEG for scientific data (lossy compression)
            options = {
                "photometric": "rgb" if len(data.shape) == 3 else "minisblack",
                "compression": "lzw",  # Lossless compression for all data
                "resolutionunit": "CENTIMETER",
                "maxworkers": 2,
            }
            tif.write(
                data,
                resolution=(1e4 / pixel_size_um, 1e4 / pixel_size_um),
                **options,
            )

    @staticmethod
    def format_imagetags(tags: dict) -> Dict[str, dict]:
        """Format image tags by grouping by prefix."""
        dx = {}
        for k in set([key.split("-")[0] for key in tags]):
            dx.update({k: {key: tags[key] for key in tags if key.startswith(k)}})
        return dx

    @staticmethod
    def create_birefringence_tile(
        pos_image: np.ndarray,
        neg_image: np.ndarray,
        output_dir: pathlib.Path,
        filename: str,
        pixel_size_um: float,
        tile_config_source: Optional[pathlib.Path] = None,
        logger=None,
    ) -> np.ndarray:
        """
        Create a single birefringence image from positive and negative angle images.

        Args:
            pos_image: Positive angle image
            neg_image: Negative angle image
            output_dir: Directory to save birefringence image
            filename: Output filename
            pixel_size_um: Pixel size for OME-TIFF metadata
            tile_config_source: Path to source TileConfiguration.txt to copy (optional)
            logger: Logger instance (optional)

        Returns:
            The birefringence image array
        """
        # Create output directory if it doesn't exist
        if not output_dir.exists():
            output_dir.mkdir(exist_ok=True)

            # Copy TileConfiguration.txt if source provided
            if tile_config_source and tile_config_source.exists():
                shutil.copy2(tile_config_source, output_dir / "TileConfiguration.txt")
                if logger:
                    logger.debug(f"Copied TileConfiguration.txt to {output_dir}")

        # Calculate birefringence (sum of absolute differences)
        output_path = output_dir / filename

        biref_img = TifWriterUtils.ppm_angle_difference(pos_image, neg_image)

        # Save as 16-bit single-channel image (no normalization)
        # Range: 0-765 (sum of absolute RGB differences)
        TifWriterUtils.ome_writer(
            filename=str(output_path),
            pixel_size_um=pixel_size_um,
            data=biref_img,  # Single channel uint16
        )

        if logger:
            logger.info(f"  Created birefringence image: {filename} (16-bit, range: {biref_img.min()}-{biref_img.max()})")

        return biref_img

    @staticmethod
    def subtraction_image(pos_image: np.ndarray, neg_image: np.ndarray) -> np.ndarray:
        """Simple subtraction of two images."""
        return pos_image.astype(np.float32) - neg_image.astype(np.float32)

    @staticmethod
    def ppm_angle_difference(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Calculate angle difference for polarized microscopy images.
        Sum of absolute differences across RGB channels.

        Args:
            img1: First image (RGB, uint8)
            img2: Second image (RGB, uint8)

        Returns:
            Difference image as single channel (uint16), range 0-765 (3 * 255)
        """
        # Convert to int16 to handle negative differences
        img1_i16 = img1.astype(np.int16)
        img2_i16 = img2.astype(np.int16)

        # Calculate absolute difference per channel and sum across RGB
        # |R1-R2| + |G1-G2| + |B1-B2|
        # Note: Background collection now uses the same metric to match backgrounds,
        # ensuring minimal birefringence signal in blank regions.
        abs_diff = np.abs(img1_i16 - img2_i16)
        sum_abs_diff = np.sum(abs_diff, axis=2)

        # Convert to uint16 (range is 0 to 765, well within uint16)
        return sum_abs_diff.astype(np.uint16)

    @staticmethod
    def ppm_normalized_difference(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Calculate normalized birefringence for polarized microscopy images.
        Formula: [I(+) - I(-)]/[I(+) + I(-)], converted to grayscale first.

        This normalization suppresses H&E staining color variations by dividing
        the birefringence signal by total intensity at each pixel.

        Args:
            img1: Positive angle image (RGB, uint8)
            img2: Negative angle image (RGB, uint8)

        Returns:
            Normalized difference as single channel uint16, scaled 0-65535
            where 32768 = 0 (no difference), >32768 = positive, <32768 = negative
        """
        # Convert RGB to grayscale using standard luminance weights
        if len(img1.shape) == 3:
            gray1 = np.dot(img1[..., :3].astype(np.float32), [0.2989, 0.5870, 0.1140])
        else:
            gray1 = img1.astype(np.float32)

        if len(img2.shape) == 3:
            gray2 = np.dot(img2[..., :3].astype(np.float32), [0.2989, 0.5870, 0.1140])
        else:
            gray2 = img2.astype(np.float32)

        # Calculate difference and sum
        diff = gray1 - gray2
        total = gray1 + gray2

        # Avoid division by zero - use small epsilon
        epsilon = 1e-6
        normalized = diff / (total + epsilon)

        # normalized is in range [-1, 1]
        # Scale to uint16: 0 -> 32768, -1 -> 0, +1 -> 65535
        scaled = (normalized + 1.0) * 32767.5

        return np.clip(scaled, 0, 65535).astype(np.uint16)

    @staticmethod
    def ppm_normalized_difference_abs(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Calculate absolute normalized birefringence for polarized microscopy images.
        Formula: |[I(+) - I(-)]/[I(+) + I(-)]|, converted to grayscale first.

        This returns the magnitude of normalized birefringence (always positive).

        Args:
            img1: Positive angle image (RGB, uint8)
            img2: Negative angle image (RGB, uint8)

        Returns:
            Absolute normalized difference as single channel uint16, scaled 0-65535
            where 0 = no birefringence, 65535 = maximum birefringence
        """
        # Convert RGB to grayscale using standard luminance weights
        if len(img1.shape) == 3:
            gray1 = np.dot(img1[..., :3].astype(np.float32), [0.2989, 0.5870, 0.1140])
        else:
            gray1 = img1.astype(np.float32)

        if len(img2.shape) == 3:
            gray2 = np.dot(img2[..., :3].astype(np.float32), [0.2989, 0.5870, 0.1140])
        else:
            gray2 = img2.astype(np.float32)

        # Calculate difference and sum
        diff = gray1 - gray2
        total = gray1 + gray2

        # Avoid division by zero - use small epsilon
        epsilon = 1e-6
        normalized = np.abs(diff / (total + epsilon))

        # normalized is in range [0, 1]
        # Scale to uint16: 0 -> 0, 1 -> 65535
        scaled = normalized * 65535.0

        return np.clip(scaled, 0, 65535).astype(np.uint16)

    @staticmethod
    def ppm_angle_sum(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Calculate angle sum for polarized microscopy images.
        Adds the two images to create a combined RGB image.

        Args:
            img1: First image (RGB)
            img2: Second image (RGB)

        Returns:
            Sum image as RGB
        """
        # Convert to float for calculations
        img1_f = img1.astype(np.float32) / 255.0
        img2_f = img2.astype(np.float32) / 255.0

        # Sum RGB values (average to keep values in [0,1] range)
        rgb_sum = (img1_f + img2_f) / 2.0

        return rgb_sum

    @staticmethod
    def create_sum_tile(
        pos_image: np.ndarray,
        neg_image: np.ndarray,
        output_dir: pathlib.Path,
        filename: str,
        pixel_size_um: float,
        tile_config_source: Optional[pathlib.Path] = None,
        logger=None,
    ) -> np.ndarray:
        """
        Create a single sum image from positive and negative angle images.

        Args:
            pos_image: Positive angle image
            neg_image: Negative angle image
            output_dir: Directory to save sum image
            filename: Output filename
            pixel_size_um: Pixel size for OME-TIFF metadata
            tile_config_source: Path to source TileConfiguration.txt to copy (optional)
            logger: Logger instance (optional)

        Returns:
            The sum image array
        """
        # Create output directory if it doesn't exist
        if not output_dir.exists():
            output_dir.mkdir(exist_ok=True)

            # Copy TileConfiguration.txt if source provided
            if tile_config_source and tile_config_source.exists():
                shutil.copy2(tile_config_source, output_dir / "TileConfiguration.txt")
                if logger:
                    logger.debug(f"Copied TileConfiguration.txt to {output_dir}")

        # Calculate sum
        output_path = output_dir / filename

        sum_img = TifWriterUtils.ppm_angle_sum(pos_image, neg_image)

        # Save float version for reference
        tf.imwrite(str(output_path)[:-4] + "_float.tif", sum_img.astype(np.float32))

        # Normalize to 8-bit RGB
        sum_img = sum_img * 255
        sum_img = np.clip(sum_img, 0, 255).astype(np.uint8)

        # Save as RGB (keep full color)
        TifWriterUtils.ome_writer(
            filename=str(output_path),
            pixel_size_um=pixel_size_um,
            data=sum_img,
        )

        if logger:
            logger.info(f"  Created sum image: {filename}")

        return sum_img

    @staticmethod
    def create_normalized_birefringence_tile(
        pos_image: np.ndarray,
        neg_image: np.ndarray,
        output_dir: pathlib.Path,
        filename: str,
        pixel_size_um: float,
        tile_config_source: Optional[pathlib.Path] = None,
        logger=None,
    ) -> np.ndarray:
        """
        Create a normalized birefringence image from positive and negative angle images.
        Uses the formula [I(+) - I(-)]/[I(+) + I(-)] to suppress H&E staining variations.

        Args:
            pos_image: Positive angle image
            neg_image: Negative angle image
            output_dir: Directory to save birefringence image
            filename: Output filename
            pixel_size_um: Pixel size for OME-TIFF metadata
            tile_config_source: Path to source TileConfiguration.txt to copy (optional)
            logger: Logger instance (optional)

        Returns:
            The normalized birefringence image array (uint16, 0-65535)
        """
        # Create output directory if it doesn't exist
        if not output_dir.exists():
            output_dir.mkdir(exist_ok=True)

            # Copy TileConfiguration.txt if source provided
            if tile_config_source and tile_config_source.exists():
                shutil.copy2(tile_config_source, output_dir / "TileConfiguration.txt")
                if logger:
                    logger.debug(f"Copied TileConfiguration.txt to {output_dir}")

        # Calculate normalized birefringence (absolute value for visualization)
        output_path = output_dir / filename

        norm_biref_img = TifWriterUtils.ppm_normalized_difference_abs(pos_image, neg_image)

        # Save as 16-bit single-channel image
        # Range: 0-65535 where 0 = no birefringence, 65535 = maximum
        TifWriterUtils.ome_writer(
            filename=str(output_path),
            pixel_size_um=pixel_size_um,
            data=norm_biref_img,
        )

        if logger:
            logger.info(f"  Created normalized birefringence: {filename} (16-bit, range: {norm_biref_img.min()}-{norm_biref_img.max()})")

        return norm_biref_img

    @staticmethod
    def apply_brightness_correction(image: np.ndarray, correction_factor: float) -> np.ndarray:
        """Apply brightness correction to an image."""
        return np.clip(image * correction_factor, 0, 255).astype(np.uint8)
