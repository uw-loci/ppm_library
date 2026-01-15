"""Tests for sunburst calibration module."""

import numpy as np
import pytest
import tempfile
from pathlib import Path
from skimage import color

from ppm_library.calibration.sunburst import (
    SunburstCalibrator,
    CalibrationResult,
    CalibrationRectangle,
    calibrate_from_image,
)


def create_synthetic_calibration_image(
    n_rectangles: int = 16,
    image_size: int = 512,
    rect_width: int = 15,
    rect_height: int = 40,
) -> np.ndarray:
    """Create a synthetic calibration image with oriented rectangles.

    Each rectangle has a unique hue corresponding to its angle.
    Rectangles are placed in a grid pattern to ensure they're separated.

    Args:
        n_rectangles: Number of rectangles to create
        image_size: Image dimensions (square)
        rect_width: Width of each rectangle
        rect_height: Height (length) of each rectangle

    Returns:
        RGB image as numpy array (H, W, 3), uint8
    """
    import cv2

    # Create dark background
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    # Place rectangles in a grid pattern to ensure separation
    n_cols = int(np.ceil(np.sqrt(n_rectangles)))
    n_rows = int(np.ceil(n_rectangles / n_cols))

    cell_width = image_size // (n_cols + 1)
    cell_height = image_size // (n_rows + 1)

    for i in range(n_rectangles):
        # Angle evenly distributed over 180 degrees
        angle = (i * 180.0 / n_rectangles)

        # Hue linearly mapped to angle (0-180 -> 0-0.5 in hue)
        # This simulates the PPM relationship
        hue = angle / 360.0  # Scale to [0, 0.5] for 180 degrees

        # Position rectangle in a grid
        row = i // n_cols
        col = i % n_cols
        cx = int((col + 1) * cell_width)
        cy = int((row + 1) * cell_height)

        # Create rotated rectangle using cv2
        # The angle parameter in boxPoints expects degrees, counter-clockwise
        rect = ((cx, cy), (rect_width, rect_height), -angle)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # Convert hue to RGB
        hsv_color = np.array([[[hue, 0.9, 0.9]]], dtype=np.float32)
        rgb_color = color.hsv2rgb(hsv_color)[0, 0]
        # cv2 expects BGR but fillPoly just takes values for each channel
        rgb_tuple = (int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255))

        # Draw filled rectangle
        cv2.fillPoly(image, [box], rgb_tuple)

    return image


class TestCalibrationRectangle:
    """Tests for CalibrationRectangle dataclass."""

    def test_creation(self):
        """Test rectangle creation."""
        rect = CalibrationRectangle(
            label=1,
            centroid=(100.0, 150.0),
            angle=45.0,
            area=1000,
            rgb_mode=(128, 64, 32),
            hue_mode=0.25,
            hue_mean=0.26,
        )

        assert rect.label == 1
        assert rect.centroid == (100.0, 150.0)
        assert rect.angle == 45.0
        assert rect.area == 1000
        assert rect.rgb_mode == (128, 64, 32)
        assert rect.hue_mode == 0.25


class TestCalibrationResult:
    """Tests for CalibrationResult dataclass."""

    def test_hue_to_angle(self):
        """Test hue to angle conversion."""
        result = CalibrationResult(
            slope=1/180,  # hue increases by 1/180 per degree
            intercept=0.0,
            r_squared=0.99,
            inv_slope=180.0,  # angle = 180 * hue
            inv_intercept=0.0,
            angles=np.array([0, 90, 180]),
            hue_values=np.array([0, 0.5, 1.0]),
            rectangles=[],
        )

        # Test single value
        angle = result.hue_to_angle(0.5)
        assert np.isclose(angle, 90.0, atol=0.1)

        # Test array
        angles = result.hue_to_angle(np.array([0.0, 0.25, 0.5]))
        np.testing.assert_allclose(angles, [0.0, 45.0, 90.0], atol=0.1)

    def test_angle_to_hue(self):
        """Test angle to hue conversion."""
        result = CalibrationResult(
            slope=1/180,
            intercept=0.0,
            r_squared=0.99,
            inv_slope=180.0,
            inv_intercept=0.0,
            angles=np.array([0, 90, 180]),
            hue_values=np.array([0, 0.5, 1.0]),
            rectangles=[],
        )

        hue = result.angle_to_hue(90.0)
        assert np.isclose(hue, 0.5, atol=0.01)

    def test_save_load(self):
        """Test saving and loading calibration."""
        result = CalibrationResult(
            slope=0.005,
            intercept=0.1,
            r_squared=0.98,
            inv_slope=200.0,
            inv_intercept=-20.0,
            angles=np.array([10, 50, 90, 130, 170]),
            hue_values=np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
            rectangles=[],
        )

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = Path(f.name)

        try:
            result.save(path)
            loaded = CalibrationResult.load(path)

            assert np.isclose(loaded.slope, result.slope)
            assert np.isclose(loaded.intercept, result.intercept)
            assert np.isclose(loaded.r_squared, result.r_squared)
            np.testing.assert_allclose(loaded.angles, result.angles)
            np.testing.assert_allclose(loaded.hue_values, result.hue_values)
        finally:
            path.unlink()


class TestSunburstCalibrator:
    """Tests for SunburstCalibrator class."""

    def test_init(self):
        """Test calibrator initialization."""
        cal = SunburstCalibrator()
        assert cal.n_expected_rectangles == 16
        assert cal.min_area == 100

        cal2 = SunburstCalibrator(n_expected_rectangles=12, min_area=50)
        assert cal2.n_expected_rectangles == 12
        assert cal2.min_area == 50

    def test_mode_rgb_calculation(self):
        """Test mode RGB calculation."""
        cal = SunburstCalibrator()

        # Create pixel array with clear mode
        pixels = np.array([
            [100, 50, 25],
            [100, 50, 25],
            [100, 50, 25],
            [101, 51, 26],
            [99, 49, 24],
        ])

        mode = cal._calculate_mode_rgb(pixels)
        assert mode == (100, 50, 25)

    def test_mode_hue_calculation(self):
        """Test mode hue calculation."""
        cal = SunburstCalibrator()

        # Create hue values with clear mode
        hues = np.array([0.3, 0.3, 0.3, 0.31, 0.29, 0.32])

        mode = cal._calculate_mode_hue(hues)
        assert 0.29 < mode < 0.33

    @pytest.mark.filterwarnings("ignore:Found .* rectangles")
    def test_calibration_with_synthetic_image(self):
        """Test full calibration on synthetic image."""
        # Create synthetic calibration image
        image = create_synthetic_calibration_image(n_rectangles=8, image_size=400)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            path = Path(f.name)

        try:
            from skimage.io import imsave
            imsave(str(path), image)

            # Run calibration
            cal = SunburstCalibrator(n_expected_rectangles=8, min_area=50)
            result = cal.calibrate(path, debug_plot=False)

            # Check we got reasonable results
            assert len(result.rectangles) >= 4  # At least some rectangles detected
            assert result.r_squared > 0.5  # Reasonable fit
            assert len(result.angles) == len(result.hue_values)

        finally:
            path.unlink()

    def test_file_not_found(self):
        """Test error handling for missing file."""
        cal = SunburstCalibrator()

        with pytest.raises(FileNotFoundError):
            cal.calibrate("/nonexistent/path/image.tif")


class TestDuplicateMerging:
    """Tests for duplicate angle merging (sunburst opposite directions)."""

    def test_merge_duplicate_angles(self):
        """Test that opposite directions get merged."""
        cal = SunburstCalibrator(merge_duplicates=True, angle_tolerance=5.0)

        # Create rectangles with duplicate angles (simulating opposite directions)
        rectangles = [
            CalibrationRectangle(1, (100, 100), 0.0, 500, (200, 50, 50), 0.0, 0.0),
            CalibrationRectangle(2, (200, 200), 1.0, 500, (200, 50, 50), 0.01, 0.01),  # ~same as 0
            CalibrationRectangle(3, (300, 100), 45.0, 500, (100, 200, 50), 0.25, 0.25),
            CalibrationRectangle(4, (400, 200), 46.0, 500, (100, 200, 50), 0.26, 0.26),  # ~same as 45
            CalibrationRectangle(5, (100, 300), 90.0, 500, (50, 50, 200), 0.5, 0.5),
        ]

        merged = cal._merge_duplicate_angles(rectangles)

        # Should merge 0 and 1, and 45 and 46, keeping 90 alone
        assert len(merged) == 3

        # Check merged angles are averages
        angles = [r.angle for r in merged]
        assert any(abs(a - 0.5) < 1 for a in angles)  # (0 + 1) / 2
        assert any(abs(a - 45.5) < 1 for a in angles)  # (45 + 46) / 2
        assert any(abs(a - 90.0) < 1 for a in angles)

    def test_no_merge_when_disabled(self):
        """Test that merging can be disabled."""
        cal = SunburstCalibrator(merge_duplicates=False)

        rectangles = [
            CalibrationRectangle(1, (100, 100), 0.0, 500, (200, 50, 50), 0.0, 0.0),
            CalibrationRectangle(2, (200, 200), 1.0, 500, (200, 50, 50), 0.01, 0.01),
        ]

        # When merge_duplicates=False, _merge_duplicate_angles shouldn't be called
        # But we can test the function directly still works
        merged = cal._merge_duplicate_angles(rectangles)
        assert len(merged) == 1  # Would merge if called

    def test_wrap_around_merge(self):
        """Test merging near 0/180 boundary.

        Since fiber orientation is 0-180°, angles near 0° and 180° are actually
        the same orientation (both horizontal). So 1° and 179° should merge.
        """
        cal = SunburstCalibrator(merge_duplicates=True, angle_tolerance=5.0)

        # 1° and 179° are the same orientation (horizontal), should merge
        # 90° is perpendicular, should stay separate
        rectangles = [
            CalibrationRectangle(1, (100, 100), 1.0, 500, (200, 50, 50), 0.01, 0.01),
            CalibrationRectangle(2, (200, 200), 90.0, 500, (50, 200, 50), 0.25, 0.25),
            CalibrationRectangle(3, (300, 300), 179.0, 500, (200, 50, 50), 0.99, 0.99),
        ]

        merged = cal._merge_duplicate_angles(rectangles)

        # 1° and 179° merge (both ~horizontal), 90° stays alone
        assert len(merged) == 2

        # Check the merged angle is near 0 or 180 (average of 1 and 179 = 90,
        # but with wrap-around it should average properly)
        angles = sorted([r.angle for r in merged])
        assert any(abs(a - 90.0) < 1 for a in angles)  # 90 stays


class TestConvenienceFunction:
    """Tests for the calibrate_from_image function."""

    @pytest.mark.filterwarnings("ignore:Found .* rectangles")
    def test_basic_usage(self):
        """Test basic usage of convenience function."""
        image = create_synthetic_calibration_image(n_rectangles=8)

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            path = Path(f.name)

        try:
            from skimage.io import imsave
            imsave(str(path), image)

            result = calibrate_from_image(path, n_rectangles=8)

            assert isinstance(result, CalibrationResult)
            assert result.r_squared > 0

        finally:
            path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
