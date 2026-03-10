"""
Unit tests for PPM imaging modules: ppm_image, hue_correction, and writer.

Tests cover PPMImage construction, AngleMap operations, hue correction utilities,
and TifWriterUtils image arithmetic functions.
"""

import numpy as np
import pytest
import tifffile

from ppm_library.imaging.ppm_image import PPMImage, AngleMap, _load_image
from ppm_library.imaging.hue_correction import (
    hue_shift,
    compute_hue_shift_from_reference,
    apply_gaussian_smoothing,
    apply_median_filter,
    preprocess_ppm_image,
)
from ppm_library.imaging.writer import TifWriterUtils


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rgb(height=64, width=64, r=128, g=64, b=64):
    """Create a uniform RGB uint8 image with the given channel values."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def _make_saturated_rgb(height=64, width=64):
    """Create an RGB image with high saturation and value (bright red).

    Ensures pixels pass the default PPMImage thresholds (sat > 0.2, val > 0.2).
    """
    return _make_rgb(height, width, r=220, g=30, b=30)


# ===========================================================================
# AngleMap tests
# ===========================================================================


class TestAngleMap:
    """Tests for the AngleMap dataclass."""

    def _make_angle_map(self, h=32, w=32):
        angles = np.full((h, w), 45.0)
        valid = np.ones((h, w), dtype=bool)
        hue = np.full((h, w), 0.25)
        sat = np.full((h, w), 0.8)
        val = np.full((h, w), 0.9)
        return AngleMap(angles=angles, valid_mask=valid, hue=hue,
                        saturation=sat, value=val)

    def test_shape_property(self):
        am = self._make_angle_map(20, 30)
        assert am.shape == (20, 30)

    def test_get_angles_in_roi(self):
        am = self._make_angle_map()
        mask = np.zeros((32, 32), dtype=bool)
        mask[0:5, 0:5] = True
        result = am.get_angles_in_roi(mask)
        assert result.shape == (25,)
        assert np.allclose(result, 45.0)

    def test_get_angles_in_roi_respects_valid_mask(self):
        am = self._make_angle_map()
        # Invalidate some pixels within the ROI
        am.valid_mask[0:3, 0:3] = False
        mask = np.zeros((32, 32), dtype=bool)
        mask[0:5, 0:5] = True
        result = am.get_angles_in_roi(mask)
        # 25 ROI pixels minus 9 invalid = 16 valid
        assert result.shape == (16,)

    def test_get_mean_angle_in_roi(self):
        am = self._make_angle_map()
        mask = np.ones((32, 32), dtype=bool)
        assert am.get_mean_angle_in_roi(mask) == pytest.approx(45.0)

    def test_get_mean_angle_empty_roi_returns_nan(self):
        am = self._make_angle_map()
        mask = np.zeros((32, 32), dtype=bool)
        result = am.get_mean_angle_in_roi(mask)
        assert np.isnan(result)

    def test_get_angle_histogram(self):
        am = self._make_angle_map()
        counts, edges = am.get_angle_histogram(bins=18)
        assert len(counts) == 18
        assert len(edges) == 19
        assert edges[0] == 0.0
        assert edges[-1] == 180.0
        assert counts.sum() == 32 * 32

    def test_get_angle_histogram_with_mask(self):
        am = self._make_angle_map()
        mask = np.zeros((32, 32), dtype=bool)
        mask[0:4, 0:4] = True
        counts, _ = am.get_angle_histogram(mask=mask, bins=18)
        assert counts.sum() == 16


# ===========================================================================
# PPMImage tests
# ===========================================================================


class TestPPMImage:
    """Tests for the PPMImage class."""

    def test_constructor_rejects_2d(self):
        gray = np.zeros((64, 64), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected 3D"):
            PPMImage(gray)

    def test_constructor_rejects_wrong_channels(self):
        four_ch = np.zeros((64, 64, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected 3 channels"):
            PPMImage(four_ch)

    def test_constructor_accepts_valid_rgb(self):
        img = _make_saturated_rgb()
        ppm = PPMImage(img)
        assert ppm.image is img

    def test_shape_property(self):
        img = _make_saturated_rgb(50, 80)
        ppm = PPMImage(img)
        assert ppm.shape == (50, 80, 3)

    def test_hue_property_shape(self):
        img = _make_saturated_rgb(30, 40)
        ppm = PPMImage(img)
        assert ppm.hue.shape == (30, 40)

    def test_saturation_property_range(self):
        img = _make_saturated_rgb()
        ppm = PPMImage(img)
        assert ppm.saturation.min() >= 0.0
        assert ppm.saturation.max() <= 1.0

    def test_value_property_range(self):
        img = _make_saturated_rgb()
        ppm = PPMImage(img)
        assert ppm.value.min() >= 0.0
        assert ppm.value.max() <= 1.0

    def test_valid_mask_with_saturated_image(self):
        img = _make_saturated_rgb()
        ppm = PPMImage(img)
        # Bright red should have high sat/val, all pixels valid
        assert ppm.valid_mask.all()

    def test_valid_mask_with_gray_image(self):
        # Gray pixels have zero saturation -- all should be invalid
        img = _make_rgb(r=128, g=128, b=128)
        ppm = PPMImage(img)
        assert not ppm.valid_mask.any()

    def test_load_from_tiff(self, tmp_path):
        """PPMImage.load reads a synthetic TIFF and returns a valid object."""
        img = _make_saturated_rgb()
        path = tmp_path / "sample.tif"
        tifffile.imwrite(str(path), img)

        ppm = PPMImage.load(str(path))
        assert ppm.shape == img.shape
        np.testing.assert_array_equal(ppm.image, img)


# ===========================================================================
# _load_image tests
# ===========================================================================


class TestLoadImage:
    """Tests for the _load_image helper function."""

    def test_channels_first_converted(self, tmp_path):
        """Channels-first (3, H, W) is transposed to (H, W, 3)."""
        # Create channels-first array and save via tifffile
        chw = np.random.randint(0, 255, (3, 40, 50), dtype=np.uint8)
        path = tmp_path / "chw.tif"
        tifffile.imwrite(str(path), chw)

        loaded = _load_image(path)
        assert loaded.shape == (40, 50, 3)
        assert loaded.dtype == np.uint8

    def test_rgba_converted_to_rgb(self, tmp_path):
        rgba = np.random.randint(0, 255, (30, 40, 4), dtype=np.uint8)
        path = tmp_path / "rgba.tif"
        tifffile.imwrite(str(path), rgba)

        loaded = _load_image(path)
        assert loaded.shape == (30, 40, 3)

    def test_grayscale_converted_to_rgb(self, tmp_path):
        gray = np.full((20, 25), 100, dtype=np.uint8)
        path = tmp_path / "gray.tif"
        tifffile.imwrite(str(path), gray)

        loaded = _load_image(path)
        assert loaded.shape == (20, 25, 3)
        # All channels should equal the original grayscale value
        np.testing.assert_array_equal(loaded[:, :, 0], 100)
        np.testing.assert_array_equal(loaded[:, :, 1], 100)
        np.testing.assert_array_equal(loaded[:, :, 2], 100)

    def test_uint8_passthrough(self, tmp_path):
        img = _make_saturated_rgb(10, 10)
        path = tmp_path / "u8.tif"
        tifffile.imwrite(str(path), img)

        loaded = _load_image(path)
        assert loaded.dtype == np.uint8
        np.testing.assert_array_equal(loaded, img)

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _load_image(tmp_path / "nonexistent.tif")


# ===========================================================================
# hue_correction tests
# ===========================================================================


class TestHueShift:
    """Tests for hue_shift."""

    def test_shift_changes_hue(self):
        img = _make_saturated_rgb()
        shifted = hue_shift(img, 30.0, apply_median_filter=False)
        # The output should differ from input
        assert not np.array_equal(img, shifted)
        assert shifted.dtype == np.uint8
        assert shifted.shape == img.shape

    def test_zero_shift_preserves_image(self):
        img = _make_saturated_rgb()
        shifted = hue_shift(img, 0.0, apply_median_filter=False)
        # A 0-degree shift should leave the image essentially unchanged
        # (allow small rounding differences from float conversion)
        np.testing.assert_allclose(
            shifted.astype(np.float64), img.astype(np.float64), atol=2
        )

    def test_rejects_non_rgb_input(self):
        gray = np.zeros((32, 32), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected RGB"):
            hue_shift(gray, 10.0)

    def test_full_360_cycle_returns_original(self):
        """Shifting by 180 hue-degrees (full PPM cycle) returns near-original."""
        img = _make_saturated_rgb()
        shifted = hue_shift(img, 180.0, apply_median_filter=False)
        # 180 degrees in PPM hue space wraps hue by 1.0 (full circle)
        np.testing.assert_allclose(
            shifted.astype(np.float64), img.astype(np.float64), atol=2
        )


class TestComputeHueShiftFromReference:
    """Tests for compute_hue_shift_from_reference."""

    def test_correct_shift_calculation(self):
        # Use a known bright, saturated image so pixels pass thresholds
        img = _make_saturated_rgb(64, 64)
        ppm = PPMImage(img)
        measured_hue = np.mean(ppm.hue[ppm.valid_mask])
        # If we claim the reference angle is the *actual* measured hue * 180,
        # the shift should be ~0
        ref_angle = measured_hue * 180.0
        shift_deg, mhue = compute_hue_shift_from_reference(img, ref_angle)
        assert abs(shift_deg) < 2.0
        assert abs(mhue - measured_hue) < 0.01

    def test_raises_on_empty_valid_pixels(self):
        # A completely black image has value ~0 -> no valid pixels
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="No valid pixels"):
            compute_hue_shift_from_reference(img, 45.0)


class TestApplyGaussianSmoothing:
    """Tests for apply_gaussian_smoothing."""

    def test_output_shape_matches(self):
        img = _make_saturated_rgb(50, 60)
        out = apply_gaussian_smoothing(img, sigma=2.0)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_reduces_noise(self):
        # Create noisy image
        rng = np.random.default_rng(42)
        base = _make_saturated_rgb(64, 64).astype(np.float64)
        noise = rng.normal(0, 25, base.shape)
        noisy = np.clip(base + noise, 0, 255).astype(np.uint8)

        smoothed = apply_gaussian_smoothing(noisy, sigma=3.0)

        # Standard deviation of smoothed should be less than noisy
        assert smoothed.astype(np.float64).std() < noisy.astype(np.float64).std()


class TestApplyMedianFilter:
    """Tests for apply_median_filter."""

    def test_output_shape_matches(self):
        img = _make_saturated_rgb(40, 50)
        out = apply_median_filter(img, size=3)
        assert out.shape == img.shape
        assert out.dtype == img.dtype

    def test_removes_salt_and_pepper(self):
        img = _make_rgb(64, 64, r=128, g=128, b=128)
        # Inject salt-and-pepper noise
        rng = np.random.default_rng(7)
        noisy = img.copy()
        salt = rng.random((64, 64)) < 0.05
        pepper = rng.random((64, 64)) < 0.05
        noisy[salt] = 255
        noisy[pepper] = 0

        filtered = apply_median_filter(noisy, size=3)

        # After filtering, values should be closer to original 128
        orig_diff = np.abs(noisy.astype(np.float64) - 128).mean()
        filt_diff = np.abs(filtered.astype(np.float64) - 128).mean()
        assert filt_diff < orig_diff


class TestPreprocessPpmImage:
    """Tests for preprocess_ppm_image."""

    def test_chains_gaussian_then_median(self):
        img = _make_saturated_rgb(48, 48)
        result = preprocess_ppm_image(img, gaussian_sigma=1.5, median_size=3)
        assert result.shape == img.shape
        assert result.dtype == np.uint8


# ===========================================================================
# writer / TifWriterUtils tests
# ===========================================================================


class TestPpmAngleDifference:
    """Tests for TifWriterUtils.ppm_angle_difference."""

    def test_known_values(self):
        img1 = _make_rgb(4, 4, r=200, g=100, b=50)
        img2 = _make_rgb(4, 4, r=100, g=100, b=100)
        result = TifWriterUtils.ppm_angle_difference(img1, img2)
        # |200-100| + |100-100| + |50-100| = 100 + 0 + 50 = 150
        expected = 150
        assert result.shape == (4, 4)
        assert result.dtype == np.uint16
        np.testing.assert_array_equal(result, expected)

    def test_identical_images_zero(self):
        img = _make_rgb(8, 8, r=120, g=80, b=40)
        result = TifWriterUtils.ppm_angle_difference(img, img)
        np.testing.assert_array_equal(result, 0)

    def test_max_range(self):
        white = _make_rgb(4, 4, r=255, g=255, b=255)
        black = _make_rgb(4, 4, r=0, g=0, b=0)
        result = TifWriterUtils.ppm_angle_difference(white, black)
        np.testing.assert_array_equal(result, 765)


class TestPpmNormalizedDifference:
    """Tests for TifWriterUtils.ppm_normalized_difference."""

    def test_output_dtype_uint16(self):
        img = _make_rgb(8, 8, r=100, g=100, b=100)
        result = TifWriterUtils.ppm_normalized_difference(img, img)
        assert result.dtype == np.uint16

    def test_identical_images_midpoint(self):
        img = _make_rgb(8, 8, r=100, g=100, b=100)
        result = TifWriterUtils.ppm_normalized_difference(img, img)
        # (diff=0) -> normalized=0 -> scaled ~ 32768
        mid = result[0, 0]
        assert abs(int(mid) - 32768) < 2

    def test_positive_negative_asymmetry(self):
        bright = _make_rgb(8, 8, r=200, g=200, b=200)
        dim = _make_rgb(8, 8, r=50, g=50, b=50)
        pos = TifWriterUtils.ppm_normalized_difference(bright, dim)
        neg = TifWriterUtils.ppm_normalized_difference(dim, bright)
        # Bright-dim should be above midpoint, dim-bright below
        assert pos[0, 0] > 32768
        assert neg[0, 0] < 32768


class TestPpmNormalizedDifferenceAbs:
    """Tests for TifWriterUtils.ppm_normalized_difference_abs."""

    def test_zero_for_identical_images(self):
        img = _make_rgb(8, 8, r=100, g=100, b=100)
        result = TifWriterUtils.ppm_normalized_difference_abs(img, img)
        assert result.dtype == np.uint16
        np.testing.assert_array_equal(result, 0)

    def test_symmetric(self):
        a = _make_rgb(8, 8, r=200, g=200, b=200)
        b = _make_rgb(8, 8, r=50, g=50, b=50)
        ab = TifWriterUtils.ppm_normalized_difference_abs(a, b)
        ba = TifWriterUtils.ppm_normalized_difference_abs(b, a)
        np.testing.assert_array_equal(ab, ba)


class TestPpmAngleSum:
    """Tests for TifWriterUtils.ppm_angle_sum."""

    def test_output_shape(self):
        a = _make_rgb(16, 16, r=100, g=50, b=200)
        b = _make_rgb(16, 16, r=50, g=100, b=50)
        result = TifWriterUtils.ppm_angle_sum(a, b)
        assert result.shape == (16, 16, 3)

    def test_value_range(self):
        a = _make_rgb(8, 8, r=200, g=200, b=200)
        b = _make_rgb(8, 8, r=200, g=200, b=200)
        result = TifWriterUtils.ppm_angle_sum(a, b)
        # Average of two identical images -> same as original (in 0-1 float)
        assert result.max() <= 1.0
        assert result.min() >= 0.0


class TestSubtractionImage:
    """Tests for TifWriterUtils.subtraction_image."""

    def test_known_difference(self):
        a = _make_rgb(4, 4, r=200, g=150, b=100)
        b = _make_rgb(4, 4, r=100, g=100, b=100)
        result = TifWriterUtils.subtraction_image(a, b)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result[:, :, 0], 100.0)
        np.testing.assert_allclose(result[:, :, 1], 50.0)
        np.testing.assert_allclose(result[:, :, 2], 0.0)

    def test_negative_difference(self):
        a = _make_rgb(4, 4, r=50, g=50, b=50)
        b = _make_rgb(4, 4, r=100, g=100, b=100)
        result = TifWriterUtils.subtraction_image(a, b)
        assert (result < 0).all()


class TestApplyBrightnessCorrection:
    """Tests for TifWriterUtils.apply_brightness_correction."""

    def test_scales_correctly(self):
        img = _make_rgb(8, 8, r=100, g=100, b=100)
        result = TifWriterUtils.apply_brightness_correction(img, 2.0)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result[:, :, 0], 200)
        np.testing.assert_array_equal(result[:, :, 1], 200)
        np.testing.assert_array_equal(result[:, :, 2], 200)

    def test_clips_at_255(self):
        img = _make_rgb(8, 8, r=200, g=200, b=200)
        result = TifWriterUtils.apply_brightness_correction(img, 2.0)
        assert result.max() == 255

    def test_factor_less_than_one_dims(self):
        img = _make_rgb(8, 8, r=200, g=200, b=200)
        result = TifWriterUtils.apply_brightness_correction(img, 0.5)
        np.testing.assert_array_equal(result[:, :, 0], 100)
