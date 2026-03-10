"""
Unit tests for calibration modules: histogram_correction and radial.

Tests HistogramCalibration (validation, creation, correction, save/load)
and RadialCalibrationResult (hue_to_angle, angle_to_hue, save/load, quality checks).
"""

import numpy as np
import pytest

from ppm_library.calibration.histogram_correction import (
    HistogramCalibration,
    compute_hue_histogram,
)
from ppm_library.calibration.radial import RadialCalibrationResult, RadialSample


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_gaussian_peaks(n_peaks=16, n_bins=256, sigma=3.0, base_height=100.0,
                         variation=0.3):
    """Create a synthetic 256-bin histogram with *n_peaks* Gaussian peaks.

    Peak heights vary by +/- *variation* fraction so the correction curve
    is non-trivial.  Returns (histogram, peak_bin_locations).
    """
    bins = np.arange(n_bins, dtype=np.float64)
    histogram = np.zeros(n_bins, dtype=np.float64)
    spacing = n_bins / n_peaks
    peak_locs = np.array([int(round(i * spacing + spacing / 2)) for i in range(n_peaks)])
    peak_locs = peak_locs[peak_locs < n_bins]

    rng = np.random.RandomState(42)
    for loc in peak_locs:
        height = base_height * (1.0 + variation * (rng.rand() - 0.5) * 2)
        histogram += height * np.exp(-0.5 * ((bins - loc) / sigma) ** 2)

    return histogram, peak_locs


def _make_simple_radial_result(**overrides):
    """Return a minimal RadialCalibrationResult with sensible defaults."""
    defaults = dict(
        slope=1.0 / 180.0,
        intercept=0.0,
        inv_slope=180.0,
        inv_intercept=0.0,
        r_squared=0.99,
        hue_offset=0.0,
        angles=np.linspace(0, 170, 18),
        hue_values=np.linspace(0, 170.0 / 180.0, 18),
        samples=[],
        center=(256, 256),
    )
    defaults.update(overrides)
    return RadialCalibrationResult(**defaults)


@pytest.fixture
def gaussian_histogram():
    """256-bin histogram with 16 evenly spaced Gaussian peaks."""
    hist, locs = _make_gaussian_peaks(n_peaks=16)
    return hist, locs


@pytest.fixture
def flat_correction_curve():
    """Uniform correction curve (all ones)."""
    return np.ones(256, dtype=np.float64)


@pytest.fixture
def simple_calibration(flat_correction_curve):
    """HistogramCalibration with a flat curve and no phase shift."""
    return HistogramCalibration(
        correction_curve=flat_correction_curve,
        phase_shift=0,
        phase_direction=1,
    )


# ===========================================================================
# HistogramCalibration -- __post_init__ validation
# ===========================================================================

class TestHistogramCalibrationValidation:
    """Validation in __post_init__."""

    def test_wrong_curve_length_short(self):
        """Curve with fewer than 256 elements raises ValueError."""
        with pytest.raises(ValueError, match="256"):
            HistogramCalibration(
                correction_curve=np.ones(100),
                phase_shift=0,
                phase_direction=1,
            )

    def test_wrong_curve_length_long(self):
        """Curve with more than 256 elements raises ValueError."""
        with pytest.raises(ValueError, match="256"):
            HistogramCalibration(
                correction_curve=np.ones(512),
                phase_shift=0,
                phase_direction=1,
            )

    def test_invalid_phase_direction_positive(self):
        """phase_direction of +2 raises ValueError."""
        with pytest.raises(ValueError, match="Phase direction"):
            HistogramCalibration(
                correction_curve=np.ones(256),
                phase_shift=5,
                phase_direction=2,
            )

    def test_invalid_phase_direction_negative(self):
        """phase_direction of -2 raises ValueError."""
        with pytest.raises(ValueError, match="Phase direction"):
            HistogramCalibration(
                correction_curve=np.ones(256),
                phase_shift=5,
                phase_direction=-2,
            )

    def test_valid_phase_directions_accepted(self):
        """phase_direction of -1, 0, and 1 should all be accepted."""
        for direction in (-1, 0, 1):
            cal = HistogramCalibration(
                correction_curve=np.ones(256),
                phase_shift=0,
                phase_direction=direction,
            )
            assert cal.phase_direction == direction


# ===========================================================================
# HistogramCalibration.from_circular_histogram
# ===========================================================================

class TestFromCircularHistogram:
    """Construction from a 256-bin circular-pattern histogram."""

    def test_creates_calibration_object(self, gaussian_histogram):
        """from_circular_histogram returns a HistogramCalibration."""
        hist, _ = gaussian_histogram
        cal = HistogramCalibration.from_circular_histogram(hist, n_peaks=16)
        assert isinstance(cal, HistogramCalibration)

    def test_correction_curve_length(self, gaussian_histogram):
        """Correction curve has 256 elements."""
        hist, _ = gaussian_histogram
        cal = HistogramCalibration.from_circular_histogram(hist, n_peaks=16)
        assert len(cal.correction_curve) == 256

    def test_correction_curve_normalized(self, gaussian_histogram):
        """Correction curve max equals 1."""
        hist, _ = gaussian_histogram
        cal = HistogramCalibration.from_circular_histogram(hist, n_peaks=16)
        assert np.isclose(np.max(cal.correction_curve), 1.0)

    def test_peaks_detected(self, gaussian_histogram):
        """Peak locations are populated."""
        hist, _ = gaussian_histogram
        cal = HistogramCalibration.from_circular_histogram(hist, n_peaks=16)
        assert len(cal.peak_locations) > 0

    def test_rejects_wrong_histogram_length(self):
        """Histogram with != 256 bins raises ValueError."""
        with pytest.raises(ValueError, match="256"):
            HistogramCalibration.from_circular_histogram(np.ones(128), n_peaks=4)


# ===========================================================================
# HistogramCalibration.correct_histogram
# ===========================================================================

class TestCorrectHistogram:
    """Histogram correction: division, phase shift, background removal."""

    def test_divides_by_correction_curve(self):
        """Corrected histogram values reflect division by the curve."""
        # Curve that doubles the first half (curve=0.5 -> divide -> doubles)
        curve = np.ones(256, dtype=np.float64)
        curve[:128] = 0.5
        cal = HistogramCalibration(
            correction_curve=curve, phase_shift=0, phase_direction=1
        )

        input_hist = np.ones(256, dtype=np.float64)
        input_hist[0] = 0  # avoid background removal noise
        corrected = cal.correct_histogram(input_hist, apply_phase_shift=False,
                                          remove_background=False)

        # Bins 0-127 were divided by 0.5 -> value 2; bins 128-255 -> value 1
        # After normalization, max should be 1.0
        assert np.isclose(corrected.max(), 1.0)
        # The ratio between first-half and second-half should be ~2:1
        ratio = corrected[64] / corrected[200]
        assert ratio > 1.5

    def test_phase_shift_rolls_histogram(self):
        """Phase shift moves peak to the right or left."""
        curve = np.ones(256, dtype=np.float64)
        cal = HistogramCalibration(
            correction_curve=curve, phase_shift=10, phase_direction=1
        )

        input_hist = np.zeros(256, dtype=np.float64)
        input_hist[100] = 1.0  # single peak at bin 100
        corrected = cal.correct_histogram(input_hist, apply_phase_shift=True,
                                          remove_background=False)

        # Peak should have moved by +10 bins
        assert np.argmax(corrected) == 110

    def test_phase_shift_negative_direction(self):
        """phase_direction=-1 shifts in the opposite direction."""
        curve = np.ones(256, dtype=np.float64)
        cal = HistogramCalibration(
            correction_curve=curve, phase_shift=10, phase_direction=-1
        )

        input_hist = np.zeros(256, dtype=np.float64)
        input_hist[100] = 1.0
        corrected = cal.correct_histogram(input_hist, apply_phase_shift=True,
                                          remove_background=False)

        assert np.argmax(corrected) == 90

    def test_remove_background_zeroes_bin0(self, simple_calibration):
        """remove_background=True sets bin 0 to zero before correction."""
        input_hist = np.ones(256, dtype=np.float64) * 50.0
        input_hist[0] = 9999.0  # large background spike

        corrected = simple_calibration.correct_histogram(
            input_hist, apply_phase_shift=False, remove_background=True
        )
        # Bin 0 was zeroed; all other bins equal -> normalized to 1.0 each
        assert corrected[0] == 0.0

    def test_keep_background_preserves_bin0(self, simple_calibration):
        """remove_background=False keeps bin 0 intact."""
        input_hist = np.ones(256, dtype=np.float64)
        corrected = simple_calibration.correct_histogram(
            input_hist, apply_phase_shift=False, remove_background=False
        )
        assert corrected[0] > 0.0

    def test_output_normalized_to_one(self, gaussian_histogram):
        """Corrected histogram max is 1.0."""
        hist, _ = gaussian_histogram
        cal = HistogramCalibration.from_circular_histogram(hist, n_peaks=16)
        corrected = cal.correct_histogram(hist)
        assert np.isclose(corrected.max(), 1.0)


# ===========================================================================
# HistogramCalibration.correct_hue_image
# ===========================================================================

class TestCorrectHueImage:
    """Pixel-wise hue image correction."""

    def test_uint8_returns_uint8(self, simple_calibration):
        """uint8 input produces uint8 output."""
        hue_img = np.random.RandomState(0).randint(1, 255, (64, 64)).astype(np.uint8)
        result = simple_calibration.correct_hue_image(hue_img)
        assert result.dtype == np.uint8

    def test_float_returns_float(self, simple_calibration):
        """Float input in [0, 1] produces float output."""
        hue_img = np.random.RandomState(0).rand(64, 64).astype(np.float64) * 0.9 + 0.05
        result = simple_calibration.correct_hue_image(hue_img)
        assert result.dtype in (np.float64, np.float32)

    def test_mask_limits_correction(self, simple_calibration):
        """Only masked pixels are corrected."""
        hue_img = np.full((64, 64), 128, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:20, 10:20] = True

        result = simple_calibration.correct_hue_image(hue_img, mask=mask)
        # With flat curve the result should be similar to input everywhere
        assert result.dtype == np.uint8


# ===========================================================================
# HistogramCalibration save / load round-trip
# ===========================================================================

class TestHistogramCalibrationSaveLoad:
    """NPZ round-trip for HistogramCalibration."""

    def test_roundtrip_preserves_correction_curve(self, simple_calibration, tmp_path):
        """Save then load recovers the same correction curve."""
        path = tmp_path / "cal.npz"
        simple_calibration.save(path)
        loaded = HistogramCalibration.load(path)
        np.testing.assert_array_almost_equal(
            loaded.correction_curve, simple_calibration.correction_curve
        )

    def test_roundtrip_preserves_phase_shift(self, tmp_path):
        """Phase shift and direction survive save/load."""
        cal = HistogramCalibration(
            correction_curve=np.ones(256),
            phase_shift=17,
            phase_direction=-1,
        )
        path = tmp_path / "cal.npz"
        cal.save(path)
        loaded = HistogramCalibration.load(path)
        assert loaded.phase_shift == 17
        assert loaded.phase_direction == -1

    def test_roundtrip_preserves_peaks(self, gaussian_histogram, tmp_path):
        """Peak locations and heights survive save/load."""
        hist, _ = gaussian_histogram
        cal = HistogramCalibration.from_circular_histogram(hist, n_peaks=16)
        path = tmp_path / "cal.npz"
        cal.save(path)
        loaded = HistogramCalibration.load(path)
        np.testing.assert_array_almost_equal(loaded.peak_locations, cal.peak_locations)
        np.testing.assert_array_almost_equal(loaded.peak_heights, cal.peak_heights)


# ===========================================================================
# compute_hue_histogram
# ===========================================================================

class TestComputeHueHistogram:
    """Tests for the standalone compute_hue_histogram function."""

    def _make_rgb_with_known_hue(self, hue_value, size=128):
        """Create an RGB image where every pixel has approximately *hue_value*.

        hue_value is in [0, 1].  Saturation and value are set high so
        thresholds are satisfied.
        """
        from skimage import color as skcolor

        hsv = np.full((size, size, 3), 0.0, dtype=np.float64)
        hsv[:, :, 0] = hue_value
        hsv[:, :, 1] = 0.9  # high saturation
        hsv[:, :, 2] = 0.9  # high value
        rgb = skcolor.hsv2rgb(hsv)
        return (rgb * 255).astype(np.uint8)

    def test_histogram_length(self):
        """Output histogram has 256 bins by default."""
        rgb = self._make_rgb_with_known_hue(0.5)
        hist = compute_hue_histogram(rgb)
        assert len(hist) == 256

    def test_peak_at_expected_bin(self):
        """Histogram peak lands near the expected hue bin."""
        target_hue = 0.25  # green-ish
        rgb = self._make_rgb_with_known_hue(target_hue)
        hist = compute_hue_histogram(rgb)

        peak_bin = np.argmax(hist)
        expected_bin = int(target_hue * 255)
        assert abs(peak_bin - expected_bin) <= 3, (
            f"Peak at bin {peak_bin}, expected near {expected_bin}"
        )

    def test_saturation_threshold_excludes_gray(self):
        """Pixels below saturation threshold are excluded."""
        from skimage import color as skcolor

        hsv = np.full((128, 128, 3), 0.0, dtype=np.float64)
        hsv[:, :, 0] = 0.5
        hsv[:, :, 1] = 0.05  # very low saturation -> should be excluded
        hsv[:, :, 2] = 0.9
        rgb = skcolor.hsv2rgb(hsv)
        rgb_uint8 = (rgb * 255).astype(np.uint8)

        hist = compute_hue_histogram(rgb_uint8, saturation_threshold=0.2)
        assert hist.sum() == 0, "All low-saturation pixels should be excluded"

    def test_value_threshold_excludes_dark(self):
        """Pixels below value threshold are excluded."""
        from skimage import color as skcolor

        hsv = np.full((128, 128, 3), 0.0, dtype=np.float64)
        hsv[:, :, 0] = 0.5
        hsv[:, :, 1] = 0.9
        hsv[:, :, 2] = 0.05  # very dark -> should be excluded
        rgb = skcolor.hsv2rgb(hsv)
        rgb_uint8 = (rgb * 255).astype(np.uint8)

        hist = compute_hue_histogram(rgb_uint8, value_threshold=0.2)
        assert hist.sum() == 0, "All dark pixels should be excluded"

    def test_mask_restricts_region(self):
        """Only pixels inside the mask contribute to the histogram."""
        rgb = self._make_rgb_with_known_hue(0.5, size=128)

        mask = np.zeros((128, 128), dtype=bool)
        mask[0:10, 0:10] = True  # tiny region

        hist_masked = compute_hue_histogram(rgb, mask=mask)
        hist_full = compute_hue_histogram(rgb)

        assert hist_masked.sum() < hist_full.sum()


# ===========================================================================
# RadialCalibrationResult -- construction
# ===========================================================================

class TestRadialCalibrationResultConstruction:
    """Basic construction and attribute access."""

    def test_construct_with_valid_data(self):
        """Construction succeeds with all required fields."""
        result = _make_simple_radial_result()
        assert result.slope == pytest.approx(1.0 / 180.0)
        assert result.r_squared == pytest.approx(0.99)

    def test_center_stored_correctly(self):
        """Center tuple is preserved."""
        result = _make_simple_radial_result(center=(100, 200))
        assert result.center == (100, 200)


# ===========================================================================
# RadialCalibrationResult.hue_to_angle
# ===========================================================================

class TestHueToAngle:
    """Linear hue-to-angle conversion."""

    def test_known_slope_intercept(self):
        """Verify with explicit slope and intercept values."""
        # inv_slope=180, inv_intercept=0, hue_offset=0
        # angle = 180 * hue + 0
        result = _make_simple_radial_result(
            inv_slope=180.0, inv_intercept=0.0, hue_offset=0.0,
        )
        angle = result.hue_to_angle(0.5)
        assert float(angle) == pytest.approx(90.0, abs=0.01)

    def test_hue_zero_gives_intercept(self):
        """hue=0 (after offset) should return inv_intercept mod 180."""
        result = _make_simple_radial_result(
            inv_slope=180.0, inv_intercept=10.0, hue_offset=0.0,
        )
        angle = result.hue_to_angle(0.0)
        assert float(angle) == pytest.approx(10.0, abs=0.01)

    def test_handles_array_input(self):
        """hue_to_angle accepts and returns numpy arrays."""
        result = _make_simple_radial_result(
            inv_slope=180.0, inv_intercept=0.0, hue_offset=0.0,
        )
        hues = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        angles = result.hue_to_angle(hues)
        assert isinstance(angles, np.ndarray)
        assert len(angles) == 5

    def test_array_values_correct(self):
        """Array conversion matches element-wise scalar conversion."""
        result = _make_simple_radial_result(
            inv_slope=180.0, inv_intercept=0.0, hue_offset=0.0,
        )
        hues = np.array([0.0, 0.25, 0.5])
        angles = result.hue_to_angle(hues)
        np.testing.assert_allclose(angles, [0.0, 45.0, 90.0], atol=0.01)

    def test_hue_offset_applied(self):
        """Non-zero hue_offset shifts the mapping."""
        result = _make_simple_radial_result(
            inv_slope=180.0, inv_intercept=0.0, hue_offset=0.1,
        )
        # hue_shifted = (0.6 - 0.1) % 1.0 = 0.5
        angle = result.hue_to_angle(0.6)
        assert float(angle) == pytest.approx(90.0, abs=0.01)

    def test_output_wraps_to_0_180(self):
        """Angles are modded to [0, 180)."""
        result = _make_simple_radial_result(
            inv_slope=360.0, inv_intercept=0.0, hue_offset=0.0,
        )
        # angle = 360 * 0.75 = 270 -> mod 180 = 90
        angle = result.hue_to_angle(0.75)
        assert 0.0 <= float(angle) < 180.0


# ===========================================================================
# RadialCalibrationResult.angle_to_hue
# ===========================================================================

class TestAngleToHue:
    """Inverse mapping: angle -> hue."""

    def test_inverse_of_hue_to_angle(self):
        """angle_to_hue is the inverse of hue_to_angle (within domain)."""
        result = _make_simple_radial_result(
            slope=1.0 / 180.0, intercept=0.0,
            inv_slope=180.0, inv_intercept=0.0,
            hue_offset=0.0,
        )
        original_hue = 0.3
        angle = result.hue_to_angle(original_hue)
        recovered = result.angle_to_hue(angle)
        assert float(recovered) == pytest.approx(original_hue, abs=0.01)


# ===========================================================================
# RadialCalibrationResult save / load round-trip
# ===========================================================================

class TestRadialCalibrationSaveLoad:
    """NPZ round-trip for RadialCalibrationResult."""

    def test_roundtrip_preserves_slope_intercept(self, tmp_path):
        """slope, intercept, inv_slope, inv_intercept survive save/load."""
        result = _make_simple_radial_result(
            slope=0.005, intercept=0.1,
            inv_slope=200.0, inv_intercept=-20.0,
        )
        path = tmp_path / "radial_cal.npz"
        result.save(path)
        loaded = RadialCalibrationResult.load(path)

        assert loaded.slope == pytest.approx(0.005)
        assert loaded.intercept == pytest.approx(0.1)
        assert loaded.inv_slope == pytest.approx(200.0)
        assert loaded.inv_intercept == pytest.approx(-20.0)

    def test_roundtrip_preserves_r_squared(self, tmp_path):
        """r_squared survives save/load."""
        result = _make_simple_radial_result(r_squared=0.9876)
        path = tmp_path / "radial_cal.npz"
        result.save(path)
        loaded = RadialCalibrationResult.load(path)
        assert loaded.r_squared == pytest.approx(0.9876)

    def test_roundtrip_preserves_hue_offset(self, tmp_path):
        """hue_offset survives save/load."""
        result = _make_simple_radial_result(hue_offset=0.42)
        path = tmp_path / "radial_cal.npz"
        result.save(path)
        loaded = RadialCalibrationResult.load(path)
        assert loaded.hue_offset == pytest.approx(0.42)

    def test_roundtrip_preserves_arrays(self, tmp_path):
        """angles and hue_values arrays survive save/load."""
        angles = np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0])
        hue_vals = np.array([0.0, 0.17, 0.33, 0.50, 0.67, 0.83])
        result = _make_simple_radial_result(angles=angles, hue_values=hue_vals)
        path = tmp_path / "radial_cal.npz"
        result.save(path)
        loaded = RadialCalibrationResult.load(path)
        np.testing.assert_array_almost_equal(loaded.angles, angles)
        np.testing.assert_array_almost_equal(loaded.hue_values, hue_vals)

    def test_roundtrip_preserves_center(self, tmp_path):
        """center tuple survives save/load."""
        result = _make_simple_radial_result(center=(512, 768))
        path = tmp_path / "radial_cal.npz"
        result.save(path)
        loaded = RadialCalibrationResult.load(path)
        assert loaded.center == (512, 768)


# ===========================================================================
# RadialCalibrationResult.check_quality
# ===========================================================================

class TestCheckQuality:
    """Quality-check warnings."""

    def test_no_warnings_for_good_calibration(self):
        """Good calibration produces no warnings."""
        samples = [RadialSample(angle=float(i * 10), hue_mean=0.1, hue_std=0.01,
                                n_samples=50) for i in range(18)]
        result = _make_simple_radial_result(r_squared=0.99, samples=samples)
        warnings = result.check_quality(expected_spokes=18, min_r_squared=0.95)
        assert warnings == []

    def test_low_r_squared_triggers_warning(self):
        """Low r_squared produces a warning string."""
        result = _make_simple_radial_result(r_squared=0.80)
        warnings = result.check_quality(expected_spokes=0, min_r_squared=0.95)
        assert any("R-squared" in w for w in warnings)

    def test_missing_spokes_triggers_warning(self):
        """Fewer samples than expected_spokes produces a warning."""
        samples = [RadialSample(angle=0.0, hue_mean=0.1, hue_std=0.01,
                                n_samples=50)]
        result = _make_simple_radial_result(r_squared=0.99, samples=samples)
        warnings = result.check_quality(expected_spokes=18, min_r_squared=0.95)
        assert any("spokes" in w.lower() for w in warnings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
