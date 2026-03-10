"""
Tests for ppm_library.analysis.region_analysis and surface_analysis.

Covers angle computation, masking, histograms, circular statistics,
boundary contour extraction, border zones, perpendicularity scoring,
and helper functions.

All test data is synthetic numpy arrays -- no calibration files needed.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Mock calibration
# ---------------------------------------------------------------------------

class MockCalibration:
    """Mock calibration that maps hue linearly to angle.

    angle = hue * 180.0 (hue in [0,1] -> angle in [0,180))
    """

    def hue_to_angle(self, hue_values):
        return np.asarray(hue_values, dtype=np.float64) * 180.0


@pytest.fixture
def mock_calibration():
    """Return a MockCalibration instance."""
    return MockCalibration()


@pytest.fixture
def _patch_load_calibration(mock_calibration):
    """Patch load_calibration so it returns our mock regardless of input."""
    with patch(
        "ppm_library.analysis.region_analysis.load_calibration",
        return_value=mock_calibration,
    ):
        yield


# ---------------------------------------------------------------------------
# Helpers for building synthetic RGB images
# ---------------------------------------------------------------------------

def _make_saturated_rgb(h, s, v, shape=(16, 16)):
    """Build an RGB uint8 image where every pixel has the given HSV.

    h, s, v are in [0, 1].
    """
    from skimage import color as skcolor

    hsv = np.full((*shape, 3), [h, s, v], dtype=np.float64)
    rgb_float = skcolor.hsv2rgb(hsv)
    return (rgb_float * 255).astype(np.uint8)


def _make_dark_rgb(shape=(16, 16)):
    """Build an RGB image that is nearly black (low value, low saturation)."""
    return np.zeros((*shape, 3), dtype=np.uint8)


# ===================================================================
# region_analysis.py tests
# ===================================================================

from ppm_library.analysis.region_analysis import (
    compute_angles_from_rgb,
    compute_ppm_positive_mask,
    compute_masked_angles,
    compute_angle_histogram,
    compute_circular_statistics,
    filter_angles_by_range,
    analyze_region,
)


# -------------------------------------------------------------------
# compute_angles_from_rgb
# -------------------------------------------------------------------

class TestComputeAnglesFromRgb:

    @pytest.mark.usefixtures("_patch_load_calibration")
    def test_returns_expected_keys(self, mock_calibration):
        rgb = _make_saturated_rgb(0.5, 0.8, 0.8)
        result = compute_angles_from_rgb(rgb, mock_calibration)
        expected_keys = {"angles", "valid_mask", "hue", "saturation", "value", "n_valid"}
        assert set(result.keys()) == expected_keys

    @pytest.mark.usefixtures("_patch_load_calibration")
    def test_valid_mask_respects_saturation_threshold(self, mock_calibration):
        # Low saturation image -- should be mostly invalid at default thresholds
        rgb = _make_saturated_rgb(0.3, 0.05, 0.8)
        result = compute_angles_from_rgb(rgb, mock_calibration, saturation_threshold=0.2)
        assert result["n_valid"] == 0

    @pytest.mark.usefixtures("_patch_load_calibration")
    def test_valid_mask_respects_value_threshold(self, mock_calibration):
        rgb = _make_dark_rgb()
        result = compute_angles_from_rgb(rgb, mock_calibration, value_threshold=0.2)
        assert result["n_valid"] == 0

    @pytest.mark.usefixtures("_patch_load_calibration")
    def test_n_valid_matches_mask_sum(self, mock_calibration):
        rgb = _make_saturated_rgb(0.25, 0.9, 0.9, shape=(10, 10))
        result = compute_angles_from_rgb(rgb, mock_calibration)
        assert result["n_valid"] == int(np.sum(result["valid_mask"]))

    @pytest.mark.usefixtures("_patch_load_calibration")
    def test_angles_nan_where_invalid(self, mock_calibration):
        rgb = _make_dark_rgb(shape=(4, 4))
        result = compute_angles_from_rgb(rgb, mock_calibration)
        assert np.all(np.isnan(result["angles"]))

    @pytest.mark.usefixtures("_patch_load_calibration")
    def test_rejects_wrong_shape(self, mock_calibration):
        gray = np.zeros((10, 10), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected RGB"):
            compute_angles_from_rgb(gray, mock_calibration)


# -------------------------------------------------------------------
# compute_ppm_positive_mask
# -------------------------------------------------------------------

class TestComputePpmPositiveMask:

    def test_2d_array_thresholding(self):
        arr = np.array([[100, 200], [50, 300]], dtype=np.uint16)
        mask = compute_ppm_positive_mask(arr, threshold=150)
        expected = np.array([[False, True], [False, True]])
        np.testing.assert_array_equal(mask, expected)

    def test_3d_array_uses_max_across_channels(self):
        # Channel max: [200, 150]  [50, 250]
        arr = np.array([
            [[100, 200, 50], [150, 100, 80]],
            [[50, 30, 20], [250, 200, 100]],
        ], dtype=np.uint16)
        mask = compute_ppm_positive_mask(arr, threshold=160)
        expected = np.array([[True, False], [False, True]])
        np.testing.assert_array_equal(mask, expected)

    def test_rejects_wrong_dims(self):
        arr = np.zeros((2, 2, 3, 4), dtype=np.uint16)
        with pytest.raises(ValueError, match="2D or 3D"):
            compute_ppm_positive_mask(arr, threshold=100)


# -------------------------------------------------------------------
# compute_masked_angles
# -------------------------------------------------------------------

class TestComputeMaskedAngles:

    @pytest.mark.usefixtures("_patch_load_calibration")
    def test_combines_color_and_biref_masks(self, mock_calibration):
        rgb = _make_saturated_rgb(0.4, 0.9, 0.9, shape=(8, 8))
        biref = np.full((8, 8), 200, dtype=np.uint16)
        # Set bottom half biref to zero -> not PPM+
        biref[4:, :] = 0

        result = compute_masked_angles(rgb, biref, mock_calibration, biref_threshold=100)
        # Top half should be combined, bottom half should not
        assert result["n_combined"] <= result["n_color_valid"]
        assert result["n_combined"] <= result["n_ppm_positive"]
        # Bottom half angles should be NaN
        assert np.all(np.isnan(result["angles"][4:, :]))

    @pytest.mark.usefixtures("_patch_load_calibration")
    def test_dimension_mismatch_raises(self, mock_calibration):
        rgb = _make_saturated_rgb(0.4, 0.9, 0.9, shape=(8, 8))
        biref = np.full((10, 10), 200, dtype=np.uint16)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            compute_masked_angles(rgb, biref, mock_calibration, biref_threshold=100)


# -------------------------------------------------------------------
# compute_angle_histogram
# -------------------------------------------------------------------

class TestComputeAngleHistogram:

    def test_correct_bin_count(self):
        angles = np.array([10.0, 30.0, 50.0, 170.0])
        result = compute_angle_histogram(angles, bins=18)
        assert len(result["counts"]) == 18
        assert len(result["bin_centers"]) == 18
        assert len(result["bin_edges"]) == 19

    def test_bin_centers_at_correct_positions(self):
        angles = np.array([5.0])
        result = compute_angle_histogram(angles, bins=18)
        # 18 bins over [0,180] -> width 10, centers at 5, 15, 25 ...
        expected_first_center = 5.0
        expected_last_center = 175.0
        assert abs(result["bin_centers"][0] - expected_first_center) < 1e-10
        assert abs(result["bin_centers"][-1] - expected_last_center) < 1e-10

    def test_n_pixels_matches(self):
        angles = np.array([[10.0, np.nan], [50.0, 90.0]])
        result = compute_angle_histogram(angles)
        assert result["n_pixels"] == 3

    def test_mask_limits_counted_pixels(self):
        angles = np.array([[10.0, 30.0], [50.0, 90.0]])
        mask = np.array([[True, False], [True, False]])
        result = compute_angle_histogram(angles, mask=mask)
        assert result["n_pixels"] == 2


# -------------------------------------------------------------------
# compute_circular_statistics
# -------------------------------------------------------------------

class TestComputeCircularStatistics:

    def test_aligned_angles_low_std_high_R(self):
        # All angles at 90 degrees -> perfectly aligned
        angles = np.full(100, 90.0)
        result = compute_circular_statistics(angles)
        assert result["circular_std"] < 1.0
        assert result["resultant_length"] > 0.99
        assert abs(result["circular_mean"] - 90.0) < 1.0

    def test_uniform_angles_high_std_low_R(self):
        # Uniformly distributed angles 0-180
        np.random.seed(42)
        angles = np.random.uniform(0, 180, 10000)
        result = compute_circular_statistics(angles)
        assert result["resultant_length"] < 0.1
        assert result["circular_std"] > 40.0

    def test_empty_input_returns_nan(self):
        angles = np.array([np.nan, np.nan])
        result = compute_circular_statistics(angles)
        assert np.isnan(result["circular_mean"])
        assert np.isnan(result["circular_std"])
        assert np.isnan(result["resultant_length"])
        assert result["n_pixels"] == 0

    def test_axial_symmetry(self):
        # 0 and 180 should be treated as similar (axial data)
        # Mix of angles near 0 and near 180 -> mean near 0 or 180
        angles = np.array([1.0, 2.0, 179.0, 178.0])
        result = compute_circular_statistics(angles)
        # Circular mean should be near 0 (or equivalently near 180)
        mean = result["circular_mean"]
        assert mean < 10.0 or mean > 170.0
        # R should be high since they are all nearly aligned
        assert result["resultant_length"] > 0.9

    def test_with_mask(self):
        angles = np.array([[45.0, 90.0], [135.0, 10.0]])
        mask = np.array([[True, True], [False, False]])
        result = compute_circular_statistics(angles, mask=mask)
        assert result["n_pixels"] == 2


# -------------------------------------------------------------------
# filter_angles_by_range
# -------------------------------------------------------------------

class TestFilterAnglesByRange:

    def test_normal_range(self):
        angles = np.array([10.0, 50.0, 90.0, 130.0, 170.0])
        mask = np.ones(5, dtype=bool)
        result = filter_angles_by_range(angles, mask, 40.0, 100.0)
        assert result["n_in_range"] == 2  # 50 and 90
        assert result["n_valid"] == 5
        assert abs(result["fraction_in_range"] - 0.4) < 1e-10

    def test_wraparound_range(self):
        # angle_low > angle_high means wrap-around
        angles = np.array([5.0, 10.0, 90.0, 170.0, 175.0])
        mask = np.ones(5, dtype=bool)
        # Range: >= 160 OR <= 20
        result = filter_angles_by_range(angles, mask, 160.0, 20.0)
        assert result["n_in_range"] == 4  # 5, 10, 170, 175

    def test_fraction_with_partial_mask(self):
        angles = np.array([10.0, 50.0, 90.0, np.nan])
        mask = np.array([True, True, True, False])
        result = filter_angles_by_range(angles, mask, 0.0, 60.0)
        assert result["n_in_range"] == 2  # 10 and 50
        assert result["n_valid"] == 3
        assert abs(result["fraction_in_range"] - 2.0 / 3.0) < 1e-10


# -------------------------------------------------------------------
# analyze_region
# -------------------------------------------------------------------

class TestAnalyzeRegion:

    @pytest.mark.usefixtures("_patch_load_calibration")
    def test_without_biref(self, mock_calibration):
        rgb = _make_saturated_rgb(0.3, 0.9, 0.9, shape=(16, 16))
        result = analyze_region(rgb, mock_calibration)
        assert "angles" in result
        assert "mask" in result
        assert "histogram" in result
        assert "stats" in result
        assert result["ppm_positive_mask"] is None
        assert result["color_valid_mask"] is not None

    @pytest.mark.usefixtures("_patch_load_calibration")
    def test_with_biref(self, mock_calibration):
        rgb = _make_saturated_rgb(0.3, 0.9, 0.9, shape=(16, 16))
        biref = np.full((16, 16), 500, dtype=np.uint16)
        result = analyze_region(rgb, mock_calibration, biref_array=biref, biref_threshold=100)
        assert result["ppm_positive_mask"] is not None
        assert result["histogram"]["n_pixels"] > 0

    @pytest.mark.usefixtures("_patch_load_calibration")
    def test_histogram_bins_respected(self, mock_calibration):
        rgb = _make_saturated_rgb(0.3, 0.9, 0.9, shape=(16, 16))
        result = analyze_region(rgb, mock_calibration, histogram_bins=36)
        assert len(result["histogram"]["counts"]) == 36


# ===================================================================
# surface_analysis.py tests
# ===================================================================

from ppm_library.analysis.surface_analysis import (
    compute_boundary_contour,
    compute_contour_normals,
    compute_border_zone_mask,
    compute_simple_perpendicularity,
    _compute_3way_split,
    _count_clusters,
)


def _make_square_mask(size=50, margin=10):
    """Create a mask with a filled square in the center."""
    mask = np.zeros((size, size), dtype=bool)
    mask[margin:size - margin, margin:size - margin] = True
    return mask


# -------------------------------------------------------------------
# compute_boundary_contour
# -------------------------------------------------------------------

class TestComputeBoundaryContour:

    def test_returns_list_of_contours(self):
        mask = _make_square_mask()
        contours = compute_boundary_contour(mask)
        assert isinstance(contours, list)
        assert len(contours) >= 1

    def test_contour_is_xy_array(self):
        mask = _make_square_mask()
        contours = compute_boundary_contour(mask)
        c = contours[0]
        assert c.ndim == 2
        assert c.shape[1] == 2  # (x, y) columns

    def test_contour_points_near_boundary(self):
        mask = _make_square_mask(size=60, margin=15)
        contours = compute_boundary_contour(mask)
        c = contours[0]
        # All contour x values should be near the boundary edges (14.5 or 44.5)
        # or the y values should be near the boundary edges
        xs = c[:, 0]
        ys = c[:, 1]
        # At least some points should be near the margins
        assert np.any(np.abs(xs - 14.5) < 1.0) or np.any(np.abs(xs - 44.5) < 1.0)


# -------------------------------------------------------------------
# compute_contour_normals
# -------------------------------------------------------------------

class TestComputeContourNormals:

    def test_normals_are_unit_vectors(self):
        # Circular contour
        t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        contour = np.column_stack([20 + 10 * np.cos(t), 20 + 10 * np.sin(t)])
        normals = compute_contour_normals(contour, outward=False)
        lengths = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-8)

    def test_correct_count(self):
        t = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        contour = np.column_stack([20 + 10 * np.cos(t), 20 + 10 * np.sin(t)])
        normals = compute_contour_normals(contour, outward=False)
        assert normals.shape == contour.shape

    def test_small_contour_returns_zeros(self):
        contour = np.array([[0.0, 0.0], [1.0, 1.0]])
        normals = compute_contour_normals(contour, outward=False)
        np.testing.assert_array_equal(normals, np.zeros((2, 2)))


# -------------------------------------------------------------------
# compute_border_zone_mask
# -------------------------------------------------------------------

class TestComputeBorderZoneMask:

    def test_outside_mode(self):
        mask = _make_square_mask(size=50, margin=15)
        result = compute_border_zone_mask(mask, dilation_px=3, mode="outside")
        zone = result["zone_mask"]
        # Zone should not overlap with mask interior
        overlap = zone & mask
        assert np.sum(overlap) == 0

    def test_inside_mode(self):
        mask = _make_square_mask(size=50, margin=15)
        result = compute_border_zone_mask(mask, dilation_px=3, mode="inside")
        zone = result["zone_mask"]
        # Inside zone should be entirely within the mask
        assert np.all(zone <= mask)

    def test_both_mode_covers_inside_and_outside(self):
        mask = _make_square_mask(size=50, margin=15)
        r_outside = compute_border_zone_mask(mask, dilation_px=3, mode="outside")
        r_inside = compute_border_zone_mask(mask, dilation_px=3, mode="inside")
        r_both = compute_border_zone_mask(mask, dilation_px=3, mode="both")
        combined = r_outside["zone_mask"] | r_inside["zone_mask"]
        np.testing.assert_array_equal(r_both["zone_mask"], combined)

    def test_outside_zone_no_interior_overlap(self):
        mask = _make_square_mask(size=60, margin=20)
        result = compute_border_zone_mask(mask, dilation_px=5, mode="outside")
        zone = result["zone_mask"]
        # Interior pixels (far from boundary) should not be in the zone
        interior = np.zeros_like(mask)
        interior[25:35, 25:35] = True
        assert np.sum(zone & interior) == 0

    def test_distance_map_sign_convention(self):
        mask = _make_square_mask(size=50, margin=15)
        result = compute_border_zone_mask(mask, dilation_px=3, mode="outside")
        dmap = result["distance_map"]
        # Outside mask -> positive distance
        assert np.all(dmap[~mask] >= 0)
        # Inside mask -> negative distance
        assert np.all(dmap[mask] <= 0)

    def test_invalid_mode_raises(self):
        mask = _make_square_mask()
        with pytest.raises(ValueError, match="Invalid mode"):
            compute_border_zone_mask(mask, dilation_px=3, mode="invalid")


# -------------------------------------------------------------------
# compute_simple_perpendicularity
# -------------------------------------------------------------------

class TestComputeSimplePerpendicularity:

    def _setup_perpendicularity_inputs(self):
        """Create basic inputs for perpendicularity tests."""
        mask = _make_square_mask(size=60, margin=15)
        zone_result = compute_border_zone_mask(mask, dilation_px=5, mode="outside")
        zone = zone_result["zone_mask"]
        # Fiber angles: all 45 degrees
        fiber_angles = np.full((60, 60), 45.0, dtype=np.float64)
        fiber_mask = np.ones((60, 60), dtype=bool)
        return fiber_angles, fiber_mask, mask, zone

    def test_returns_expected_keys(self):
        fiber_angles, fiber_mask, mask, zone = self._setup_perpendicularity_inputs()
        result = compute_simple_perpendicularity(fiber_angles, fiber_mask, mask, zone)
        expected_keys = {
            "deviation_angles",
            "mean_deviation_deg",
            "std_deviation_deg",
            "histogram_10deg",
            "histogram_3way",
            "n_valid_pixels",
        }
        assert set(result.keys()) == expected_keys

    def test_histogram_3way_sums_to_100(self):
        fiber_angles, fiber_mask, mask, zone = self._setup_perpendicularity_inputs()
        result = compute_simple_perpendicularity(fiber_angles, fiber_mask, mask, zone)
        h3 = result["histogram_3way"]
        total_pct = h3["pct_parallel"] + h3["pct_oblique"] + h3["pct_perpendicular"]
        assert abs(total_pct - 100.0) < 0.01

    def test_n_valid_pixels_positive(self):
        fiber_angles, fiber_mask, mask, zone = self._setup_perpendicularity_inputs()
        result = compute_simple_perpendicularity(fiber_angles, fiber_mask, mask, zone)
        assert result["n_valid_pixels"] > 0


# -------------------------------------------------------------------
# _compute_3way_split
# -------------------------------------------------------------------

class TestCompute3waySplit:

    def test_all_parallel(self):
        devs = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
        result = _compute_3way_split(devs)
        assert result["parallel_count"] == 5
        assert result["oblique_count"] == 0
        assert result["perpendicular_count"] == 0
        assert abs(result["pct_parallel"] - 100.0) < 1e-10

    def test_all_perpendicular(self):
        devs = np.array([65.0, 70.0, 80.0, 85.0, 90.0])
        result = _compute_3way_split(devs)
        assert result["perpendicular_count"] == 5
        assert result["parallel_count"] == 0
        assert abs(result["pct_perpendicular"] - 100.0) < 1e-10

    def test_mixed(self):
        # 2 parallel (10, 20), 1 oblique (45), 2 perpendicular (70, 80)
        devs = np.array([10.0, 20.0, 45.0, 70.0, 80.0])
        result = _compute_3way_split(devs)
        assert result["parallel_count"] == 2
        assert result["oblique_count"] == 1
        assert result["perpendicular_count"] == 2
        total = result["pct_parallel"] + result["pct_oblique"] + result["pct_perpendicular"]
        assert abs(total - 100.0) < 1e-10

    def test_empty_input(self):
        devs = np.array([])
        result = _compute_3way_split(devs)
        assert result["parallel_count"] == 0
        assert result["oblique_count"] == 0
        assert result["perpendicular_count"] == 0
        assert result["pct_parallel"] == 0.0


# -------------------------------------------------------------------
# _count_clusters
# -------------------------------------------------------------------

class TestCountClusters:

    def test_known_pattern(self):
        # [2, 3, 3, 2, 3, 2] -> 2 clusters of value 3
        tacs = np.array([2, 3, 3, 2, 3, 2])
        assert _count_clusters(tacs, target=3) == 2

    def test_all_target(self):
        tacs = np.array([3, 3, 3, 3])
        assert _count_clusters(tacs, target=3) == 1

    def test_no_target(self):
        tacs = np.array([2, 2, 2, 2])
        assert _count_clusters(tacs, target=3) == 0

    def test_single_element_cluster(self):
        tacs = np.array([2, 3, 2, 3, 2])
        assert _count_clusters(tacs, target=3) == 2

    def test_empty_input(self):
        tacs = np.array([], dtype=int)
        assert _count_clusters(tacs, target=3) == 0

    def test_alternating(self):
        tacs = np.array([3, 2, 3, 2, 3])
        assert _count_clusters(tacs, target=3) == 3
