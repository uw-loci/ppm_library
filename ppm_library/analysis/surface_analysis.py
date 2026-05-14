"""
Surface perpendicularity analysis for PPM fiber orientation.

Implements two approaches for measuring fiber orientation relative to
tissue/annotation boundaries:

1. **Simple perpendicularity**: Average deviation angle across all valid
   pixels in a border zone around the annotation.

2. **PS-TACS scoring**: Per-contour-pixel TACS scoring with Gaussian
   distance weighting, based on Qian et al. (2025):
   "Computationally Enabled Polychromatic Polarized Imaging Enables
   Mapping of Matrix Architectures that Promote Pancreatic Ductal
   Adenocarcinoma Dissemination"
   Am J Pathol 2025; 195:1242-1253.
   DOI: https://doi.org/10.1016/j.ajpath.2025.04.017

All functions accept numpy arrays and return dicts. No GUI or file I/O
dependencies -- designed for use from QuPath (via CLI), Jupyter notebooks,
or standalone scripts.

Angle convention (user-facing):
    0 deg  = fiber parallel to boundary surface
    90 deg = fiber perpendicular to boundary surface

Internally, we compute the angle between the fiber direction vector and
the boundary normal vector, then convert:
    deviation_from_tangent = 90 - angle_from_normal

This matches the intuitive meaning: higher values = more perpendicular.
"""

import json

import numpy as np
from scipy import ndimage
from skimage import measure

# =========================================================================
# GeoJSON -> Mask
# =========================================================================


def rasterize_geojson_to_mask(geojson_path, width, height, fill_holes=True):
    """Rasterize a GeoJSON polygon to a binary mask.

    Args:
        geojson_path: path to GeoJSON file, or a dict already parsed from JSON.
            When called from Appose, a dict is passed directly to avoid
            writing a temporary file.
        width: output mask width in pixels
        height: output mask height in pixels
        fill_holes: if True, fill holes in polygons before rasterization

    Returns:
        (height, width) bool array, True inside the polygon(s)
    """

    if isinstance(geojson_path, dict):
        geojson = geojson_path
    else:
        with open(str(geojson_path)) as f:
            geojson = json.load(f)

    mask = np.zeros((height, width), dtype=bool)

    # Handle FeatureCollection or single Feature
    if geojson.get("type") == "FeatureCollection":
        features = geojson["features"]
    elif geojson.get("type") == "Feature":
        features = [geojson]
    else:
        # Bare geometry
        features = [{"geometry": geojson}]

    for feature in features:
        geom = feature.get("geometry", feature)
        _rasterize_geometry(geom, mask, fill_holes)

    return mask


def _rasterize_geometry(geom, mask, fill_holes):
    """Rasterize a single GeoJSON geometry onto a mask."""

    geom_type = geom.get("type", "")
    coords = geom.get("coordinates", [])

    if geom_type == "Polygon":
        _rasterize_polygon(coords, mask, fill_holes)
    elif geom_type == "MultiPolygon":
        for poly_coords in coords:
            _rasterize_polygon(poly_coords, mask, fill_holes)
    elif geom_type == "GeometryCollection":
        for sub_geom in geom.get("geometries", []):
            _rasterize_geometry(sub_geom, mask, fill_holes)


def _rasterize_polygon(coords, mask, fill_holes):
    """Rasterize a single polygon (with optional holes) onto a mask."""
    from skimage.draw import polygon as draw_polygon

    if not coords:
        return

    # Outer ring
    outer = np.array(coords[0])
    if outer.shape[0] < 3:
        return
    rr, cc = draw_polygon(outer[:, 1], outer[:, 0], shape=mask.shape)
    mask[rr, cc] = True

    # Holes (subtract from mask) -- only if not filling holes
    if not fill_holes and len(coords) > 1:
        for hole in coords[1:]:
            hole_arr = np.array(hole)
            if hole_arr.shape[0] < 3:
                continue
            rr, cc = draw_polygon(hole_arr[:, 1], hole_arr[:, 0], shape=mask.shape)
            mask[rr, cc] = False


# =========================================================================
# Boundary contour extraction
# =========================================================================


def compute_boundary_contour(boundary_mask, fill_holes=True):
    """Extract ordered boundary coordinates from a filled region mask.

    Args:
        boundary_mask: (H, W) bool, True inside the annotation
        fill_holes: if True, fill holes before extracting boundary

    Returns:
        list of (N, 2) arrays of (x, y) coordinates, one per connected
        component. Each array is ordered along the contour.
    """
    mask = boundary_mask.copy()
    if fill_holes:
        mask = ndimage.binary_fill_holes(mask)

    # Find contours using marching squares (gives sub-pixel accuracy)
    contours = measure.find_contours(mask.astype(float), level=0.5)

    # Convert from (row, col) to (x, y) format
    result = []
    for contour in contours:
        xy = contour[:, ::-1].copy()  # (row, col) -> (x, y)
        result.append(xy)

    return result


# =========================================================================
# Contour normals
# =========================================================================


def compute_contour_normals(contour_points, outward=True, boundary_mask=None):
    """Compute unit normal vectors at each contour point.

    Uses finite differences on the contour tangent. Normals point outward
    by default (away from the mask interior).

    Args:
        contour_points: (N, 2) array of (x, y) coordinates
        outward: if True, orient normals outward from the region
        boundary_mask: (H, W) bool mask, used to determine outward direction

    Returns:
        (N, 2) array of unit normal vectors at each contour point
    """
    N = len(contour_points)
    if N < 3:
        return np.zeros((N, 2))

    # Tangent via central differences (circular)
    tangents = np.zeros((N, 2))
    for i in range(N):
        prev_idx = (i - 1) % N
        next_idx = (i + 1) % N
        tangents[i] = contour_points[next_idx] - contour_points[prev_idx]

    # Normal = tangent rotated 90 degrees: (dx, dy) -> (-dy, dx)
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

    # Normalize
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths < 1e-10] = 1.0
    normals = normals / lengths

    # Orient outward if mask provided
    if outward and boundary_mask is not None:
        _orient_normals_outward(contour_points, normals, boundary_mask)

    return normals


def _orient_normals_outward(contour_points, normals, mask):
    """Flip normals so they point away from the mask interior."""
    h, w = mask.shape

    # Sample a few points along each normal and check if they're inside
    test_dist = 3.0  # pixels
    for i in range(len(contour_points)):
        test_pt = contour_points[i] + normals[i] * test_dist
        tx, ty = int(round(test_pt[0])), int(round(test_pt[1]))

        if 0 <= ty < h and 0 <= tx < w:
            if mask[ty, tx]:
                # Normal points inward, flip it
                normals[i] = -normals[i]


# =========================================================================
# Border zone
# =========================================================================


def compute_border_zone_mask(boundary_mask, dilation_px, mode="outside", fill_holes=True):
    """Create the analysis zone around the annotation boundary.

    Args:
        boundary_mask: (H, W) bool, True inside the annotation
        dilation_px: dilation distance in pixels (rounded to int)
        mode: 'outside' (default), 'inside', or 'both'
        fill_holes: fill holes in mask before computing zone

    Returns:
        dict with:
            'zone_mask': (H, W) bool, the border zone
            'distance_map': (H, W) float, signed distance from boundary
                (positive = outside, negative = inside)
    """
    mask = boundary_mask.copy()
    if fill_holes:
        mask = ndimage.binary_fill_holes(mask)

    dilation_px = max(1, int(round(dilation_px)))

    # Distance transforms
    dist_outside = ndimage.distance_transform_edt(~mask)
    dist_inside = ndimage.distance_transform_edt(mask)

    # Signed distance (positive outside, negative inside)
    signed_distance = dist_outside - dist_inside

    # Build zone mask
    if mode == "outside":
        zone = (dist_outside > 0) & (dist_outside <= dilation_px)
    elif mode == "inside":
        zone = (dist_inside > 0) & (dist_inside <= dilation_px)
    elif mode == "both":
        zone = ((dist_outside > 0) & (dist_outside <= dilation_px)) | (
            (dist_inside > 0) & (dist_inside <= dilation_px)
        )
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'outside', 'inside', or 'both'.")

    return {
        "zone_mask": zone,
        "distance_map": signed_distance,
        "dist_from_boundary": np.minimum(dist_outside, dist_inside),
    }


# =========================================================================
# Simple perpendicularity (fallback approach)
# =========================================================================


def compute_simple_perpendicularity(
    fiber_angles, fiber_mask, boundary_mask, zone_mask, fill_holes=True
):
    """Compute average fiber-to-boundary deviation using the distance
    transform gradient for surface normals.

    This is the simpler approach: computes a single normal angle per pixel
    from the distance transform gradient, then measures the deviation of
    each fiber from its local normal.

    Args:
        fiber_angles: (H, W) float, fiber angles in degrees (0-180)
        fiber_mask: (H, W) bool, valid fiber pixels
        boundary_mask: (H, W) bool, annotation interior
        zone_mask: (H, W) bool, analysis zone
        fill_holes: fill holes before computing normals

    Returns:
        dict with:
            'deviation_angles': (H, W) float, deviation from tangent in degrees
                (0=parallel, 90=perpendicular), NaN where invalid
            'mean_deviation_deg': float, mean deviation across valid pixels
            'std_deviation_deg': float, std of deviation
            'histogram_10deg': dict with counts per 10-degree bin
            'histogram_3way': dict with parallel/oblique/perpendicular counts
            'n_valid_pixels': int
    """
    mask = boundary_mask.copy()
    if fill_holes:
        mask = ndimage.binary_fill_holes(mask)

    # Distance transform for normal direction
    dist = ndimage.distance_transform_edt(~mask)
    # Gradient of distance field = direction away from boundary = outward normal
    grad_y, grad_x = np.gradient(dist)

    # Normal angle at each pixel (0-180, axial)
    # Negate grad_y to convert from image coordinates (y-down) to math
    # convention (y-up), matching the fiber angle convention from calibration.
    normal_angle_rad = np.arctan2(-grad_y, grad_x)
    normal_angle_deg = np.degrees(normal_angle_rad) % 180

    # Analysis mask: in zone AND valid fiber
    analysis_mask = zone_mask & fiber_mask & ~np.isnan(fiber_angles)

    # Compute deviation from tangent for each pixel
    # angle_from_normal = |fiber - normal| (mod 180, axial)
    # deviation_from_tangent = 90 - angle_from_normal
    deviation = np.full_like(fiber_angles, np.nan)

    if np.any(analysis_mask):
        diff = np.abs(fiber_angles[analysis_mask] - normal_angle_deg[analysis_mask])
        # Handle axial wrap: min(diff, 180 - diff)
        angle_from_normal = np.minimum(diff, 180.0 - diff)
        # Convert: 0 from normal = perpendicular to boundary = 90 deviation from tangent
        deviation_from_tangent = 90.0 - angle_from_normal
        # Clamp to [0, 90]
        deviation_from_tangent = np.clip(deviation_from_tangent, 0.0, 90.0)
        deviation[analysis_mask] = deviation_from_tangent

    valid_deviations = deviation[analysis_mask] if np.any(analysis_mask) else np.array([])
    n_valid = len(valid_deviations)

    # 10-degree histogram
    hist_10deg = _compute_deviation_histogram(valid_deviations, bin_width=10)

    # 3-way split
    hist_3way = _compute_3way_split(valid_deviations)

    return {
        "deviation_angles": deviation,
        "normal_angle_deg": normal_angle_deg,
        "mean_deviation_deg": float(np.mean(valid_deviations)) if n_valid > 0 else float("nan"),
        "std_deviation_deg": float(np.std(valid_deviations)) if n_valid > 0 else float("nan"),
        "histogram_10deg": hist_10deg,
        "histogram_3way": hist_3way,
        "n_valid_pixels": n_valid,
    }


# =========================================================================
# PS-TACS scoring (Qian et al. algorithm)
# =========================================================================


def compute_tacs_scores(
    fiber_angles,
    fiber_mask,
    boundary_mask,
    contour_points,
    contour_normals,
    zone_mask,
    distance_from_boundary,
    tacs_threshold_deg=30.0,
    falloff_sigma_px=None,
    smoothing_window=10,
):
    """Compute per-contour-pixel TACS score using distance-weighted dot products.

    Implements the PS-TACS algorithm from Qian et al. (Eq. 2-4).
    For each contour pixel:
    - Find fiber pixels in the analysis zone
    - Compute angle between fiber direction and boundary normal (dot product)
    - Weight by Gaussian based on distance from contour pixel
    - Assign TACS-2 (parallel) or TACS-3 (perpendicular) based on threshold
    - Average weighted scores

    Note: tacs_threshold_deg is measured from the boundary NORMAL.
    Fibers within tacs_threshold_deg of the normal are TACS-3 (perpendicular).

    Args:
        fiber_angles: (H, W) float, fiber angles in degrees (0-180)
        fiber_mask: (H, W) bool, valid fiber pixels
        boundary_mask: (H, W) bool, annotation interior
        contour_points: (N, 2) ordered (x, y) boundary coordinates
        contour_normals: (N, 2) outward unit normal vectors
        zone_mask: (H, W) bool, analysis zone
        distance_from_boundary: (H, W) float, distance to nearest boundary
        tacs_threshold_deg: angle from normal below which = TACS-3 (default 30)
        falloff_sigma_px: Gaussian sigma for distance weighting; None = auto
        smoothing_window: window size for moving average of contour scores

    Returns:
        dict with:
            'contour_scores_raw': (N,) float, raw TACS score [-1, +1]
            'contour_scores_smoothed': (N,) float, smoothed scores [0, 1]
            'contour_tacs_class': (N,) int, 2 or 3 per contour pixel
            'contour_points': (N, 2) the contour coordinates
            'pct_tacs2': float
            'pct_tacs3': float
            'n_tacs3_clusters': int
            'contour_length_px': int
            'tacs_threshold_deg': float (echo back for reference)
    """
    N = len(contour_points)
    analysis_mask = zone_mask & fiber_mask & ~np.isnan(fiber_angles)

    if N < 3 or not np.any(analysis_mask):
        return _empty_tacs_result(N, contour_points, tacs_threshold_deg)

    # Get fiber pixel locations and angles
    fiber_ys, fiber_xs = np.where(analysis_mask)
    fiber_locs = np.column_stack([fiber_xs, fiber_ys]).astype(float)
    fiber_angle_values = fiber_angles[fiber_ys, fiber_xs]

    # Fiber direction vectors (unit vectors from angle)
    # Fiber angles are in math convention (CCW from horizontal, y-up) but
    # contour normals are in image coordinates (y-down). Negate the
    # y-component so the dot product uses a consistent coordinate system.
    fiber_rad = np.radians(fiber_angle_values)
    fiber_vecs = np.column_stack([np.cos(fiber_rad), -np.sin(fiber_rad)])

    # Auto-compute falloff sigma if not provided
    if falloff_sigma_px is None:
        # Mean distance from boundary of fiber pixels
        fiber_dists = distance_from_boundary[fiber_ys, fiber_xs]
        mean_dist = np.mean(fiber_dists) if len(fiber_dists) > 0 else 10.0
        falloff_sigma_px = max(mean_dist / 3.0, 1.0)

    # Per-contour-pixel scoring
    scores_raw = np.zeros(N)
    density_raw = np.zeros(N)  # Gaussian-weighted fiber count per contour pixel
    threshold_rad = np.radians(tacs_threshold_deg)

    for i in range(N):
        cx, cy = contour_points[i]
        nx, ny = contour_normals[i]
        normal_vec = np.array([nx, ny])

        # Distances from this contour pixel to all fiber pixels
        dists = np.sqrt((fiber_locs[:, 0] - cx) ** 2 + (fiber_locs[:, 1] - cy) ** 2)

        # Gaussian weights
        weights = np.exp(-0.5 * (dists / falloff_sigma_px) ** 2)

        # Only consider fibers within a reasonable range (3*sigma)
        in_range = dists <= 3 * falloff_sigma_px
        if not np.any(in_range):
            scores_raw[i] = 0.0
            continue

        w = weights[in_range]
        density_raw[i] = np.sum(w)
        fv = fiber_vecs[in_range]

        # Dot product: angle between fiber and normal
        dots = np.abs(fv @ normal_vec)
        dots = np.clip(dots, -1.0, 1.0)
        angles_from_normal = np.arccos(dots)  # [0, pi/2] for axial data

        # TACS assignment per fiber pixel
        # TACS-3 (perpendicular): angle_from_normal < threshold
        # TACS-2 (parallel): angle_from_normal >= threshold
        tacs_assign = np.where(angles_from_normal < threshold_rad, 1.0, -1.0)

        # Weighted average
        if np.sum(w) > 0:
            scores_raw[i] = np.sum(tacs_assign * w) / np.sum(w)

    # Smooth along contour
    if smoothing_window > 1 and smoothing_window < N:
        kernel = np.ones(smoothing_window) / smoothing_window
        # Circular convolution
        padded = np.concatenate(
            [scores_raw[-smoothing_window:], scores_raw, scores_raw[:smoothing_window]]
        )
        smoothed_padded = np.convolve(padded, kernel, mode="same")
        scores_smoothed = smoothed_padded[smoothing_window:-smoothing_window]
    else:
        scores_smoothed = scores_raw.copy()

    # Map from [-1, 1] to [0, 1] for display (0 = TACS-2, 1 = TACS-3)
    scores_01 = (scores_smoothed + 1.0) / 2.0
    scores_01 = np.clip(scores_01, 0.0, 1.0)

    # TACS class per contour pixel
    tacs_class = np.where(scores_01 >= 0.5, 3, 2)

    # Count TACS-3 clusters (contiguous runs of class 3)
    n_tacs3_clusters = _count_clusters(tacs_class, target=3)

    # Percentages
    n_tacs2 = int(np.sum(tacs_class == 2))
    n_tacs3 = int(np.sum(tacs_class == 3))
    total = n_tacs2 + n_tacs3

    return {
        "contour_scores_raw": scores_raw,
        "contour_scores_smoothed": scores_01,
        "contour_tacs_class": tacs_class,
        "contour_points": contour_points,
        "pct_tacs2": 100.0 * n_tacs2 / total if total > 0 else 0.0,
        "pct_tacs3": 100.0 * n_tacs3 / total if total > 0 else 0.0,
        "n_tacs3_clusters": n_tacs3_clusters,
        "contour_length_px": N,
        "tacs_threshold_deg": tacs_threshold_deg,
        "contour_density_raw": density_raw,
    }


def compute_extended_tacs(pstacs_result, min_collagen_density=0.1, min_signal_threshold=0.02):
    """Reclassify contour segments using Unclassified/TACS-1/2/3 scheme.

    Takes the output of compute_tacs_scores() and reclassifies contour
    pixels based on collagen density:
      - Below min_signal_threshold: class 0 (Unclassified, no collagen)
      - Between min_signal and min_collagen_density: class 1 (TACS-1, sparse)
      - Above min_collagen_density: retains PS-TACS class (2 or 3)

    This is NOT part of the PS-TACS method (Qian et al.). It extends
    the classification with TACS-1 as defined in Provenzano et al.
    (2006, BMC Medicine) and adds an Unclassified state for regions
    with no meaningful collagen signal.

    Args:
        pstacs_result: dict from compute_tacs_scores(), must contain
            'contour_tacs_class', 'contour_density_raw', 'contour_points'
        min_collagen_density: float in [0, 1], threshold on normalized
            density. Below this but above signal threshold = TACS-1.
            Default 0.1 (10% of the densest region).
        min_signal_threshold: float in [0, 1], threshold below which
            a contour pixel is considered to have no collagen signal
            at all (Unclassified, class 0). Default 0.02 (2% of peak).

    Returns:
        dict with:
            'extended_tacs_class': (N,) int, 0/1/2/3 per contour pixel
                0=Unclassified, 1=TACS-1, 2=TACS-2, 3=TACS-3
            'contour_points': (N, 2) the contour coordinates
            'density_normalized': (N,) float [0, 1], collagen density
            'pct_unclassified': float
            'pct_tacs1': float
            'pct_tacs2': float
            'pct_tacs3': float
            'n_tacs1_clusters': int
            'n_tacs3_clusters': int
            'min_collagen_density': float (echo back)
            'min_signal_threshold': float (echo back)
    """
    density_raw = pstacs_result["contour_density_raw"]
    max_density = (
        float(np.max(density_raw)) if len(density_raw) > 0 and np.max(density_raw) > 0 else 1.0
    )
    density_norm = density_raw / max_density

    # Start from PS-TACS classes (2 or 3), then reclassify by density
    ext_class = pstacs_result["contour_tacs_class"].copy()
    # Sparse collagen -> TACS-1
    ext_class[density_norm < min_collagen_density] = 1
    # No collagen signal -> Unclassified (0)
    ext_class[density_norm < min_signal_threshold] = 0

    N = len(ext_class)
    n0 = int(np.sum(ext_class == 0))
    n1 = int(np.sum(ext_class == 1))
    n2 = int(np.sum(ext_class == 2))
    n3 = int(np.sum(ext_class == 3))
    total = n0 + n1 + n2 + n3

    return {
        "extended_tacs_class": ext_class,
        "contour_points": pstacs_result["contour_points"],
        "density_normalized": density_norm,
        "pct_unclassified": 100.0 * n0 / total if total > 0 else 0.0,
        "pct_tacs1": 100.0 * n1 / total if total > 0 else 0.0,
        "pct_tacs2": 100.0 * n2 / total if total > 0 else 0.0,
        "pct_tacs3": 100.0 * n3 / total if total > 0 else 0.0,
        "n_tacs1_clusters": _count_clusters(ext_class, target=1),
        "n_tacs3_clusters": _count_clusters(ext_class, target=3),
        "min_collagen_density": min_collagen_density,
        "min_signal_threshold": min_signal_threshold,
    }


# =========================================================================
# All-in-one entry point
# =========================================================================


def analyze_perpendicularity(
    rgb_array,
    calibration,
    boundary_mask,
    dilation_um,
    pixel_size_um,
    mode="outside",
    fill_holes=True,
    tacs_threshold_deg=30.0,
    smoothing_window=10,
    boundary_smoothing_sigma=5.0,
    biref_array=None,
    biref_threshold=100,
    saturation_threshold=0.2,
    value_threshold=0.2,
    foreground_mask=None,
    min_rgb_intensity=100,
    extended_tacs=False,
    min_collagen_density=0.1,
    min_signal_threshold=0.02,
    biref_blur_sigma=0.0,
    hsv_blur_sigma=0.0,
):
    """All-in-one entry point for surface perpendicularity analysis.

    Runs both the simple approach and the PS-TACS approach.

    Args:
        rgb_array: (H, W, 3) uint8 sum image
        calibration: RadialCalibrationResult with hue-to-angle transform
        boundary_mask: (H, W) bool, annotation interior
        dilation_um: border zone width in microns
        pixel_size_um: pixel size in microns (for um -> px conversion)
        mode: 'outside', 'inside', or 'both'
        fill_holes: fill holes in boundary mask
        tacs_threshold_deg: angle threshold for PS-TACS (default 30)
        smoothing_window: PS-TACS contour smoothing window
        boundary_smoothing_sigma: Gaussian sigma for smoothing the boundary
            mask before contour extraction and normal computation. Removes
            pixel-level staircase artifacts from segmentation boundaries so
            that normals reflect the general tumor surface direction. Set to
            0 to disable smoothing. Default 5.0 pixels.
        biref_array: optional birefringence image for PPM+ masking
            (ignored if foreground_mask is provided)
        biref_threshold: biref intensity threshold
            (ignored if foreground_mask is provided)
        saturation_threshold: min HSV saturation for valid fiber pixels
        value_threshold: min HSV value for valid fiber pixels
        foreground_mask: optional external binary mask (H, W), True for
            foreground pixels. Replaces biref-based masking when provided.
        min_rgb_intensity: minimum max(R,G,B) to include a pixel. Excludes
            dark absorbing tissue (e.g. hematoxylin nuclei). Default 100.
        biref_blur_sigma: Gaussian sigma (px) applied to the biref image
            before threshold gating. Only affects the biref-positive mask;
            angle computation is unchanged. 0 disables. Default 0.
        hsv_blur_sigma: Gaussian sigma (px) applied to a copy of the RGB
            image before HSV / value / min-RGB validity testing. The angle
            field is still computed from the original RGB so fiber
            orientation isn't smoothed. 0 disables. Default 0.

    Returns:
        dict with:
            'simple': results from compute_simple_perpendicularity()
            'pstacs': results from compute_tacs_scores()
            'dilation_px': int, computed dilation in pixels
            'pixel_size_um': float
            'contour_length_um': float
            'n_contours': int
    """
    import time as _time
    import logging as _logging

    _perf_log = _logging.getLogger("ppm.perf")
    _t0 = _time.perf_counter()

    def _lap(label):
        nonlocal _t0
        now = _time.perf_counter()
        _perf_log.info("PERF %s: %.3fs", label, now - _t0)
        _t0 = now

    from ppm_library.analysis.region_analysis import (
        compute_angles_from_rgb,
        compute_ppm_positive_mask,
    )

    _lap("imports")

    dilation_px = dilation_um / pixel_size_um
    dilation_px_int = max(1, int(round(dilation_px)))

    # Compute fiber angles from the ORIGINAL RGB so orientation isn't smoothed.
    angle_result = compute_angles_from_rgb(
        rgb_array,
        calibration,
        saturation_threshold=saturation_threshold,
        value_threshold=value_threshold,
        exclude_clipped=True,
        min_rgb_intensity=min_rgb_intensity,
    )
    _lap("compute_angles_from_rgb")
    fiber_angles = angle_result["angles"]
    fiber_mask = angle_result["valid_mask"]
    n_clipped = angle_result.get("n_clipped", 0)
    n_dark = angle_result.get("n_dark_excluded", 0)
    total_pixels = rgb_array.shape[0] * rgb_array.shape[1]
    hsv_valid_count = int(np.sum(fiber_mask))

    # Optionally re-derive the HSV/intensity validity mask from a BLURRED copy
    # of the RGB image so threshold edges are less ragged. Angle values stay
    # from the original RGB above; only the validity mask is replaced.
    if hsv_blur_sigma and hsv_blur_sigma > 0:
        from skimage import color as _skcolor

        blurred_rgb_f = np.empty(rgb_array.shape, dtype=np.float32)
        for c in range(3):
            blurred_rgb_f[:, :, c] = ndimage.gaussian_filter(
                rgb_array[:, :, c].astype(np.float32), sigma=hsv_blur_sigma
            )
        blurred_rgb = np.clip(blurred_rgb_f, 0, 255).astype(np.uint8)
        hsv_b = _skcolor.rgb2hsv(blurred_rgb)
        blurred_valid = (hsv_b[:, :, 1] >= saturation_threshold) & (
            hsv_b[:, :, 2] >= value_threshold
        )
        if min_rgb_intensity > 0:
            blurred_valid = blurred_valid & ~(np.max(blurred_rgb, axis=2) < min_rgb_intensity)
        # Exclude clipped pixels from the blurred image for consistency
        blurred_valid = blurred_valid & ~np.any(blurred_rgb == 255, axis=2)
        fiber_mask = blurred_valid
        hsv_valid_count = int(np.sum(fiber_mask))
        _lap("hsv_blur_revalid")

    # Apply foreground mask (from pixel classifier) or biref mask
    biref_valid_count = -1
    if foreground_mask is not None:
        fg = foreground_mask.astype(bool)
        if fg.shape != fiber_mask.shape:
            raise ValueError(f"Foreground mask shape {fg.shape} != image shape {fiber_mask.shape}")
        fiber_mask = fiber_mask & fg
    elif biref_array is not None:
        biref_for_thresh = biref_array
        if biref_blur_sigma and biref_blur_sigma > 0:
            biref_for_thresh = ndimage.gaussian_filter(
                biref_array.astype(np.float32), sigma=biref_blur_sigma
            )
        biref_mask = compute_ppm_positive_mask(biref_for_thresh, biref_threshold)
        biref_valid_count = int(np.sum(biref_mask))
        fiber_mask = fiber_mask & biref_mask
    combined_valid_count = int(np.sum(fiber_mask))
    _lap("biref_masking")

    # Fill holes if requested
    mask_for_analysis = boundary_mask.copy()
    if fill_holes:
        mask_for_analysis = ndimage.binary_fill_holes(mask_for_analysis)

    # Smooth boundary for contour/normal computation to remove pixel-level
    # staircase artifacts. Zone computation uses the original mask so the
    # analysis region stays true to the annotation.
    if boundary_smoothing_sigma and boundary_smoothing_sigma > 0:
        smoothed_float = ndimage.gaussian_filter(
            mask_for_analysis.astype(np.float64), sigma=boundary_smoothing_sigma
        )
        mask_for_contours = smoothed_float > 0.5
        # Preserve the overall shape -- re-fill holes after smoothing
        if fill_holes:
            mask_for_contours = ndimage.binary_fill_holes(mask_for_contours)
    else:
        mask_for_contours = mask_for_analysis

    _lap("boundary_smoothing")

    # Border zone (uses original annotation, not smoothed)
    zone_result = compute_border_zone_mask(
        mask_for_analysis, dilation_px_int, mode=mode, fill_holes=False
    )
    zone_mask = zone_result["zone_mask"]
    dist_from_boundary = zone_result["dist_from_boundary"]
    _lap("compute_border_zone_mask")

    # Simple approach (uses smoothed mask for distance-transform normals)
    simple_result = compute_simple_perpendicularity(
        fiber_angles,
        fiber_mask,
        mask_for_contours,
        zone_mask,
        fill_holes=False,
    )

    _lap("compute_simple_perpendicularity")

    # PS-TACS approach (uses smoothed mask for contour extraction)
    contours = compute_boundary_contour(mask_for_contours, fill_holes=False)

    pstacs_result = None
    contour_length_um = 0.0
    if contours:
        # Use the longest contour (main boundary)
        main_contour = max(contours, key=len)
        normals = compute_contour_normals(
            main_contour, outward=True, boundary_mask=mask_for_contours
        )
        pstacs_result = compute_tacs_scores(
            fiber_angles,
            fiber_mask,
            mask_for_analysis,
            main_contour,
            normals,
            zone_mask,
            dist_from_boundary,
            tacs_threshold_deg=tacs_threshold_deg,
            smoothing_window=smoothing_window,
        )

        _lap("compute_tacs_scores")

        # Contour length in microns
        diffs = np.diff(main_contour, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        contour_length_um = float(np.sum(segment_lengths) * pixel_size_um)

    # Extended TACS: reclassify sparse-collagen regions as TACS-1
    extended_tacs_result = None
    if extended_tacs and pstacs_result is not None:
        extended_tacs_result = compute_extended_tacs(
            pstacs_result,
            min_collagen_density=min_collagen_density,
            min_signal_threshold=min_signal_threshold,
        )

    return {
        "simple": simple_result,
        "pstacs": pstacs_result,
        "extended_tacs": extended_tacs_result,
        "dilation_px": dilation_px_int,
        "pixel_size_um": pixel_size_um,
        "contour_length_um": contour_length_um,
        "n_contours": len(contours),
        "n_clipped_pixels": n_clipped,
        # Intermediate masks and per-pixel fields for visualization / persistence
        "fiber_mask": fiber_mask,
        "zone_mask": zone_mask,
        "fiber_angles": fiber_angles,
        "dist_from_boundary": dist_from_boundary,
        # Diagnostic counts (avoids recomputation of masks for stats)
        "mask_diagnostics": {
            "total_pixels": total_pixels,
            "hsv_valid_pixels": hsv_valid_count,
            "clipped_pixels": n_clipped,
            "dark_excluded_pixels": n_dark,
            "biref_valid_pixels": biref_valid_count,
            "combined_valid_pixels": combined_valid_count,
            "zone_pixels": int(np.sum(zone_mask)),
        },
    }


# =========================================================================
# Persistence and rendering of per-pixel results
# =========================================================================


def save_pixel_arrays(result, output_dir):
    """Save per-pixel arrays from analyze_perpendicularity() to .npy files.

    Writes (when present in result):
        deviation_angles.npy   -- (H, W) float, 0..90, NaN where invalid
        fiber_angles.npy       -- (H, W) float, 0..180, NaN where invalid
        fiber_mask.npy         -- (H, W) bool, valid fiber pixels
        zone_mask.npy          -- (H, W) bool, interrogation zone
        dist_from_boundary.npy -- (H, W) float, pixels to nearest boundary
        normal_angle_deg.npy   -- (H, W) float, local outward-normal angle 0..180

    Args:
        result: dict from analyze_perpendicularity(); may include any subset
            of the arrays above. Missing keys are silently skipped.
        output_dir: directory to write into. Created if needed.
    """
    import os

    os.makedirs(str(output_dir), exist_ok=True)
    out = str(output_dir).rstrip("/\\")

    simple = result.get("simple") or {}
    pairs = [
        ("deviation_angles", simple.get("deviation_angles")),
        ("normal_angle_deg", simple.get("normal_angle_deg")),
        ("fiber_angles", result.get("fiber_angles")),
        ("fiber_mask", result.get("fiber_mask")),
        ("zone_mask", result.get("zone_mask")),
        ("dist_from_boundary", result.get("dist_from_boundary")),
    ]
    for name, arr in pairs:
        if arr is None:
            continue
        np.save(os.path.join(out, name + ".npy"), arr)


def render_orientation_overlay(
    deviation_angles,
    fiber_mask,
    output_path,
    cmap_name="seismic",
):
    """Render a blue-to-red RGBA PNG of pixel-wise relative orientation.

    Maps deviation angles [0, 90] deg through a diverging blue->red colormap
    (parallel/TACS-2 = blue, perpendicular/TACS-3 = red), matching the
    convention used in Qian et al. 2025 Fig 4 E/F. Pixels outside fiber_mask
    or with NaN deviation are rendered fully transparent so the underlying
    biref image shows through.

    Args:
        deviation_angles: (H, W) float, values in [0, 90], NaN allowed
        fiber_mask: (H, W) bool, valid fiber pixels; the overlay is opaque
            inside the mask and transparent outside
        output_path: PNG file path to write
        cmap_name: matplotlib colormap name (default 'seismic'). Use
            'RdBu_r' or 'coolwarm' for slightly different blue/red ramps.
    """
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = deviation_angles.shape
    valid = np.asarray(fiber_mask, dtype=bool) & ~np.isnan(deviation_angles)

    # Normalize deviation [0, 90] -> [0, 1] for colormap input.
    norm = np.zeros_like(deviation_angles, dtype=np.float32)
    if np.any(valid):
        norm[valid] = np.clip(deviation_angles[valid] / 90.0, 0.0, 1.0)

    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(norm)  # (H, W, 4) float in [0, 1]
    # Override alpha: opaque where valid, transparent elsewhere.
    rgba[..., 3] = valid.astype(np.float32)

    rgba_u8 = (rgba * 255.0).clip(0, 255).astype(np.uint8)

    out_path = str(output_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # Use Pillow directly to avoid pyplot figure machinery.
    from PIL import Image

    Image.fromarray(rgba_u8, mode="RGBA").save(out_path)


# =========================================================================
# Histogram helpers
# =========================================================================


def _compute_deviation_histogram(deviations, bin_width=10):
    """Compute histogram of deviation angles in fixed-width bins.

    Args:
        deviations: 1D array of deviation angles (0-90 deg)
        bin_width: bin width in degrees

    Returns:
        dict with 'bin_edges', 'bin_centers', 'counts', 'fractions'
    """
    n_bins = int(90 / bin_width)
    bin_edges = np.linspace(0, 90, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if len(deviations) == 0:
        return {
            "bin_edges": bin_edges.tolist(),
            "bin_centers": bin_centers.tolist(),
            "counts": [0] * n_bins,
            "fractions": [0.0] * n_bins,
        }

    counts, _ = np.histogram(deviations, bins=bin_edges)
    total = len(deviations)

    return {
        "bin_edges": bin_edges.tolist(),
        "bin_centers": bin_centers.tolist(),
        "counts": counts.tolist(),
        "fractions": (counts / total).tolist() if total > 0 else [0.0] * n_bins,
    }


def _compute_3way_split(deviations):
    """Compute parallel / oblique / perpendicular split.

    Categories:
        Parallel:      0-30 deg from boundary
        Oblique:       30-60 deg from boundary
        Perpendicular: 60-90 deg from boundary

    Returns:
        dict with counts and percentages for each category
    """
    n = len(deviations)
    if n == 0:
        return {
            "parallel_count": 0,
            "oblique_count": 0,
            "perpendicular_count": 0,
            "pct_parallel": 0.0,
            "pct_oblique": 0.0,
            "pct_perpendicular": 0.0,
        }

    parallel = int(np.sum(deviations < 30))
    oblique = int(np.sum((deviations >= 30) & (deviations < 60)))
    perpendicular = int(np.sum(deviations >= 60))

    return {
        "parallel_count": parallel,
        "oblique_count": oblique,
        "perpendicular_count": perpendicular,
        "pct_parallel": 100.0 * parallel / n,
        "pct_oblique": 100.0 * oblique / n,
        "pct_perpendicular": 100.0 * perpendicular / n,
    }


def _count_clusters(tacs_class, target=3):
    """Count contiguous clusters of a target class along the contour."""
    in_cluster = False
    count = 0
    for val in tacs_class:
        if val == target:
            if not in_cluster:
                count += 1
                in_cluster = True
        else:
            in_cluster = False
    return count


def _empty_tacs_result(N, contour_points, tacs_threshold_deg):
    """Return an empty PS-TACS result when no valid data is available."""
    return {
        "contour_scores_raw": np.zeros(N),
        "contour_scores_smoothed": np.full(N, 0.5),
        "contour_tacs_class": np.full(N, 2, dtype=int),
        "contour_points": contour_points,
        "pct_tacs2": 100.0,
        "pct_tacs3": 0.0,
        "n_tacs3_clusters": 0,
        "contour_length_px": N,
        "tacs_threshold_deg": tacs_threshold_deg,
        "contour_density_raw": np.zeros(N),
    }
