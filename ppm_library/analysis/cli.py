"""
CLI entry point for PPM region analysis.

Called by QuPath (via ProcessBuilder) to analyze image regions using
ppm_library functions. Accepts temp file paths for image data and
outputs JSON results on stdout.

Modes:
    analyze (default) - Full region analysis with histogram and circular stats
    filter            - Angle range filtering, reports pixel counts in range
    perpendicularity  - Surface perpendicularity analysis (simple + PS-TACS)

Usage examples:

    # Full analysis (histogram + stats)
    python -m ppm_library.analysis.cli \\
        --sum /tmp/sum_region.tif \\
        --calibration /path/to/calibration.npz \\
        --biref /tmp/biref_region.tif \\
        --biref-threshold 100 \\
        --bins 18

    # Angle range filter
    python -m ppm_library.analysis.cli \\
        --mode filter \\
        --sum /tmp/sum_region.tif \\
        --calibration /path/to/calibration.npz \\
        --angle-low 30 --angle-high 60

Output JSON format (analyze mode):
    {
        "histogram_counts": [12, 45, ...],
        "histogram_bin_edges": [0.0, 10.0, ...],
        "histogram_bin_centers": [5.0, 15.0, ...],
        "circular_mean": 67.3,
        "circular_std": 23.1,
        "resultant_length": 0.72,
        "n_pixels": 12345,
        "arithmetic_mean": 68.1,
        "arithmetic_std": 24.5
    }

Output JSON format (filter mode):
    {
        "angle_low": 30.0,
        "angle_high": 60.0,
        "n_in_range": 5432,
        "n_valid": 12345,
        "fraction_in_range": 0.44
    }

    # Surface perpendicularity analysis
    python -m ppm_library.analysis.cli \\
        --mode perpendicularity \\
        --sum /tmp/sum_region.tif \\
        --calibration /path/to/calibration.npz \\
        --boundary /tmp/boundary.geojson \\
        --dilation-um 50 \\
        --pixel-size-um 0.5 \\
        --output-dir /path/to/results/

Output JSON format (perpendicularity mode):
    {
        "simple": {
            "mean_deviation_deg": 42.3,
            "std_deviation_deg": 18.7,
            "histogram_10deg": {...},
            "histogram_3way": {...},
            "n_valid_pixels": 8432
        },
        "pstacs": {
            "pct_tacs2": 62.1,
            "pct_tacs3": 37.9,
            "n_tacs3_clusters": 3,
            "contour_length_px": 1245,
            "tacs_threshold_deg": 30.0
        },
        "dilation_px": 100,
        "pixel_size_um": 0.5,
        "contour_length_um": 622.5,
        "n_contours": 1
    }
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from skimage.io import imread


def main():
    parser = argparse.ArgumentParser(
        description="PPM region analysis - compute fiber angle statistics"
    )
    parser.add_argument("--sum", required=True, help="Path to sum image region (RGB TIFF/PNG)")
    parser.add_argument("--calibration", required=True, help="Path to calibration .npz file")
    parser.add_argument(
        "--biref", help="Path to birefringence image region (optional, for PPM+ masking)"
    )
    parser.add_argument(
        "--biref-threshold",
        type=float,
        default=100.0,
        help="Birefringence threshold for PPM-positive detection (default: 100)",
    )
    parser.add_argument(
        "--saturation-threshold",
        type=float,
        default=0.2,
        help="Min HSV saturation for valid pixels (default: 0.2)",
    )
    parser.add_argument(
        "--value-threshold",
        type=float,
        default=0.2,
        help="Min HSV value for valid pixels (default: 0.2)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=18,
        help="Number of histogram bins (default: 18 = 10-degree bins)",
    )
    parser.add_argument("--roi-mask", help="Path to ROI mask image (binary, same dims as sum)")
    parser.add_argument(
        "--foreground-mask",
        help="Path to external foreground mask (binary TIFF, same dims as sum). "
        "When provided, replaces the biref threshold mask for foreground detection. "
        "Typically generated from a QuPath pixel classifier or thresholder.",
    )
    parser.add_argument(
        "--mode",
        choices=["analyze", "filter", "perpendicularity"],
        default="analyze",
        help="Analysis mode: 'analyze' for full stats, 'filter' for angle range, 'perpendicularity' for surface analysis",
    )
    parser.add_argument(
        "--angle-low",
        type=float,
        default=0.0,
        help="Low angle bound in degrees for filter mode (default: 0)",
    )
    parser.add_argument(
        "--angle-high",
        type=float,
        default=180.0,
        help="High angle bound in degrees for filter mode (default: 180)",
    )
    # Perpendicularity mode args
    parser.add_argument("--boundary", help="Path to boundary GeoJSON file (perpendicularity mode)")
    parser.add_argument(
        "--dilation-um", type=float, default=50.0, help="Border zone width in microns (default: 50)"
    )
    parser.add_argument(
        "--pixel-size-um",
        type=float,
        help="Pixel size in microns (required for perpendicularity mode)",
    )
    parser.add_argument(
        "--zone-mode",
        choices=["outside", "inside", "both"],
        default="outside",
        help="Border zone mode (default: outside)",
    )
    parser.add_argument(
        "--fill-holes",
        action="store_true",
        default=True,
        help="Fill holes in boundary mask (default: True)",
    )
    parser.add_argument(
        "--no-fill-holes", action="store_true", help="Do NOT fill holes in boundary mask"
    )
    parser.add_argument(
        "--tacs-threshold",
        type=float,
        default=30.0,
        help="PS-TACS threshold in degrees from normal (default: 30)",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=10,
        help="PS-TACS contour smoothing window (default: 10)",
    )
    parser.add_argument(
        "--output-dir", help="Directory to save detailed results (perpendicularity mode)"
    )

    args = parser.parse_args()

    # Resolve fill_holes flag
    if args.no_fill_holes:
        args.fill_holes = False

    try:
        if args.mode == "perpendicularity":
            result = run_perpendicularity(args)
        elif args.mode == "filter":
            result = run_filter(args)
        else:
            result = run_analysis(args)
        # Output JSON to stdout
        print(json.dumps(result))
    except Exception as e:
        # Output error as JSON
        print(json.dumps({"error": str(e)}), file=sys.stdout)
        sys.exit(1)


def run_analysis(args):
    """Run the analysis and return a JSON-serializable dict."""
    from ppm_library.analysis.region_analysis import (
        analyze_region,
    )
    from ppm_library.calibration.radial import RadialCalibrationResult

    # Load sum image
    sum_image = _load_image(args.sum)

    # Load calibration
    calibration = RadialCalibrationResult.load(args.calibration)

    # Load foreground mask or biref image (optional, mutually exclusive)
    foreground_mask = _load_foreground_mask(args)
    biref_image = None
    if foreground_mask is None and args.biref:
        biref_path = Path(args.biref)
        if biref_path.exists():
            biref_image = imread(str(biref_path))

    # Load ROI mask (optional)
    roi_mask = None
    if args.roi_mask:
        roi_path = Path(args.roi_mask)
        if roi_path.exists():
            roi_mask = imread(str(roi_path))
            if roi_mask.ndim == 3:
                roi_mask = roi_mask[:, :, 0]
            roi_mask = roi_mask > 0

    # Run analysis
    result = analyze_region(
        rgb_array=sum_image,
        calibration=calibration,
        biref_array=biref_image,
        biref_threshold=args.biref_threshold,
        saturation_threshold=args.saturation_threshold,
        value_threshold=args.value_threshold,
        histogram_bins=args.bins,
        foreground_mask=foreground_mask,
    )

    # Apply ROI mask if provided (restrict to annotation shape)
    if roi_mask is not None and roi_mask.shape == result["mask"].shape:
        result["mask"] = result["mask"] & roi_mask
        # Recompute stats with ROI-restricted mask
        from ppm_library.analysis.region_analysis import (
            compute_angle_histogram,
            compute_circular_statistics,
        )

        result["histogram"] = compute_angle_histogram(
            result["angles"], result["mask"], bins=args.bins
        )
        result["stats"] = compute_circular_statistics(result["angles"], result["mask"])

    # Convert to JSON-serializable format
    return {
        "histogram_counts": result["histogram"]["counts"].tolist(),
        "histogram_bin_edges": result["histogram"]["bin_edges"].tolist(),
        "histogram_bin_centers": result["histogram"]["bin_centers"].tolist(),
        "n_histogram_pixels": result["histogram"]["n_pixels"],
        "circular_mean": _safe_float(result["stats"]["circular_mean"]),
        "circular_std": _safe_float(result["stats"]["circular_std"]),
        "resultant_length": _safe_float(result["stats"]["resultant_length"]),
        "n_pixels": result["stats"]["n_pixels"],
        "arithmetic_mean": _safe_float(result["stats"]["arithmetic_mean"]),
        "arithmetic_std": _safe_float(result["stats"]["arithmetic_std"]),
    }


def run_filter(args):
    """Run angle range filtering and return pixel counts."""
    from ppm_library.analysis.region_analysis import (
        compute_angles_from_rgb,
        filter_angles_by_range,
    )
    from ppm_library.calibration.radial import RadialCalibrationResult

    sum_image = _load_image(args.sum)
    calibration = RadialCalibrationResult.load(args.calibration)

    angle_result = compute_angles_from_rgb(
        sum_image,
        calibration,
        saturation_threshold=args.saturation_threshold,
        value_threshold=args.value_threshold,
    )

    angles = angle_result["angles"]
    mask = angle_result["valid_mask"]

    # Apply ROI mask if provided
    if args.roi_mask:
        roi_path = Path(args.roi_mask)
        if roi_path.exists():
            roi_mask = imread(str(roi_path))
            if roi_mask.ndim == 3:
                roi_mask = roi_mask[:, :, 0]
            roi_mask = roi_mask > 0
            if roi_mask.shape == mask.shape:
                mask = mask & roi_mask

    # Apply foreground mask (from pixel classifier) or biref mask
    foreground_mask = _load_foreground_mask(args)
    if foreground_mask is not None:
        if foreground_mask.shape == mask.shape:
            mask = mask & foreground_mask
    elif args.biref:
        biref_path = Path(args.biref)
        if biref_path.exists():
            from ppm_library.analysis.region_analysis import compute_ppm_positive_mask

            biref_image = imread(str(biref_path))
            biref_mask = compute_ppm_positive_mask(biref_image, args.biref_threshold)
            if biref_mask.shape == mask.shape:
                mask = mask & biref_mask

    filter_result = filter_angles_by_range(angles, mask, args.angle_low, args.angle_high)

    n_valid = int(np.sum(mask))
    n_in_range = filter_result["n_in_range"]

    return {
        "angle_low": args.angle_low,
        "angle_high": args.angle_high,
        "n_in_range": n_in_range,
        "n_valid": n_valid,
        "fraction_in_range": n_in_range / n_valid if n_valid > 0 else 0.0,
    }


def run_perpendicularity(args):
    """Run surface perpendicularity analysis and return JSON-serializable dict."""
    from ppm_library.analysis.surface_analysis import (
        analyze_perpendicularity,
        rasterize_geojson_to_mask,
    )
    from ppm_library.calibration.radial import RadialCalibrationResult

    if not args.boundary:
        raise ValueError("--boundary (GeoJSON path) is required for perpendicularity mode")
    if not args.pixel_size_um:
        raise ValueError("--pixel-size-um is required for perpendicularity mode")

    sum_image = _load_image(args.sum)
    calibration = RadialCalibrationResult.load(args.calibration)

    # Load boundary mask from GeoJSON
    h, w = sum_image.shape[:2]
    boundary_mask = rasterize_geojson_to_mask(
        args.boundary, width=w, height=h, fill_holes=args.fill_holes
    )

    # Load foreground mask or biref image (optional, mutually exclusive)
    foreground_mask = _load_foreground_mask(args)
    biref_image = None
    if foreground_mask is None and args.biref:
        biref_path = Path(args.biref)
        if biref_path.exists():
            biref_image = imread(str(biref_path))

    result = analyze_perpendicularity(
        rgb_array=sum_image,
        calibration=calibration,
        boundary_mask=boundary_mask,
        dilation_um=args.dilation_um,
        pixel_size_um=args.pixel_size_um,
        mode=args.zone_mode,
        fill_holes=args.fill_holes,
        tacs_threshold_deg=args.tacs_threshold,
        smoothing_window=args.smoothing_window,
        biref_array=biref_image,
        biref_threshold=args.biref_threshold,
        saturation_threshold=args.saturation_threshold,
        value_threshold=args.value_threshold,
        foreground_mask=foreground_mask,
    )

    # Save detailed results if output dir provided
    if args.output_dir:
        _save_perpendicularity_details(result, args.output_dir)

    # Build JSON-serializable output
    output = {
        "dilation_px": result["dilation_px"],
        "pixel_size_um": result["pixel_size_um"],
        "contour_length_um": result["contour_length_um"],
        "n_contours": result["n_contours"],
    }

    # Simple results
    simple = result["simple"]
    output["simple"] = {
        "mean_deviation_deg": _safe_float(simple["mean_deviation_deg"]),
        "std_deviation_deg": _safe_float(simple["std_deviation_deg"]),
        "histogram_10deg": simple["histogram_10deg"],
        "histogram_3way": simple["histogram_3way"],
        "n_valid_pixels": simple["n_valid_pixels"],
    }

    # PS-TACS results
    pstacs = result.get("pstacs")
    if pstacs is not None:
        output["pstacs"] = {
            "pct_tacs2": pstacs["pct_tacs2"],
            "pct_tacs3": pstacs["pct_tacs3"],
            "n_tacs3_clusters": pstacs["n_tacs3_clusters"],
            "contour_length_px": pstacs["contour_length_px"],
            "tacs_threshold_deg": pstacs["tacs_threshold_deg"],
            # Per-contour-pixel data for border visualization
            "contour_points": pstacs["contour_points"].tolist(),
            "contour_tacs_class": pstacs["contour_tacs_class"].tolist(),
        }
    else:
        output["pstacs"] = None

    return output


def _save_perpendicularity_details(result, output_dir):
    """Save detailed perpendicularity results (arrays, contour data) to disk."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save simple deviation angles as numpy
    simple = result["simple"]
    if simple["deviation_angles"] is not None:
        np.save(str(out_path / "deviation_angles.npy"), simple["deviation_angles"])

    # Save PS-TACS contour data
    pstacs = result.get("pstacs")
    if pstacs is not None:
        np.savez(
            str(out_path / "pstacs_contour.npz"),
            contour_points=pstacs["contour_points"],
            contour_scores_raw=pstacs["contour_scores_raw"],
            contour_scores_smoothed=pstacs["contour_scores_smoothed"],
            contour_tacs_class=pstacs["contour_tacs_class"],
        )


def _load_foreground_mask(args):
    """Load external foreground mask if --foreground-mask was provided.

    Returns a boolean numpy array (H, W) or None.
    """
    if not getattr(args, "foreground_mask", None):
        return None
    fg_path = Path(args.foreground_mask)
    if not fg_path.exists():
        return None
    fg = imread(str(fg_path))
    if fg.ndim == 3:
        fg = fg[:, :, 0]
    return fg > 0


def _load_image(path):
    """Load image, ensuring RGB uint8 output."""
    img = imread(str(path))
    if img.ndim == 2:
        # Grayscale -> RGB
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:
        # RGBA -> RGB
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _safe_float(val):
    """Convert to float, replacing NaN with null for JSON."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return float(val)


if __name__ == "__main__":
    main()
