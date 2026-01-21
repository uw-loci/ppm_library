#!/usr/bin/env python3
"""
Example: Sunburst Calibration for PPM Images

This example demonstrates how to use the SunburstCalibrator to create
a hue-to-angle regression model from a calibration slide image.

Usage:
    python calibration_example.py /path/to/calibration_slide.tif
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from ppm_library import SunburstCalibrator
from ppm_library.calibration.sunburst import CalibrationResult, calibrate_from_image


def run_calibration(image_path: str, n_rectangles: int = 16, output_path: str = None):
    """Run calibration on a sunburst slide image.

    Args:
        image_path: Path to the calibration slide image
        n_rectangles: Expected number of calibration rectangles
        output_path: Optional path to save calibration results
    """
    print(f"Loading calibration image: {image_path}")

    # Create calibrator with custom settings if needed
    calibrator = SunburstCalibrator(
        n_expected_rectangles=n_rectangles,
        min_area=100,  # Minimum pixel area for detected regions
        saturation_threshold=0.1,  # Threshold for foreground detection
        value_threshold=0.1,
    )

    # Run calibration with debug visualization
    print("Running calibration...")
    result = calibrator.calibrate(image_path, debug_plot=True)

    # Print results
    print("\n" + "=" * 50)
    print("CALIBRATION RESULTS")
    print("=" * 50)
    print(f"Number of rectangles detected: {len(result.rectangles)}")
    print(f"R-squared: {result.r_squared:.6f}")
    print(f"\nRegression (hue [0-1] to angle [0-180]):")
    print(f"  angle = {result.inv_slope:.4f} * hue + {result.inv_intercept:.4f}")
    print(f"\nInverse (angle to hue):")
    print(f"  hue = {result.slope:.6f} * angle + {result.intercept:.6f}")

    # Print per-rectangle data
    print("\nPer-rectangle calibration data:")
    print("-" * 50)
    print(f"{'#':<4} {'Angle (deg)':<12} {'Hue':<8} {'RGB Mode':<20} {'Area':<8}")
    print("-" * 50)
    for i, rect in enumerate(result.rectangles):
        print(
            f"{i+1:<4} {rect.angle:<12.2f} {rect.hue_mode:<8.4f} "
            f"{str(rect.rgb_mode):<20} {rect.area:<8}"
        )

    # Save if output path provided
    if output_path:
        result.save(output_path)
        print(f"\nCalibration saved to: {output_path}")

    # Demonstrate usage
    print("\n" + "=" * 50)
    print("EXAMPLE CONVERSIONS")
    print("=" * 50)

    test_hues = np.array([0.0, 0.125, 0.25, 0.375, 0.5])
    test_angles = result.hue_to_angle(test_hues)

    print("Hue -> Angle conversions:")
    for h, a in zip(test_hues, test_angles):
        print(f"  hue={h:.3f} -> angle={a:.1f} degrees")

    return result


def demo_with_synthetic():
    """Demonstrate calibration with a synthetic PPM calibration phantom.

    Creates a realistic sunburst pattern with:
    - 17 orientations (34 total spokes for opposite directions)
    - Paddle-shaped spokes (wide at outer, narrow at center)
    - Full rainbow hue spread over 180 degrees
    - Test gratings on the right side
    """
    import tempfile
    from PIL import Image

    # Import the phantom generator (same directory)
    import sys
    from pathlib import Path as P
    sys.path.insert(0, str(P(__file__).parent))
    from create_phantom import create_calibration_phantom

    print("Creating synthetic PPM calibration phantom...")
    print("  - 17 orientations x 2 = 34 spokes")
    print("  - Paddle-shaped spokes (tapered)")
    print("  - Full 180 deg rainbow hue spread")
    print("  - 2x resolution (924x532, simulating 20x objective)")
    print()

    # Create the phantom
    image = create_calibration_phantom(
        width=924,
        height=532,
        n_spokes=17,
        spoke_length=140,
        spoke_width=8,
        paddle_shape=True,
        include_gratings=True,
    )

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        temp_path = f.name

    Image.fromarray(image).save(temp_path)
    print(f"Synthetic phantom saved to: {temp_path}")

    # Run calibration (expect 34 spokes, which merge to ~17 after duplicate removal)
    result = run_calibration(temp_path, n_rectangles=34)

    # Clean up
    Path(temp_path).unlink()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run sunburst calibration on a PPM calibration slide"
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        help="Path to calibration slide image (.tif or .ome.tif)",
    )
    parser.add_argument(
        "-n",
        "--n-rectangles",
        type=int,
        default=16,
        help="Expected number of rectangles (default: 16)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output path for calibration file (.npz)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with synthetic image",
    )

    args = parser.parse_args()

    if args.demo:
        demo_with_synthetic()
    elif args.image_path:
        run_calibration(args.image_path, args.n_rectangles, args.output)
    else:
        print("Error: Please provide an image path or use --demo flag")
        parser.print_help()


if __name__ == "__main__":
    main()
