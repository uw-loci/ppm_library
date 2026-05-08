#!/usr/bin/env python3
"""
Create a synthetic PPM calibration phantom image.

This creates an image similar to a real calibration slide with:
- A sunburst pattern with radial spokes at different orientations
- Each spoke colored according to its orientation (full rainbow over 180 deg)
- Opposite spokes (180 deg apart) have the same color (same fiber orientation)
- Optional test gratings on the right side
"""

import numpy as np
from PIL import Image
from skimage import color
import cv2


def create_calibration_phantom(
    width: int = 924,
    height: int = 532,
    n_spokes: int = 16,
    spoke_length: int = 140,
    spoke_width: int = 8,
    center_hole_radius: int = 6,
    background_color: tuple = (60, 60, 60),  # Neutral gray (saturation=0)
    include_gratings: bool = True,
    saturation: float = 0.9,
    value: float = 0.9,
    paddle_shape: bool = True,
) -> np.ndarray:
    """Create a synthetic PPM calibration phantom.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        n_spokes: Number of unique orientations in the sunburst (default 16, which gives
                  17 spokes from horizontal to horizontal INCLUDING both endpoints).
                  Each orientation appears twice (opposite directions) = 2*n_spokes total.
        spoke_length: Length of each spoke from center
        spoke_width: Width of each spoke (at the outer end for paddle shape)
        center_hole_radius: Radius of the center hole (default 6 for ~12px diameter at 20x)
        background_color: RGB tuple for background
        include_gratings: Whether to include test grating patterns
        saturation: HSV saturation for spoke colors (0-1)
        value: HSV value/brightness for spoke colors (0-1)
        paddle_shape: If True, spokes are paddle-shaped (wide outer, narrow inner)

    Returns:
        RGB image as numpy array (H, W, 3), uint8
    """
    # Create image with background
    image = np.full((height, width, 3), background_color, dtype=np.uint8)

    # Sunburst center position (left side of image)
    cx = width // 4
    cy = height // 2

    # Draw spokes - each orientation appears twice (opposite directions)
    # n_spokes orientations over 180 deg, so 2*n_spokes total spokes over 360 deg
    for i in range(n_spokes * 2):
        # Angle in the full 360 deg circle
        angle_360 = i * 360.0 / (n_spokes * 2)

        # Orientation angle (0-180 deg) - opposite spokes have same orientation
        orientation = angle_360 % 180.0

        # Map orientation (0-180 deg) to hue (0-1)
        # Full rainbow spread over 180 degrees
        hue = orientation / 180.0

        # Convert HSV to RGB
        hsv_color = np.array([[[hue, saturation, value]]], dtype=np.float32)
        rgb_color = color.hsv2rgb(hsv_color)[0, 0]
        rgb_tuple = tuple(int(c * 255) for c in rgb_color)

        angle_rad = np.radians(angle_360)

        if paddle_shape:
            # Draw paddle-shaped spoke as a filled polygon
            # Tapers from thin at center to wide at outer edge
            # - Outer third: full width
            # - Middle third: half width
            # - Inner third: thin (1/4 width)

            third_length = (spoke_length - center_hole_radius) / 3

            # Define radii for each section boundary
            r_inner = center_hole_radius
            r_mid1 = center_hole_radius + third_length
            r_mid2 = center_hole_radius + 2 * third_length
            r_outer = spoke_length

            # Define widths at each boundary (half-width for each side)
            w_inner = 0.5  # Very thin at center
            w_mid1 = spoke_width / 8  # 1/4 width at 1/3
            w_mid2 = spoke_width / 4  # 1/2 width at 2/3
            w_outer = spoke_width / 2  # Full width at outer

            # Calculate perpendicular direction for width
            perp_rad = angle_rad + np.pi / 2

            # Build polygon points (one side out, other side back)
            points = []

            # Inner point (center side) - nearly a point
            points.append(
                [
                    cx + r_inner * np.cos(angle_rad) + w_inner * np.cos(perp_rad),
                    cy - r_inner * np.sin(angle_rad) - w_inner * np.sin(perp_rad),
                ]
            )

            # Mid1 point (1/3 out) - left side
            points.append(
                [
                    cx + r_mid1 * np.cos(angle_rad) + w_mid1 * np.cos(perp_rad),
                    cy - r_mid1 * np.sin(angle_rad) - w_mid1 * np.sin(perp_rad),
                ]
            )

            # Mid2 point (2/3 out) - left side
            points.append(
                [
                    cx + r_mid2 * np.cos(angle_rad) + w_mid2 * np.cos(perp_rad),
                    cy - r_mid2 * np.sin(angle_rad) - w_mid2 * np.sin(perp_rad),
                ]
            )

            # Outer point - left side
            points.append(
                [
                    cx + r_outer * np.cos(angle_rad) + w_outer * np.cos(perp_rad),
                    cy - r_outer * np.sin(angle_rad) - w_outer * np.sin(perp_rad),
                ]
            )

            # Outer point - right side
            points.append(
                [
                    cx + r_outer * np.cos(angle_rad) - w_outer * np.cos(perp_rad),
                    cy - r_outer * np.sin(angle_rad) + w_outer * np.sin(perp_rad),
                ]
            )

            # Mid2 point - right side
            points.append(
                [
                    cx + r_mid2 * np.cos(angle_rad) - w_mid2 * np.cos(perp_rad),
                    cy - r_mid2 * np.sin(angle_rad) + w_mid2 * np.sin(perp_rad),
                ]
            )

            # Mid1 point - right side
            points.append(
                [
                    cx + r_mid1 * np.cos(angle_rad) - w_mid1 * np.cos(perp_rad),
                    cy - r_mid1 * np.sin(angle_rad) + w_mid1 * np.sin(perp_rad),
                ]
            )

            # Inner point - right side
            points.append(
                [
                    cx + r_inner * np.cos(angle_rad) - w_inner * np.cos(perp_rad),
                    cy - r_inner * np.sin(angle_rad) + w_inner * np.sin(perp_rad),
                ]
            )

            # Draw filled polygon
            pts = np.array(points, dtype=np.int32)
            cv2.fillPoly(image, [pts], rgb_tuple)

        else:
            # Simple uniform-width spoke
            x1 = int(cx + center_hole_radius * np.cos(angle_rad))
            y1 = int(cy - center_hole_radius * np.sin(angle_rad))
            x2 = int(cx + spoke_length * np.cos(angle_rad))
            y2 = int(cy - spoke_length * np.sin(angle_rad))
            cv2.line(image, (x1, y1), (x2, y2), rgb_tuple, thickness=spoke_width)

    # Draw center hole (dark)
    cv2.circle(image, (cx, cy), center_hole_radius, background_color, -1)

    # Add test gratings on the right side
    if include_gratings:
        _add_gratings(image, cx, cy, spoke_length, background_color, saturation, value)

    return image


def _add_gratings(
    image: np.ndarray,
    sunburst_cx: int,
    sunburst_cy: int,
    spoke_length: int,
    background_color: tuple,
    saturation: float,
    value: float,
):
    """Add numbered test grating patterns to the right side of the image."""
    height, width = image.shape[:2]

    # Grating parameters
    grating_width = 60
    grating_height = 80
    grating_spacing = 30

    # Start position for gratings (right of sunburst)
    start_x = sunburst_cx + spoke_length + 80
    grating_y = sunburst_cy - grating_height // 2

    # Line spacings for each grating (in pixels)
    spacings = [12, 8, 4]  # Grating 1, 2, 3

    # Use a single orientation angle (e.g., 90 deg = vertical lines)
    # All gratings at same orientation so they have same hue
    orientation = 90.0
    hue = orientation / 180.0
    hsv_color = np.array([[[hue, saturation, value]]], dtype=np.float32)
    rgb_color = color.hsv2rgb(hsv_color)[0, 0]
    rgb_tuple = tuple(int(c * 255) for c in rgb_color)

    for idx, spacing in enumerate(spacings):
        grating_x = start_x + idx * (grating_width + grating_spacing)

        # Check bounds
        if grating_x + grating_width > width - 10:
            break

        # Draw vertical lines for grating
        for line_x in range(grating_x, grating_x + grating_width, spacing):
            cv2.line(
                image,
                (line_x, grating_y),
                (line_x, grating_y + grating_height),
                rgb_tuple,
                thickness=max(1, spacing // 3),
            )

        # Add number label above grating
        label = str(3 - idx)  # Labels 3, 2, 1
        label_y = grating_y - 15

        # Simple text rendering using cv2
        cv2.putText(
            image,
            label,
            (grating_x + grating_width // 2 - 10, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            rgb_tuple,
            2,
        )


def create_simple_sunburst(
    size: int = 512,
    n_spokes: int = 18,
    spoke_length: int = 200,
    spoke_width: int = 12,
) -> np.ndarray:
    """Create a simple square sunburst image for testing.

    Args:
        size: Image size (square)
        n_spokes: Number of unique orientations (total spokes = 2 * n_spokes)
        spoke_length: Length of each spoke
        spoke_width: Width of each spoke

    Returns:
        RGB image as numpy array
    """
    return create_calibration_phantom(
        width=size,
        height=size,
        n_spokes=n_spokes,
        spoke_length=spoke_length,
        spoke_width=spoke_width,
        include_gratings=False,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create PPM calibration phantom")
    parser.add_argument("-o", "--output", default="phantom.png", help="Output file path")
    parser.add_argument("--width", type=int, default=924, help="Image width (default: 924 for 2x)")
    parser.add_argument(
        "--height", type=int, default=532, help="Image height (default: 532 for 2x)"
    )
    parser.add_argument("--spokes", type=int, default=36, help="Number of spoke orientations")
    parser.add_argument("--no-gratings", action="store_true", help="Omit test gratings")
    parser.add_argument("--simple", action="store_true", help="Create simple square sunburst")

    args = parser.parse_args()

    if args.simple:
        image = create_simple_sunburst()
    else:
        image = create_calibration_phantom(
            width=args.width,
            height=args.height,
            n_spokes=args.spokes,
            include_gratings=not args.no_gratings,
        )

    # Save image
    Image.fromarray(image).save(args.output)
    print(f"Saved phantom to: {args.output}")
    print(f"Size: {image.shape[1]}x{image.shape[0]}")
    print(f"Spokes: {args.spokes} orientations x 2 = {args.spokes * 2} total spokes")
