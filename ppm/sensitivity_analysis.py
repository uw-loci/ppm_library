#!/usr/bin/env python3
"""
PPM Rotation Sensitivity Analysis

This script analyzes the impact of angular deviations on Polarized Phase Microscopy (PPM)
image quality and birefringence calculations. It provides quantitative metrics for how
small angular errors affect:
1. Direct image differences at various angles
2. Birefringence image quality and pattern stability
3. Retardance and orientation calculations

Usage:
    python ppm_rotation_sensitivity_analysis.py <image_directory> [options]
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from typing import Dict, List, Tuple, Optional
import pandas as pd
from scipy import ndimage
from skimage import metrics, io
import json
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class PPMRotationAnalyzer:
    """Analyzes rotation sensitivity in PPM imaging."""

    def __init__(self, base_path: Path, output_dir: Path = None):
        """
        Initialize the analyzer.

        Args:
            base_path: Directory containing PPM images at different angles
            output_dir: Directory for output results (creates timestamped folder if None)
        """
        self.base_path = Path(base_path)
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(f"ppm_rotation_analysis_{timestamp}")
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Standard PPM angles (in degrees)
        self.standard_angles = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91]

        # Store loaded images
        self.images = {}
        self.image_files = {}  # angle -> filename mapping
        self.image_shape = None

    def load_images(self, angle_pattern: str = "*_{angle}deg*.tif") -> Dict[float, np.ndarray]:
        """
        Load PPM images at different angles.

        Args:
            angle_pattern: Filename pattern with {angle} placeholder

        Returns:
            Dictionary mapping angles to image arrays
        """
        print("Loading PPM images...")

        # Try to find images with various naming conventions
        patterns = [
            angle_pattern,
            "*_{angle}degree*.tif",
            "*_angle{angle}*.tif",
            "*_{angle}*.tif"
        ]

        for angle in self.standard_angles:
            for pattern in patterns:
                search_pattern = pattern.replace("{angle}", str(angle))
                files = list(self.base_path.glob(search_pattern))
                if files:
                    # Load the first matching file
                    img = cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        # Convert to float32 for calculations
                        img = img.astype(np.float32)
                        if len(img.shape) == 3:
                            # Convert to grayscale if color
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        self.images[angle] = img
                        if self.image_shape is None:
                            self.image_shape = img.shape
                        print(f"  Loaded {angle} degree image: {files[0].name}")
                        break

        print(f"Loaded {len(self.images)} images")
        return self.images

    def load_deviation_images(self) -> Dict[float, np.ndarray]:
        """
        Load PPM deviation test images in BGACQUIRE format: {angle}.tif (e.g., 45.05.tif, 44.0.tif)

        This is the format used by qp_acquisition.simple_background_collection().

        Returns:
            Dictionary mapping actual angles to image arrays
        """
        import re
        print("Loading PPM deviation images...")

        # Track which file each angle came from
        self.image_files = {}  # angle -> filename

        # Load BGACQUIRE format: {angle}.tif (e.g., 45.05.tif)
        angle_pattern = re.compile(r'^(-?\d+\.?\d*)\.tif$', re.IGNORECASE)

        for file_path in self.base_path.glob("*.tif"):
            match = angle_pattern.match(file_path.name)
            if match:
                try:
                    angle = float(match.group(1))
                    img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        img = img.astype(np.float32)
                        if len(img.shape) == 3:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        self.images[angle] = img
                        self.image_files[angle] = file_path.name  # Track filename
                        if self.image_shape is None:
                            self.image_shape = img.shape
                        print(f"  Loaded {angle:.2f} deg image: {file_path.name}")
                except ValueError:
                    pass  # Not a valid angle filename

        print(f"Loaded {len(self.images)} deviation images")
        return self.images

    def simulate_angular_deviations(self, reference_angle: float = 7.0,
                                   deviations: List[float] = [0.1, 0.2, 0.3, 0.5, 1.0]) -> Dict:
        """
        Simulate images at slightly deviated angles through interpolation.

        Args:
            reference_angle: Reference angle in degrees
            deviations: List of angular deviations to test

        Returns:
            Dictionary of simulated images at deviated angles
        """
        if reference_angle not in self.images:
            print(f"Warning: Reference angle {reference_angle} not found in loaded images")
            return {}

        simulated = {}
        ref_image = self.images[reference_angle]

        # Find nearest neighbor angles for interpolation
        angles = sorted(self.images.keys())
        ref_idx = angles.index(reference_angle)

        for dev in deviations:
            target_angle = reference_angle + dev

            # Simple rotation simulation (small angle approximation)
            # In reality, this would be more complex due to polarizer physics
            rotation_matrix = cv2.getRotationMatrix2D(
                (ref_image.shape[1]//2, ref_image.shape[0]//2),
                dev, 1.0
            )
            rotated = cv2.warpAffine(ref_image, rotation_matrix,
                                    (ref_image.shape[1], ref_image.shape[0]))

            # Also try interpolation between neighboring angles if available
            if ref_idx < len(angles) - 1:
                next_angle = angles[ref_idx + 1]
                if next_angle in self.images:
                    # Linear interpolation weight
                    weight = dev / (next_angle - reference_angle)
                    interpolated = (1 - weight) * ref_image + weight * self.images[next_angle]
                    simulated[target_angle] = interpolated
                else:
                    simulated[target_angle] = rotated
            else:
                simulated[target_angle] = rotated

        return simulated

    def compute_image_differences(self, reference_angle: float = 7.0) -> pd.DataFrame:
        """
        Compute various difference metrics between reference and deviated images.

        Args:
            reference_angle: Reference angle for comparison

        Returns:
            DataFrame with difference metrics
        """
        results = []

        if reference_angle not in self.images:
            print(f"Error: Reference angle {reference_angle} not found")
            return pd.DataFrame()

        ref_image = self.images[reference_angle]

        # Use FIXED normalization (full bit depth) for consistent comparison
        full_range = 65535.0 if ref_image.max() > 255 else 255.0

        # Test against actual angles
        for angle, img in self.images.items():
            if angle == reference_angle:
                continue

            # Compute various metrics
            mae = np.mean(np.abs(img - ref_image))
            mse = np.mean((img - ref_image) ** 2)
            rmse = np.sqrt(mse)

            # Normalized metrics (relative to FULL BIT DEPTH, not image range)
            nmae = mae / full_range
            nrmse = rmse / full_range

            # Structural similarity - use full range
            ssim = metrics.structural_similarity(ref_image, img, data_range=full_range)

            # Peak signal-to-noise ratio - use full range
            psnr = metrics.peak_signal_noise_ratio(ref_image, img, data_range=full_range)

            # Pearson correlation
            correlation = np.corrcoef(ref_image.flatten(), img.flatten())[0, 1]

            results.append({
                'reference_angle': reference_angle,
                'comparison_angle': angle,
                'angular_difference': abs(angle - reference_angle),
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'normalized_mae': nmae,
                'normalized_rmse': nrmse,
                'ssim': ssim,
                'psnr': psnr,
                'correlation': correlation,
                'max_pixel_diff': float(np.max(np.abs(img - ref_image))),
                'std_diff': np.std(img - ref_image)
            })

        return pd.DataFrame(results)

    def compute_adjacent_differences(self) -> pd.DataFrame:
        """
        Compare each angle to its immediate neighbors to measure fine sensitivity.

        This answers: "How much does the image change between 7.00 and 7.05 degrees?"

        Returns:
            DataFrame with columns: angle1, angle2, delta_deg, mae, pct_change, ssim
        """
        results = []
        sorted_angles = sorted(self.images.keys())

        print(f"\nComputing adjacent angle differences for {len(sorted_angles)} angles...")

        for i in range(len(sorted_angles) - 1):
            angle1 = sorted_angles[i]
            angle2 = sorted_angles[i + 1]
            delta = angle2 - angle1

            img1 = self.images[angle1]
            img2 = self.images[angle2]

            # Compute metrics
            mae = float(np.mean(np.abs(img2 - img1)))

            # Use FIXED normalization (full bit depth) for consistent comparison
            # NOT image range which varies with angle
            full_range = 65535.0 if img1.max() > 255 else 255.0
            pct_change = (mae / full_range) * 100

            # SSIM for structural similarity
            try:
                ssim = metrics.structural_similarity(img1, img2, data_range=full_range)
            except Exception:
                ssim = np.nan

            results.append({
                'angle1': angle1,
                'angle2': angle2,
                'delta_deg': delta,
                'mae': mae,
                'pct_change': pct_change,
                'ssim': ssim,
                'median_intensity_1': float(np.median(img1)),
                'median_intensity_2': float(np.median(img2)),
                'intensity_change': float(np.median(img2) - np.median(img1))
            })

        return pd.DataFrame(results)

    def analyze_zero_rotation_baseline(self) -> pd.DataFrame:
        """
        Analyze zero-rotation baseline image pairs to establish measurement noise floor.

        Looks for image pairs named baseline_*_A.tif and baseline_*_B.tif in the
        base_path directory. These pairs were acquired at the SAME angle without
        any rotation, so differences should be essentially 0%.

        Returns:
            DataFrame with baseline pair comparison metrics
        """
        results = []

        # Find baseline image pairs
        baseline_files = list(self.base_path.glob("baseline_*_A.tif"))

        if not baseline_files:
            print("No baseline image pairs found (baseline_*_A.tif)")
            return pd.DataFrame()

        print(f"\nAnalyzing {len(baseline_files)} zero-rotation baseline pairs...")

        # ============================================================
        # SANITY CHECK: Compare first image to EXACT COPY of itself
        # This MUST show 0.0000% - if not, calculation is broken
        # ============================================================
        if baseline_files:
            test_file = baseline_files[0]
            print(f"\n  SANITY CHECK: Comparing {test_file.name} to EXACT COPY...")
            try:
                test_img = io.imread(test_file)
                print(f"    Raw image dtype: {test_img.dtype}, shape: {test_img.shape}")
                print(f"    Raw image range: [{test_img.min()}, {test_img.max()}]")

                # Make exact copy (not a view)
                test_copy = test_img.copy()

                # Convert same way as real comparison
                if test_img.ndim == 3:
                    test_gray = np.mean(test_img, axis=2)
                    copy_gray = np.mean(test_copy, axis=2)
                else:
                    test_gray = test_img
                    copy_gray = test_copy

                test_f64 = test_gray.astype(np.float64)
                copy_f64 = copy_gray.astype(np.float64)

                # Compute difference - MUST be exactly 0
                diff = test_f64 - copy_f64
                mae_self = float(np.mean(np.abs(diff)))
                max_diff_self = float(np.max(np.abs(diff)))
                img_range = max(test_f64.max() - test_f64.min(), 1)
                pct_self = (mae_self / img_range) * 100

                print(f"    After conversion dtype: {test_f64.dtype}")
                print(f"    MAE (self vs copy): {mae_self:.10f}")
                print(f"    Max pixel diff: {max_diff_self:.10f}")
                print(f"    Percent change: {pct_self:.10f}%")

                if mae_self == 0.0 and max_diff_self == 0.0:
                    print(f"    --> PASS: Self-comparison is exactly 0.0")
                else:
                    print(f"    --> FAIL: Self-comparison should be 0.0!")

                # Add as first result for visibility
                results.append({
                    'angle': -999.0,  # Special marker for self-test
                    'pair': 0,
                    'mae': mae_self,
                    'pct_change': pct_self,
                    'median_intensity_a': float(np.median(test_f64)),
                    'median_intensity_b': float(np.median(copy_f64)),
                    'median_pct_change': 0.0,
                    'ssim': 1.0,
                    'max_pixel_diff': max_diff_self,
                    'std_diff': 0.0,
                    'test_type': 'SELF_COPY_SANITY_CHECK'
                })

            except Exception as e:
                print(f"    Sanity check failed: {e}")

        print(f"\n  Analyzing {len(baseline_files)} actual baseline pairs...")

        for file_a in baseline_files:
            # Find matching B file
            file_b = file_a.parent / file_a.name.replace("_A.tif", "_B.tif")

            if not file_b.exists():
                print(f"  Warning: Missing pair for {file_a.name}")
                continue

            # Load images - use plugin=None to avoid any compression issues
            try:
                img_a = io.imread(file_a)
                img_b = io.imread(file_b)

                # Report raw data info
                print(f"  {file_a.name}: dtype={img_a.dtype}, shape={img_a.shape}")

                # Convert to grayscale if RGB
                if img_a.ndim == 3:
                    img_a = np.mean(img_a, axis=2)
                if img_b.ndim == 3:
                    img_b = np.mean(img_b, axis=2)

                # Ensure same type for comparison
                img_a = img_a.astype(np.float64)
                img_b = img_b.astype(np.float64)

                # Compute metrics
                mae = float(np.mean(np.abs(img_b - img_a)))

                # Use FIXED normalization (full bit depth) for consistent comparison
                full_range = 65535.0 if img_a.max() > 255 else 255.0
                pct_change = (mae / full_range) * 100

                # Compute median intensity change (relative to median, not range)
                median_a = float(np.median(img_a))
                median_b = float(np.median(img_b))
                median_pct_change = abs(median_b - median_a) / max(median_a, 1) * 100

                # SSIM - use full range for consistent comparison
                try:
                    ssim = metrics.structural_similarity(img_a, img_b, data_range=full_range)
                except Exception:
                    ssim = np.nan

                # Extract angle from filename (baseline_7.0deg_pair1_A.tif)
                parts = file_a.stem.split("_")
                angle = float(parts[1].replace("deg", ""))
                pair_num = int(parts[2].replace("pair", ""))

                results.append({
                    'angle': angle,
                    'pair': pair_num,
                    'mae': mae,
                    'pct_change': pct_change,
                    'median_intensity_a': median_a,
                    'median_intensity_b': median_b,
                    'median_pct_change': median_pct_change,
                    'ssim': ssim,
                    'max_pixel_diff': float(np.max(np.abs(img_b - img_a))),
                    'std_diff': float(np.std(img_b - img_a))
                })

                print(f"  {file_a.name}: {pct_change:.4f}% change (MAE={mae:.2f})")

            except Exception as e:
                print(f"  Error loading {file_a.name}: {e}")
                continue

        if results:
            df = pd.DataFrame(results)
            # Summary statistics
            print(f"\n  BASELINE SUMMARY (expected: ~0% change):")
            print(f"    Mean intensity change: {df['pct_change'].mean():.4f}%")
            print(f"    Max intensity change: {df['pct_change'].max():.4f}%")
            print(f"    Mean SSIM: {df['ssim'].mean():.6f}")
            return df
        else:
            return pd.DataFrame()

    def analyze_fine_sensitivity(self, base_angles: List[float] = None) -> Dict:
        """
        Analyze sensitivity to fine angular changes around specific base angles.

        Groups angles by proximity to base angles (e.g., 7, 0, -7, 90) and computes
        statistics for fine deviations within each group.

        Args:
            base_angles: List of nominal angles to analyze (default: [7, 0, -7, 90])

        Returns:
            Dictionary with sensitivity statistics per base angle
        """
        if base_angles is None:
            base_angles = [7, 0, -7, 90]

        results = {}
        sorted_angles = sorted(self.images.keys())

        for base in base_angles:
            # Find angles within +/- 2 degrees of base
            nearby = [a for a in sorted_angles if abs(a - base) <= 2.0]

            if len(nearby) < 2:
                continue

            print(f"\n=== Fine sensitivity analysis around {base} deg ===")
            print(f"Found {len(nearby)} angles: {[f'{a:.2f}' for a in nearby]}")

            # Compare each angle to the base angle (or nearest to base)
            base_actual = min(nearby, key=lambda x: abs(x - base))
            if base_actual not in self.images:
                print(f"  WARNING: Base angle {base_actual} not found in images!")
                continue

            base_img = self.images[base_actual]
            group_results = []

            # DIAGNOSTIC: Check if base image exists and print info
            print(f"  Using base angle: {base_actual:.2f} deg")
            print(f"  Base image shape: {base_img.shape}, dtype: {base_img.dtype}")

            # Determine normalization: use full bit depth (255 for 8-bit, 65535 for 16-bit)
            # NOT the image's dynamic range, which varies with angle
            if base_img.max() > 255:
                full_range = 65535.0  # 16-bit
            else:
                full_range = 255.0  # 8-bit

            # Also compute the base image's actual dynamic range for reporting
            base_range = float(base_img.max() - base_img.min())
            base_median = float(np.median(base_img))
            print(f"  Base image stats: median={base_median:.1f}, range={base_range:.1f}, max={base_img.max()}")

            for angle in nearby:
                if angle == base_actual:
                    continue

                img = self.images[angle]
                deviation = angle - base_actual

                # DIAGNOSTIC: Print median intensities to see if images are fundamentally different
                img_median = float(np.median(img))
                intensity_diff = img_median - base_median

                mae = float(np.mean(np.abs(img - base_img)))

                # Use FIXED normalization (full bit depth) for consistent comparison
                # This prevents flat images (like 90 deg) from showing inflated percentages
                pct_change = (mae / full_range) * 100

                # Also compute the "old" way for comparison (will be much higher for flat images)
                old_pct = (mae / max(base_range, 1)) * 100

                group_results.append({
                    'deviation': deviation,
                    'angle': angle,
                    'mae': mae,
                    'pct_change': pct_change,
                    'pct_change_old': old_pct,  # For debugging
                    'base_median': base_median,
                    'img_median': img_median
                })

                # Show detailed diagnostic info
                print(f"  {base_actual:.2f} -> {angle:.2f} (delta={deviation:+.2f}):")
                print(f"      Base median: {base_median:.1f}, This median: {img_median:.1f}, "
                      f"Intensity diff: {intensity_diff:+.1f}")
                print(f"      MAE={mae:.2f}, {pct_change:.3f}% (of {full_range:.0f})")
                if abs(intensity_diff) > 10:
                    print(f"      WARNING: Large intensity offset ({intensity_diff:+.1f}) suggests "
                          f"different exposure or acquisition conditions!")

            if group_results:
                # Compute sensitivity rate (% change per degree)
                deviations = [abs(r['deviation']) for r in group_results]
                pct_changes = [r['pct_change'] for r in group_results]

                # Linear fit for sensitivity rate
                if len(deviations) > 1:
                    slope = np.polyfit(deviations, pct_changes, 1)[0]
                else:
                    slope = pct_changes[0] / deviations[0] if deviations[0] != 0 else 0

                results[base] = {
                    'base_angle': base_actual,
                    'n_samples': len(group_results),
                    'sensitivity_pct_per_deg': slope,
                    'details': group_results
                }

                print(f"  --> Sensitivity: ~{slope:.2f}% intensity change per degree")

        return results

    def compute_birefringence(self, angles: List[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute birefringence (retardance and orientation) from PPM images.

        Args:
            angles: List of angles to use (defaults to all available)

        Returns:
            Tuple of (retardance, orientation) arrays
        """
        if angles is None:
            angles = sorted(self.images.keys())

        # Filter to available angles
        angles = [a for a in angles if a in self.images]

        if len(angles) < 3:
            print("Error: Need at least 3 angles for birefringence calculation")
            return None, None

        # Stack images
        image_stack = np.array([self.images[a] for a in angles])
        angles_rad = np.array(angles) * np.pi / 180

        # Fourier analysis for retardance and orientation
        # This is a simplified version - actual implementation depends on specific PPM setup
        n_pixels = image_stack.shape[1] * image_stack.shape[2]

        # Compute Fourier coefficients
        a0 = np.mean(image_stack, axis=0)
        a2 = np.zeros_like(a0)
        b2 = np.zeros_like(a0)

        for i, angle in enumerate(angles_rad):
            a2 += (2/len(angles)) * image_stack[i] * np.cos(2 * angle)
            b2 += (2/len(angles)) * image_stack[i] * np.sin(2 * angle)

        # Retardance (magnitude)
        retardance = np.sqrt(a2**2 + b2**2) / (a0 + 1e-10)

        # Orientation (phase)
        orientation = 0.5 * np.arctan2(b2, a2)

        return retardance, orientation

    def analyze_birefringence_sensitivity(self, deviations: List[float] = [0.1, 0.2, 0.5, 1.0]) -> pd.DataFrame:
        """
        Analyze how angular deviations affect birefringence calculations.

        Args:
            deviations: Angular deviations to test (in degrees)

        Returns:
            DataFrame with birefringence sensitivity metrics
        """
        results = []

        # Compute reference birefringence with all correct angles
        ref_retardance, ref_orientation = self.compute_birefringence()

        if ref_retardance is None:
            return pd.DataFrame()

        # Test with one angle deviated at a time
        base_angles = sorted(self.images.keys())

        for target_angle in [7, 14, 21]:  # Test key angles
            if target_angle not in base_angles:
                continue

            for deviation in deviations:
                # Create modified angle list
                modified_angles = base_angles.copy()
                idx = modified_angles.index(target_angle)

                # Simulate the deviated image
                simulated = self.simulate_angular_deviations(target_angle, [deviation])

                if target_angle + deviation in simulated:
                    # Replace the original image with simulated one
                    temp_img = self.images[target_angle]
                    self.images[target_angle] = simulated[target_angle + deviation]

                    # Compute birefringence with deviated angle
                    dev_retardance, dev_orientation = self.compute_birefringence()

                    # Restore original image
                    self.images[target_angle] = temp_img

                    if dev_retardance is not None:
                        # Compute differences
                        retardance_diff = dev_retardance - ref_retardance
                        orientation_diff = dev_orientation - ref_orientation

                        # Wrap orientation difference to [-pi, pi]
                        orientation_diff = np.angle(np.exp(1j * orientation_diff))

                        results.append({
                            'deviated_angle': target_angle,
                            'deviation': deviation,
                            'mean_retardance_error': np.mean(np.abs(retardance_diff)),
                            'max_retardance_error': np.max(np.abs(retardance_diff)),
                            'std_retardance_error': np.std(retardance_diff),
                            'mean_orientation_error_deg': np.mean(np.abs(orientation_diff)) * 180/np.pi,
                            'max_orientation_error_deg': np.max(np.abs(orientation_diff)) * 180/np.pi,
                            'std_orientation_error_deg': np.std(orientation_diff) * 180/np.pi,
                            'retardance_rmse': np.sqrt(np.mean(retardance_diff**2)),
                            'orientation_rmse_deg': np.sqrt(np.mean(orientation_diff**2)) * 180/np.pi
                        })

        return pd.DataFrame(results)

    def visualize_difference_maps(self, reference_angle: float = 7.0,
                                 comparison_angles: List[float] = None):
        """
        Create visual difference maps between reference and comparison images.

        Args:
            reference_angle: Reference angle
            comparison_angles: Angles to compare (defaults to small deviations)
        """
        if reference_angle not in self.images:
            print(f"Error: Reference angle {reference_angle} not found")
            return

        ref_image = self.images[reference_angle]

        if comparison_angles is None:
            # Use small deviations if available
            comparison_angles = [a for a in self.images.keys()
                               if a != reference_angle and abs(a - reference_angle) <= 7]

        n_comparisons = len(comparison_angles)
        if n_comparisons == 0:
            print("No comparison angles available")
            return

        fig, axes = plt.subplots(3, n_comparisons, figsize=(4*n_comparisons, 12))
        if n_comparisons == 1:
            axes = axes.reshape(-1, 1)

        for i, angle in enumerate(comparison_angles):
            if angle not in self.images:
                continue

            comp_image = self.images[angle]
            diff_image = comp_image - ref_image
            abs_diff = np.abs(diff_image)

            # Original comparison image
            im1 = axes[0, i].imshow(comp_image, cmap='gray')
            axes[0, i].set_title(f'{angle} degrees')
            axes[0, i].axis('off')
            plt.colorbar(im1, ax=axes[0, i], fraction=0.046)

            # Difference map
            vmax = np.percentile(abs_diff, 99)
            im2 = axes[1, i].imshow(diff_image, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            axes[1, i].set_title(f'Difference from {reference_angle} deg')
            axes[1, i].axis('off')
            plt.colorbar(im2, ax=axes[1, i], fraction=0.046)

            # Absolute difference map
            im3 = axes[2, i].imshow(abs_diff, cmap='hot')
            axes[2, i].set_title(f'Absolute difference')
            axes[2, i].axis('off')
            plt.colorbar(im3, ax=axes[2, i], fraction=0.046)

        plt.suptitle(f'PPM Image Differences (Reference: {reference_angle} degrees)', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'difference_maps_ref{reference_angle}.png', dpi=150)
        plt.close()  # Don't block - file is already saved

    def visualize_birefringence_comparison(self, angle_sets: List[List[float]] = None):
        """
        Compare birefringence calculations with different angle sets.

        Args:
            angle_sets: List of angle sets to compare
        """
        if angle_sets is None:
            # Default: compare full set vs set with one angle slightly off
            full_set = sorted(self.images.keys())
            angle_sets = [
                full_set,
                full_set  # Will modify one angle in the second set
            ]

        fig, axes = plt.subplots(2, len(angle_sets), figsize=(6*len(angle_sets), 10))

        for i, angles in enumerate(angle_sets):
            retardance, orientation = self.compute_birefringence(angles)

            if retardance is None:
                continue

            # Retardance map
            im1 = axes[0, i].imshow(retardance, cmap='viridis')
            axes[0, i].set_title(f'Retardance (Angles: {len(angles)})')
            axes[0, i].axis('off')
            plt.colorbar(im1, ax=axes[0, i], fraction=0.046, label='Retardance')

            # Orientation map (convert to degrees)
            orientation_deg = orientation * 180 / np.pi
            im2 = axes[1, i].imshow(orientation_deg, cmap='hsv', vmin=-90, vmax=90)
            axes[1, i].set_title('Orientation')
            axes[1, i].axis('off')
            plt.colorbar(im2, ax=axes[1, i], fraction=0.046, label='Orientation (deg)')

        plt.suptitle('Birefringence Analysis Comparison', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'birefringence_comparison.png', dpi=150)
        plt.close()  # Don't block - file is already saved

    def generate_sensitivity_plots(self, df_differences: pd.DataFrame,
                                  df_birefringence: pd.DataFrame):
        """
        Generate plots showing sensitivity to angular deviations.

        Args:
            df_differences: DataFrame from compute_image_differences
            df_birefringence: DataFrame from analyze_birefringence_sensitivity
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Plot 1: Image difference metrics vs angular deviation
        if not df_differences.empty:
            ax = axes[0, 0]
            ax.plot(df_differences['angular_difference'],
                   df_differences['normalized_mae'], 'o-', label='Normalized MAE')
            ax.plot(df_differences['angular_difference'],
                   1 - df_differences['ssim'], 's-', label='1 - SSIM')
            ax.set_xlabel('Angular Difference (degrees)')
            ax.set_ylabel('Error Metric')
            ax.set_title('Image Difference vs Angular Deviation')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 2: Correlation vs angular deviation
        if not df_differences.empty:
            ax = axes[0, 1]
            ax.plot(df_differences['angular_difference'],
                   df_differences['correlation'], 'o-', color='blue')
            ax.set_xlabel('Angular Difference (degrees)')
            ax.set_ylabel('Correlation Coefficient')
            ax.set_title('Image Correlation vs Angular Deviation')
            ax.grid(True, alpha=0.3)

        # Plot 3: PSNR vs angular deviation
        if not df_differences.empty:
            ax = axes[0, 2]
            ax.plot(df_differences['angular_difference'],
                   df_differences['psnr'], 'o-', color='green')
            ax.set_xlabel('Angular Difference (degrees)')
            ax.set_ylabel('PSNR (dB)')
            ax.set_title('Peak Signal-to-Noise Ratio')
            ax.grid(True, alpha=0.3)

        # Plot 4: Retardance error vs deviation
        if not df_birefringence.empty:
            ax = axes[1, 0]
            for angle in df_birefringence['deviated_angle'].unique():
                subset = df_birefringence[df_birefringence['deviated_angle'] == angle]
                ax.plot(subset['deviation'], subset['mean_retardance_error'],
                       'o-', label=f'{angle} deg')
            ax.set_xlabel('Angular Deviation (degrees)')
            ax.set_ylabel('Mean Retardance Error')
            ax.set_title('Retardance Sensitivity to Angular Deviation')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 5: Orientation error vs deviation
        if not df_birefringence.empty:
            ax = axes[1, 1]
            for angle in df_birefringence['deviated_angle'].unique():
                subset = df_birefringence[df_birefringence['deviated_angle'] == angle]
                ax.plot(subset['deviation'], subset['mean_orientation_error_deg'],
                       's-', label=f'{angle} deg')
            ax.set_xlabel('Angular Deviation (degrees)')
            ax.set_ylabel('Mean Orientation Error (degrees)')
            ax.set_title('Orientation Sensitivity to Angular Deviation')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 6: Combined RMSE metrics
        if not df_birefringence.empty:
            ax = axes[1, 2]
            mean_by_dev = df_birefringence.groupby('deviation').mean()
            ax.plot(mean_by_dev.index, mean_by_dev['retardance_rmse'],
                   'o-', label='Retardance RMSE')
            ax2 = ax.twinx()
            ax2.plot(mean_by_dev.index, mean_by_dev['orientation_rmse_deg'],
                    's-', color='red', label='Orientation RMSE')
            ax.set_xlabel('Angular Deviation (degrees)')
            ax.set_ylabel('Retardance RMSE', color='blue')
            ax2.set_ylabel('Orientation RMSE (deg)', color='red')
            ax.set_title('Birefringence RMSE vs Angular Deviation')
            ax.grid(True, alpha=0.3)

        plt.suptitle('PPM Rotation Sensitivity Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sensitivity_analysis.png', dpi=150)
        plt.close()  # Don't block - file is already saved

    def compute_birefringence_hue_shift(self, angle1: float, angle2: float) -> Dict:
        """
        Compute birefringence-related hue shift between two angles.

        For PPM, birefringence is computed from paired angles (+/- from 45 deg).
        This measures how much the "hue" (orientation/phase) changes.

        Args:
            angle1: First angle
            angle2: Second angle (typically angle1's pair or small deviation)

        Returns:
            Dict with hue shift metrics
        """
        if angle1 not in self.images or angle2 not in self.images:
            return {'error': f'Missing angle: {angle1} or {angle2}'}

        img1 = self.images[angle1]
        img2 = self.images[angle2]

        # Compute phase difference (simplified birefringence proxy)
        diff = img2.astype(np.float64) - img1.astype(np.float64)

        # Find background level using mode (most common value)
        # Use appropriate number of bins for image bit depth
        img_min, img_max = float(img1.min()), float(img1.max())
        img_range = img_max - img_min

        if img_range == 0:
            return {'error': 'Image has no intensity variation'}

        # Use 256 bins but scaled to actual data range
        n_bins = min(256, int(img_range) + 1)
        hist, bin_edges = np.histogram(img1.flatten(), bins=n_bins)
        mode_idx = np.argmax(hist)
        background_level = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2

        # Threshold: 5% of image range above mode, or at least 1 intensity unit
        threshold = max(img_range * 0.05, 1.0)
        above_bg_mask = img1 > (background_level + threshold)

        n_above = int(np.sum(above_bg_mask))
        if n_above == 0:
            # Fall back to top 25% of pixels
            threshold_75 = np.percentile(img1, 75)
            above_bg_mask = img1 > threshold_75
            n_above = int(np.sum(above_bg_mask))
            background_level = threshold_75  # Update for reporting

        if n_above == 0:
            return {'error': 'No pixels above background (even with fallback)'}

        # Compute statistics for pixels above background
        diff_above_bg = diff[above_bg_mask]

        return {
            'background_mode': float(background_level),
            'threshold_used': float(threshold) if 'threshold' in dir() else float(background_level),
            'n_pixels_above_bg': n_above,
            'pct_pixels_above_bg': float(n_above / above_bg_mask.size * 100),
            'mean_diff_above_bg': float(np.mean(diff_above_bg)),
            'std_diff_above_bg': float(np.std(diff_above_bg)),
            'median_diff_above_bg': float(np.median(diff_above_bg)),
            'mean_intensity_above_bg_1': float(np.mean(img1[above_bg_mask])),
            'mean_intensity_above_bg_2': float(np.mean(img2[above_bg_mask])),
        }

    def generate_report(self, df_differences: pd.DataFrame,
                       df_birefringence: pd.DataFrame,
                       df_adjacent: pd.DataFrame = None,
                       fine_sensitivity: Dict = None):
        """
        Generate a comprehensive report of the analysis.

        Args:
            df_differences: DataFrame from compute_image_differences
            df_birefringence: DataFrame from analyze_birefringence_sensitivity
            df_adjacent: DataFrame from compute_adjacent_differences
            fine_sensitivity: Dict from analyze_fine_sensitivity
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'base_path': str(self.base_path),
            'n_images_loaded': len(self.images),
            'angles_available': sorted(self.images.keys()),
            'image_shape': self.image_shape,
            'summary': {}
        }

        # Save detailed DataFrames
        if df_differences is not None and not df_differences.empty:
            df_differences.to_csv(self.output_dir / 'image_differences.csv', index=False)
        if df_birefringence is not None and not df_birefringence.empty:
            df_birefringence.to_csv(self.output_dir / 'birefringence_sensitivity.csv', index=False)

        # Save JSON report
        with open(self.output_dir / 'analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Generate comprehensive text summary
        with open(self.output_dir / 'summary.txt', 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("PPM ROTATION SENSITIVITY ANALYSIS SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Analysis Date: {report['timestamp']}\n")
            f.write(f"Images Analyzed: {report['n_images_loaded']}\n")
            f.write(f"Image Shape: {self.image_shape}\n\n")

            # ============================================================
            # SECTION 0: Zero-Rotation Baseline (Sanity Check)
            # ============================================================
            f.write("=" * 70 + "\n")
            f.write("0. ZERO-ROTATION BASELINE (Measurement Noise Floor)\n")
            f.write("   (Images acquired at SAME angle - should show ~0% difference)\n")
            f.write("=" * 70 + "\n\n")

            # Analyze baseline pairs
            df_baseline = self.analyze_zero_rotation_baseline()
            if not df_baseline.empty:
                # First, show SANITY CHECK result (self-copy comparison)
                sanity_check = df_baseline[df_baseline['angle'] == -999.0]
                if not sanity_check.empty:
                    f.write("  SANITY CHECK: Image compared to EXACT COPY of itself\n")
                    f.write("  " + "-" * 50 + "\n")
                    sc = sanity_check.iloc[0]
                    f.write(f"    MAE: {sc['mae']:.10f}\n")
                    f.write(f"    Max pixel diff: {sc['max_pixel_diff']:.10f}\n")
                    f.write(f"    Percent change: {sc['pct_change']:.10f}%\n")
                    if sc['mae'] == 0.0 and sc['max_pixel_diff'] == 0.0:
                        f.write(f"    --> PASS: Calculation verified (exactly 0.0)\n\n")
                    else:
                        f.write(f"    --> FAIL: Should be exactly 0.0! Check calculation.\n\n")

                # Filter out sanity check for actual baseline analysis
                df_actual = df_baseline[df_baseline['angle'] != -999.0]

                if not df_actual.empty:
                    f.write(f"  ACTUAL BASELINE PAIRS: {len(df_actual)}\n")
                    f.write("  " + "-" * 50 + "\n\n")

                    # Group by angle
                    for angle in sorted(df_actual['angle'].unique()):
                        angle_data = df_actual[df_actual['angle'] == angle]
                        f.write(f"  Angle {angle:.1f} deg:\n")
                        f.write(f"    Mean intensity change: {angle_data['pct_change'].mean():.4f}%\n")
                        f.write(f"    Max intensity change: {angle_data['pct_change'].max():.4f}%\n")
                        f.write(f"    Mean SSIM: {angle_data['ssim'].mean():.6f}\n")
                        f.write(f"    Pairs: {len(angle_data)}\n\n")

                    # Overall summary (excluding sanity check)
                    f.write(f"  OVERALL BASELINE (actual pairs only):\n")
                    f.write(f"    Mean intensity change: {df_actual['pct_change'].mean():.4f}%\n")
                    f.write(f"    Max intensity change: {df_actual['pct_change'].max():.4f}%\n")
                    f.write(f"    Mean SSIM: {df_actual['ssim'].mean():.6f}\n\n")

                    # Interpretation
                    mean_baseline = df_actual['pct_change'].mean()
                    if mean_baseline < 0.1:
                        f.write(f"  --> GOOD: Baseline noise is low ({mean_baseline:.4f}%)\n")
                        f.write(f"      Measurement methodology is valid.\n\n")
                    elif mean_baseline < 1.0:
                        f.write(f"  --> WARNING: Baseline noise is moderate ({mean_baseline:.4f}%)\n")
                        f.write(f"      Consider this when interpreting small deviations.\n\n")
                    else:
                        f.write(f"  --> PROBLEM: Baseline noise is HIGH ({mean_baseline:.4f}%)\n")
                        f.write(f"      This indicates a measurement problem, NOT rotation error!\n")
                        f.write(f"      Check: camera noise, light fluctuation, or exposure drift.\n\n")
            else:
                f.write("  No baseline image pairs found.\n")
                f.write("  Run zero-rotation baseline test to establish noise floor.\n\n")

            # ============================================================
            # SECTION 1: Fine Angular Sensitivity Around Base Angles
            # ============================================================
            f.write("=" * 70 + "\n")
            f.write("1. FINE ANGULAR SENSITIVITY\n")
            f.write("   (Intensity change for small deviations from base angles)\n")
            f.write("=" * 70 + "\n\n")

            if fine_sensitivity:
                for base_angle, data in fine_sensitivity.items():
                    f.write(f"Base Angle: {data['base_angle']:.2f} deg\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"  Samples: {data['n_samples']}\n")
                    f.write(f"  Sensitivity: {data['sensitivity_pct_per_deg']:.3f}% intensity change per degree\n")

                    # Calculate baseline offset (constant offset present in all comparisons)
                    # This is the minimum pct_change, which represents the baseline noise floor
                    all_pcts = [item['pct_change'] for item in data['details']]
                    baseline_offset = min(all_pcts) if all_pcts else 0

                    f.write(f"  Baseline offset: {baseline_offset:.4f}% (constant offset in all comparisons)\n\n")

                    # Table of deviations - show BOTH raw and corrected values
                    # Show actual filename if available
                    ref_file = "unknown"
                    if hasattr(self, 'image_files') and data['base_angle'] in self.image_files:
                        ref_file = self.image_files[data['base_angle']]
                    f.write(f"  Reference: {data['base_angle']:.2f} deg (file: {ref_file})\n")
                    f.write("  'Raw Change' = total intensity difference from reference.\n")
                    f.write("  'Angular Signal' = Raw - offset = change due to angle alone.\n\n")
                    f.write("  Deviation (deg)  |  Raw Change (%)  |  Angular Signal (%)\n")
                    f.write("  " + "-" * 60 + "\n")

                    # Sort by absolute deviation
                    sorted_details = sorted(data['details'], key=lambda x: abs(x['deviation']))
                    for item in sorted_details:
                        dev = item['deviation']
                        pct = item['pct_change']
                        corrected = pct - baseline_offset
                        f.write(f"  {dev:+7.2f}          |  {pct:8.4f}%      |  {corrected:8.4f}%\n")

                    f.write("\n")
            else:
                f.write("  No fine sensitivity data available.\n\n")

            # ============================================================
            # SECTION 2: Intensity Change by Base Angle and Step Size
            # ============================================================
            f.write("=" * 70 + "\n")
            f.write("2. INTENSITY CHANGE BY BASE ANGLE AND STEP SIZE\n")
            f.write("   (Shows sensitivity at each polarization angle)\n")
            f.write("=" * 70 + "\n\n")

            if df_adjacent is not None and not df_adjacent.empty:
                # Define base angle regions and step size bins
                base_regions = {
                    '0 deg (crossed)': (-1.0, 2.0),
                    '7 deg': (5.0, 10.0),
                    '-7 deg': (-10.0, -5.0),
                    '90 deg (uncrossed)': (88.0, 92.0),
                }

                step_bins = [
                    ('0.05', 0.0, 0.06),
                    ('0.10', 0.06, 0.12),
                    ('0.15', 0.12, 0.18),
                    ('0.20', 0.18, 0.22),
                    ('0.25', 0.22, 0.28),
                    ('0.30', 0.28, 0.35),
                    ('0.50', 0.45, 0.55),
                    ('0.70', 0.65, 0.75),
                    ('1.00', 0.95, 1.05),
                ]

                for base_name, (base_min, base_max) in base_regions.items():
                    # Find adjacent comparisons where angle1 is in this base region
                    base_data = df_adjacent[
                        (df_adjacent['angle1'] >= base_min) &
                        (df_adjacent['angle1'] <= base_max)
                    ]

                    if len(base_data) == 0:
                        continue

                    f.write(f"  {base_name}:\n")
                    f.write("  " + "-" * 50 + "\n")
                    f.write("    Step     |  MAE   |  % Change  |  N\n")
                    f.write("    " + "-" * 40 + "\n")

                    found_any = False
                    for step_name, step_min, step_max in step_bins:
                        step_data = base_data[
                            (base_data['delta_deg'] >= step_min) &
                            (base_data['delta_deg'] <= step_max)
                        ]

                        if len(step_data) > 0:
                            found_any = True
                            mean_mae = step_data['mae'].mean()
                            mean_pct = step_data['pct_change'].mean()
                            n = len(step_data)
                            f.write(f"    {step_name:5s} deg |  {mean_mae:5.2f} |  {mean_pct:8.4f}%  |  {n}\n")

                    if not found_any:
                        f.write("    (No fine step data in this region)\n")

                    f.write("\n")

                # Also show any large jumps (between base angles)
                large_jumps = df_adjacent[df_adjacent['delta_deg'] > 5.0]
                if len(large_jumps) > 0:
                    f.write("  Large jumps (between base angles):\n")
                    f.write("  " + "-" * 50 + "\n")
                    for _, row in large_jumps.iterrows():
                        f.write(f"    {row['angle1']:.1f} -> {row['angle2']:.1f} "
                               f"(delta={row['delta_deg']:.1f}): "
                               f"MAE={row['mae']:.2f}, {row['pct_change']:.3f}%\n")
                    f.write("\n")

            else:
                f.write("  No adjacent difference data available.\n\n")

            # ============================================================
            # SECTION 3: Intensity Comparison - Small vs Large Deviations
            # ============================================================
            f.write("=" * 70 + "\n")
            f.write("3. INTENSITY CHANGE: SMALL DEVIATIONS vs LARGE BASELINE SHIFT\n")
            f.write("   (Comparing fine steps to standard angle steps)\n")
            f.write("=" * 70 + "\n\n")

            # Find reference pairs for comparison
            sorted_angles = sorted(self.images.keys())

            # Look for 7 degree and nearby fine steps
            angles_near_7 = [a for a in sorted_angles if 6.0 <= a <= 8.0]
            angles_near_0 = [a for a in sorted_angles if -0.5 <= a <= 0.5]

            if angles_near_7 and len(angles_near_7) > 1:
                base_7 = min(angles_near_7, key=lambda x: abs(x - 7.0))
                f.write(f"  Reference angle: {base_7:.2f} deg\n\n")

                # Use fixed normalization (255 for 8-bit)
                full_range = 65535.0 if self.images[base_7].max() > 255 else 255.0

                # Small deviations from 7
                f.write("  Small deviations from reference:\n")
                f.write("  " + "-" * 50 + "\n")
                pcts = []
                for angle in sorted(angles_near_7):
                    if angle != base_7:
                        dev = angle - base_7
                        img_ref = self.images[base_7]
                        img_comp = self.images[angle]
                        mae = float(np.mean(np.abs(img_comp - img_ref)))
                        pct = (mae / full_range) * 100
                        pcts.append(pct)
                        f.write(f"    {base_7:.2f} -> {angle:.2f} (delta={dev:+.2f}): {pct:.4f}% change\n")

                # Show corrected values
                if pcts:
                    min_pct = min(pcts)
                    f.write(f"\n  Baseline offset: {min_pct:.4f}%\n")
                    f.write("  After subtracting offset, angular signal ranges from 0% to "
                           f"{max(pcts) - min_pct:.4f}%\n")

                # Large baseline shift (7 vs 0)
                if angles_near_0:
                    base_0 = min(angles_near_0, key=lambda x: abs(x))
                    f.write(f"\n  Large baseline shift ({base_7:.2f} vs {base_0:.2f}):\n")
                    f.write("  " + "-" * 50 + "\n")
                    img_ref = self.images[base_7]
                    img_0 = self.images[base_0]
                    mae = float(np.mean(np.abs(img_0 - img_ref)))
                    pct = (mae / full_range) * 100
                    f.write(f"    {base_7:.2f} -> {base_0:.2f} (delta={base_0 - base_7:.2f}): {pct:.4f}% change\n")

                f.write("\n")

            # ============================================================
            # SECTION 4: Birefringence Hue Analysis (PPM Angles)
            # ============================================================
            f.write("=" * 70 + "\n")
            f.write("4. BIREFRINGENCE HUE ANALYSIS (PPM Angles)\n")
            f.write("   (Intensity difference for pixels above background mode)\n")
            f.write("=" * 70 + "\n\n")

            # Analyze hue shift for PPM-relevant angle groups: 7, -7, 0, 90
            ppm_base_angles = [7, -7, 0, 90]

            for ppm_base in ppm_base_angles:
                # Find angles near this PPM base angle
                angles_near_base = [a for a in sorted_angles if abs(a - ppm_base) <= 2.0]

                if len(angles_near_base) >= 2:
                    base_angle = min(angles_near_base, key=lambda x: abs(x - ppm_base))
                    f.write(f"  PPM Base Angle: {ppm_base} deg (actual: {base_angle:.2f} deg)\n")
                    f.write("  " + "-" * 50 + "\n")

                    f.write("  Angle Pair          |  Background  |  Mean Diff  |  Std Diff  |  Pixels > BG\n")
                    f.write("  " + "-" * 75 + "\n")

                    hue_results_found = False
                    for angle in sorted(angles_near_base):
                        if angle != base_angle:
                            hue_data = self.compute_birefringence_hue_shift(base_angle, angle)
                            if hue_data and 'error' not in hue_data:
                                hue_results_found = True
                                f.write(f"  {base_angle:.2f} -> {angle:.2f}  |  "
                                       f"{hue_data['background_mode']:10.1f}  |  "
                                       f"{hue_data['mean_diff_above_bg']:9.2f}  |  "
                                       f"{hue_data['std_diff_above_bg']:8.2f}  |  "
                                       f"{hue_data['pct_pixels_above_bg']:6.1f}%\n")
                            elif hue_data and 'error' in hue_data:
                                f.write(f"  {base_angle:.2f} -> {angle:.2f}  |  ERROR: {hue_data['error']}\n")

                    if not hue_results_found:
                        f.write("  (No valid hue comparisons found)\n")

                    f.write("\n")

            # Cross-angle comparison: 7 vs -7 (should show birefringence signal)
            if angles_near_7:
                base_pos7 = min(angles_near_7, key=lambda x: abs(x - 7.0))
                angles_near_neg7 = [a for a in sorted_angles if abs(a - (-7.0)) <= 1.0]
                if angles_near_neg7:
                    base_neg7 = min(angles_near_neg7, key=lambda x: abs(x - (-7.0)))
                    f.write(f"  Cross-angle comparison (+7 vs -7 deg):\n")
                    f.write("  " + "-" * 50 + "\n")
                    hue_data = self.compute_birefringence_hue_shift(base_pos7, base_neg7)
                    if hue_data and 'error' not in hue_data:
                        f.write(f"    {base_pos7:.2f} deg vs {base_neg7:.2f} deg\n")
                        f.write(f"    Background mode: {hue_data['background_mode']:.1f}\n")
                        f.write(f"    Mean diff (above BG): {hue_data['mean_diff_above_bg']:.2f}\n")
                        f.write(f"    Std diff (above BG): {hue_data['std_diff_above_bg']:.2f}\n")
                        f.write(f"    Pixels above BG: {hue_data['pct_pixels_above_bg']:.1f}%\n")
                    elif hue_data and 'error' in hue_data:
                        f.write(f"    ERROR: {hue_data['error']}\n")
                    f.write("\n")

            # ============================================================
            # SECTION 5: Key Findings Summary
            # ============================================================
            f.write("=" * 70 + "\n")
            f.write("5. KEY FINDINGS\n")
            f.write("=" * 70 + "\n\n")

            if fine_sensitivity:
                for base_angle, data in fine_sensitivity.items():
                    sens = data['sensitivity_pct_per_deg']
                    f.write(f"  - At {data['base_angle']:.0f} deg: {sens:.3f}% intensity change per degree\n")
                    f.write(f"    -> 0.05 deg error causes ~{sens * 0.05:.4f}% intensity change\n")
                    f.write(f"    -> 0.10 deg error causes ~{sens * 0.10:.4f}% intensity change\n")
                    f.write(f"    -> 1.00 deg error causes ~{sens * 1.00:.3f}% intensity change\n\n")

            f.write("=" * 70 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 70 + "\n")

        print(f"\nAnalysis complete. Results saved to: {self.output_dir}")
        return report


def main():
    parser = argparse.ArgumentParser(description='Analyze PPM rotation sensitivity')
    parser.add_argument('image_dir', help='Directory containing PPM images')
    parser.add_argument('--output', '-o', help='Output directory for results')
    parser.add_argument('--reference-angle', '-r', type=float, default=7.0,
                       help='Reference angle for comparisons (default: 7.0)')
    parser.add_argument('--deviations', '-d', nargs='+', type=float,
                       default=[0.1, 0.2, 0.3, 0.5, 1.0],
                       help='Angular deviations to test (default: 0.1 0.2 0.3 0.5 1.0)')
    parser.add_argument('--skip-visualization', action='store_true',
                       help='Skip visualization plots')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = PPMRotationAnalyzer(args.image_dir, args.output)

    # Load images
    images = analyzer.load_images()
    if len(images) < 3:
        print("Error: Need at least 3 images at different angles for analysis")
        return

    # Compute image differences
    print("\nAnalyzing image differences...")
    df_differences = analyzer.compute_image_differences(args.reference_angle)

    # Analyze birefringence sensitivity
    print("Analyzing birefringence sensitivity...")
    df_birefringence = analyzer.analyze_birefringence_sensitivity(args.deviations)

    # Generate visualizations
    if not args.skip_visualization:
        print("Generating visualizations...")
        analyzer.visualize_difference_maps(args.reference_angle)
        analyzer.visualize_birefringence_comparison()
        analyzer.generate_sensitivity_plots(df_differences, df_birefringence)

    # Generate report
    print("Generating report...")
    report = analyzer.generate_report(df_differences, df_birefringence)

    # Print summary to console
    print("\n" + "=" * 50)
    print("ANALYSIS SUMMARY")
    print("=" * 50)

    if not df_differences.empty:
        print(f"\nImage Quality at Various Angular Differences:")
        print(df_differences[['angular_difference', 'correlation', 'ssim', 'psnr']].to_string(index=False))

    if not df_birefringence.empty:
        print(f"\nBirefringence Sensitivity (Mean Errors):")
        summary = df_birefringence.groupby('deviation')[
            ['mean_retardance_error', 'mean_orientation_error_deg']
        ].mean()
        print(summary.to_string())

    print(f"\nFull results saved to: {analyzer.output_dir}")


if __name__ == "__main__":
    main()