#!/usr/bin/env python3
"""
PPM Rotation Sensitivity Testing Suite

This module integrates with the existing qp_server infrastructure to systematically
test PPM rotation sensitivity by acquiring images at precise angles and analyzing
the impact of angular deviations on image quality and birefringence calculations.

This script leverages existing utilities from:
- qp_utils.py: PolarizerCalibrationUtils, BackgroundCorrectionUtils
- test_client.py: QuPathTestClient for server communication
- qp_server.py: Server command protocol
"""

import sys
import os
import socket
import struct
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
import yaml

# Import from existing project infrastructure
from smart_wsi_scanner.tests.test_client import QuPathTestClient
from smart_wsi_scanner.qp_utils import (
    PolarizerCalibrationUtils,
    BackgroundCorrectionUtils,
)
from smart_wsi_scanner.server.protocol import ExtendedCommand
from smart_wsi_scanner.config import ConfigManager

# Import the analysis module from same package
try:
    from smart_wsi_scanner.ppm.sensitivity_analysis import PPMRotationAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PPM rotation analyzer not found ({e}). Analysis will be skipped.")
    ANALYZER_AVAILABLE = False


class PPMRotationSensitivityTester:
    """
    Comprehensive PPM rotation sensitivity testing using existing infrastructure.

    This class coordinates systematic image acquisition at precise angles and
    analyzes the impact on image quality and birefringence calculations.
    """

    # Fixed exposures for each angle category (from background calibration)
    # These are predetermined values - SNAP uses them directly without adaptive adjustment
    # Can be overridden via CLI --angle-exposures or future QuPath command
    ANGLE_EXPOSURES_MS = {
        -7.0: 21.1,
        0.0: 96.81,
        7.0: 22.63,
        90.0: 0.57,
    }

    def __init__(self,
                 config_yaml: str,
                 output_dir: str = None,
                 host: str = "127.0.0.1",
                 port: int = 5000,
                 angle_exposures: Dict[float, float] = None,
                 keep_images: bool = True):
        """
        Initialize the tester with configuration and connection parameters.

        Args:
            config_yaml: Path to microscope configuration YAML file
            output_dir: Output directory for test results (default: configurations/ppm_sensitivity_tests/)
            host: qp_server host address
            port: qp_server port
            angle_exposures: Optional dict of {angle: exposure_ms} to override defaults.
                            If provided, updates ANGLE_EXPOSURES_MS.
            keep_images: If True, keep acquired .tif images after analysis.
                        If False, delete them to conserve disk space (default: True).
        """
        self.keep_images = keep_images
        self.config_yaml = Path(config_yaml)

        # Default output to configurations/ppm_sensitivity_tests/
        if output_dir is None:
            config_dir = Path(__file__).parent.parent / "smart_wsi_scanner" / "configurations"
            self.output_dir = config_dir / "ppm_sensitivity_tests" / f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logging
        self.setup_logging()

        # Load configuration
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config_file(str(self.config_yaml))

        # Initialize client connection
        self.client = QuPathTestClient(host=host, port=port)
        self.connected = False

        # Initialize utilities
        self.pol_utils = PolarizerCalibrationUtils()
        self.bg_utils = BackgroundCorrectionUtils()

        # Store test results
        self.test_results = {}
        self.acquired_images = {}

        # Standard PPM angles from config or defaults
        self.standard_angles = self.config.get('ppm', {}).get('angles',
            [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91])

        # Override angle exposures if provided
        if angle_exposures:
            self.ANGLE_EXPOSURES_MS.update(angle_exposures)
            self.logger.info(f"Updated angle exposures with overrides: {angle_exposures}")

        # Load imaging profile exposures if available (for reference only)
        self.imaging_profile_exposures = self._load_imaging_profile_exposures()

        self.logger.info(f"PPM Rotation Sensitivity Tester initialized")
        self.logger.info(f"Config: {self.config_yaml}")
        self.logger.info(f"Output: {self.output_dir}")
        self.logger.info(f"Angle exposures (fixed, no adaptive): {self.ANGLE_EXPOSURES_MS}")

    def _load_imaging_profile_exposures(self) -> Dict[str, float]:
        """
        Load exposure times from imaging profile config.

        Returns:
            Dictionary mapping angle categories to exposure times (ms):
            {'negative': 800, 'crossed': 1200, 'positive': 800, 'uncrossed': 15}
        """
        exposures = {}
        try:
            # Try to load imageprocessing config
            config_dir = Path(__file__).parent.parent / "smart_wsi_scanner" / "configurations"
            imgproc_file = config_dir / "imageprocessing_PPM.yml"

            if imgproc_file.exists():
                imgproc = self.config_manager.load_config_file(str(imgproc_file))

                # Get current objective/detector from main config
                objective = self.config.get('objective', 'LOCI_OBJECTIVE_OLYMPUS_20X_POL_001')
                detector = self.config.get('detector', 'LOCI_DETECTOR_TELEDYNE_001')

                # Navigate to exposures
                profiles = imgproc.get('imaging_profiles', {}).get('ppm', {})
                obj_profile = profiles.get(objective, {})
                det_profile = obj_profile.get(detector, {})
                exp_config = det_profile.get('exposures_ms', {})

                # Extract exposures (handle both simple and per-channel formats)
                for category in ['negative', 'crossed', 'positive', 'uncrossed']:
                    val = exp_config.get(category)
                    if isinstance(val, dict):
                        exposures[category] = val.get('all', list(val.values())[0])
                    elif val is not None:
                        exposures[category] = float(val)

            else:
                self.logger.warning(f"Imaging profile not found at {imgproc_file}")

        except Exception as e:
            self.logger.warning(f"Could not load imaging profile exposures: {e}")

        return exposures

    def get_exposure_for_angle(self, angle: float) -> float:
        """
        Get the fixed exposure time for an angle.

        Uses predetermined exposures from ANGLE_EXPOSURES_MS and interpolates
        logarithmically for angles between calibration points.

        Calibration points (from background acquisition):
        - -7.0 deg: 21.1 ms (negative polarization)
        - 0.0 deg: 96.81 ms (crossed polars - darkest)
        - 7.0 deg: 22.63 ms (positive polarization)
        - 90.0 deg: 0.57 ms (uncrossed - brightest)

        Interpolation: For angles like 45 deg, logarithmic interpolation
        between 7 deg and 90 deg exposures is used.

        Args:
            angle: Rotation angle in degrees

        Returns:
            Exposure time in ms (fixed, not adaptive)
        """
        exp = self.ANGLE_EXPOSURES_MS

        # Check for exact match first
        if angle in exp:
            return exp[angle]

        # Map angle to nearest calibration category
        abs_angle = abs(angle)

        # Near 0 (crossed polars): use 0 deg exposure
        if abs_angle <= 3:
            return exp[0.0]

        # Near +/-7 (polarization angles): use 7 or -7 deg exposure
        elif 3 < abs_angle <= 10:
            if angle >= 0:
                return exp[7.0]
            else:
                return exp[-7.0]

        # Near 90 (uncrossed): use 90 deg exposure
        elif 85 <= abs_angle <= 95:
            return exp[90.0]

        # For angles between 10-85, interpolate logarithmically
        # (exposures span ~2 orders of magnitude: 22ms to 0.57ms)
        elif 10 < abs_angle < 85:
            import math
            exp_7 = exp[7.0]
            exp_90 = exp[90.0]
            # Normalize angle to 0-1 range between 7 and 90
            t = (abs_angle - 7) / (90 - 7)
            # Log interpolation for large exposure ranges
            log_exp = math.log(exp_7) * (1 - t) + math.log(exp_90) * t
            return math.exp(log_exp)

        # Fallback for any edge cases
        else:
            return exp[7.0]  # Default to 7 deg exposure

    def setup_logging(self):
        """Setup comprehensive logging for the test session."""
        log_file = self.output_dir / "ppm_sensitivity_test.log"

        # Configure logger
        self.logger = logging.getLogger("PPMSensitivityTest")
        self.logger.setLevel(logging.DEBUG)

        # Prevent propagation to root logger (avoids duplicate messages)
        self.logger.propagate = False

        # Clear existing handlers (in case of re-initialization)
        self.logger.handlers = []

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def connect(self) -> bool:
        """Connect to the qp_server."""
        try:
            self.client.connect()
            self.connected = True
            self.logger.info(f"Connected to server at {self.client.host}:{self.client.port}")

            # Test connection with status check
            status = self.client.test_status()
            self.logger.info(f"Server status: {status}")

            return True
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from the server."""
        if self.connected:
            try:
                self.client.disconnect()
                self.connected = False
                self.logger.info("Disconnected from server")
            except Exception as e:
                self.logger.error(f"Error during disconnect: {e}")

    def _drain_socket(self):
        """Drain any leftover data from the socket buffer."""
        if not self.client.socket:
            return
        try:
            self.client.socket.setblocking(False)
            while True:
                try:
                    data = self.client.socket.recv(4096)
                    if not data:
                        break
                    self.logger.debug(f"Drained {len(data)} bytes from socket")
                except BlockingIOError:
                    break  # No more data
                except Exception:
                    break
        finally:
            self.client.socket.setblocking(True)
            self.client.socket.settimeout(10.0)

    def _reconnect(self):
        """Reconnect to the server to reset socket state."""
        self.logger.info("Reconnecting to reset socket state...")
        try:
            if self.client.socket:
                try:
                    self.client.socket.close()
                except:
                    pass
                self.client.socket = None
            self.connected = False
            time.sleep(0.5)
            return self.connect()
        except Exception as e:
            self.logger.error(f"Reconnection failed: {e}")
            return False

    def acquire_at_angle(self, angle: float, save_name: str,
                        exposure_ms: float = None) -> Optional[Path]:
        """
        Acquire a single image at specified rotation angle using SNAP command.

        Uses fixed exposure (no adaptive adjustment). If exposure_ms is not
        provided, looks up the appropriate exposure from ANGLE_EXPOSURES_MS.

        Args:
            angle: Rotation angle in degrees
            save_name: Name for the saved image
            exposure_ms: Exposure time in milliseconds. If None, uses
                        get_exposure_for_angle() to determine appropriate exposure.

        Returns:
            Path to acquired image or None if failed
        """
        if not self.connected:
            self.logger.error("Not connected to server")
            return None

        # Look up exposure if not provided
        if exposure_ms is None:
            exposure_ms = self.get_exposure_for_angle(angle)
            self.logger.debug(f"Using looked-up exposure for {angle} deg: {exposure_ms:.2f} ms")

        try:
            # Step 1: Move to angle and wait for completion
            self.logger.debug(f"Moving to angle {angle} degrees")
            self.client.test_move_rotation(angle)
            time.sleep(0.5)  # Allow settling

            # Step 2: Verify position (this is a complete request/response cycle)
            actual_angle = self.client.test_get_rotation()
            angle_error = abs(actual_angle - angle)
            self.logger.info(f"Set: {angle:.2f} deg, Read: {actual_angle:.2f} deg, Error: {angle_error:.3f} deg")

            if angle_error > 0.5:  # Warning threshold
                self.logger.warning(f"Large angle error: {angle_error:.3f} degrees")

            # Step 3: Use test_snap method for proper socket handling
            output_path = self.output_dir / save_name

            self.logger.debug(f"Sending SNAP: angle={angle:.2f}, exposure={exposure_ms:.2f}ms")
            result = self.client.test_snap(
                angle=angle,
                exposure_ms=exposure_ms,
                output_path=str(output_path)
            )

            if result:
                self.logger.info(f"SNAP complete: {save_name}")
                return output_path
            else:
                self.logger.error(f"SNAP failed for {save_name}")
                return None

        except Exception as e:
            self.logger.error(f"Error acquiring at angle {angle}: {e}")
            return None

    def run_standard_angles_test(self) -> Dict[float, Path]:
        """
        Acquire images at all standard PPM angles with per-angle fixed exposures.

        Each angle uses its predetermined exposure from ANGLE_EXPOSURES_MS
        (or interpolated for angles not in the dictionary). These are FIXED
        exposures - no adaptive adjustment occurs.

        Returns:
            Dictionary mapping angles to image paths
        """
        self.logger.info("=" * 60)
        self.logger.info("STANDARD ANGLES TEST (PER-ANGLE FIXED EXPOSURES)")
        self.logger.info("=" * 60)
        self.logger.info(f"Exposure lookup: {self.ANGLE_EXPOSURES_MS}")
        self.logger.info("Each angle uses its own fixed exposure (interpolated if not in table)")

        acquired = {}

        for i, angle in enumerate(self.standard_angles):
            exposure = self.get_exposure_for_angle(angle)
            self.logger.info(f"[{i+1}/{len(self.standard_angles)}] Acquiring at {angle} deg (exposure: {exposure:.2f} ms)")

            save_name = f"standard_{angle:05.1f}deg.tif"
            image_path = self.acquire_at_angle(angle, save_name, exposure_ms=exposure)

            if image_path:
                acquired[angle] = image_path
            else:
                self.logger.warning(f"Failed to acquire at {angle} degrees")

        self.logger.info(f"Acquired {len(acquired)}/{len(self.standard_angles)} standard angles")
        self.test_results['standard_angles'] = acquired
        return acquired

    def run_fine_deviation_test(self, base_angle: float = 7.0,
                               deviations: List[float] = None,
                               fixed_exposure_ms: float = None,
                               acquire_zero_reference: bool = True) -> Dict[float, Path]:
        """
        Test fine angular deviations around a base angle.

        All images in a deviation cluster use the SAME exposure (from background
        calibration) to ensure valid sensitivity comparisons.

        Exposure selection:
        1. If fixed_exposure_ms is provided, use that
        2. Otherwise, use background calibration exposure for base angle

        Args:
            base_angle: Center angle for testing
            deviations: List of deviations to test (positive and negative)
            fixed_exposure_ms: Override exposure time for ALL images in cluster.
                              If None, uses background calibration exposure.
            acquire_zero_reference: If True, also acquire 0 deg image with
                                   the cluster's exposure for comparison.

        Returns:
            Dictionary mapping angles to image paths
        """
        if deviations is None:
            # Note: 0 (base angle) is acquired separately first, so start with 0.05
            deviations = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 1.0]

        self.logger.info("=" * 60)
        self.logger.info(f"FINE DEVIATION TEST: {base_angle} degrees")
        self.logger.info("=" * 60)
        self.logger.info(f"Testing deviations: {deviations}")

        # Determine exposure for this cluster from background calibration
        cluster_exposure = fixed_exposure_ms
        if cluster_exposure is None:
            cluster_exposure = self.get_exposure_for_angle(base_angle)
            self.logger.info(f"Using BACKGROUND CALIBRATION exposure for {base_angle} deg: "
                           f"{cluster_exposure:.2f} ms")

        self.logger.info(f"FIXED exposure for entire cluster: {cluster_exposure:.2f} ms")
        self.logger.info("All images in this cluster will have identical exposure")

        acquired = {}

        # First, acquire the BASE ANGLE reference image
        # This is the most important reference - all deviations are compared to this
        self.logger.info("-" * 40)
        self.logger.info(f"Acquiring BASE ANGLE reference: {base_angle:.2f} deg with cluster exposure ({cluster_exposure:.2f} ms)")
        base_save_name = f"{base_angle:.2f}.tif"
        base_image_path = self.acquire_at_angle(base_angle, base_save_name, exposure_ms=cluster_exposure)
        if base_image_path:
            acquired[float(base_angle)] = base_image_path
            self.logger.info(f"Acquired base angle reference: {base_save_name}")
        self.logger.info("-" * 40)

        # Optionally acquire 0 degree reference with THIS cluster's exposure
        # This allows comparison to crossed polars baseline
        if acquire_zero_reference and base_angle != 0:
            self.logger.info("-" * 40)
            self.logger.info(f"Acquiring 0 deg reference with cluster exposure ({cluster_exposure:.2f} ms)")
            # Save with cluster-specific name so multiple runs don't overwrite
            save_name = f"0.00_at_{base_angle:.0f}deg_exposure.tif"
            image_path = self.acquire_at_angle(0.0, save_name, exposure_ms=cluster_exposure)
            if image_path:
                # Store with special key to indicate it's a reference
                acquired[f"0.0_ref_{base_angle}"] = image_path
                self.logger.info(f"Acquired 0 deg reference for {base_angle} deg cluster comparison")
            self.logger.info("-" * 40)

        # Acquire deviation images
        for i, dev in enumerate(deviations):
            # Test positive deviation
            angle_pos = base_angle + dev
            if 0 <= angle_pos <= 180:  # Check hardware limits
                self.logger.info(f"[{i*2+1}/{len(deviations)*2}] Testing {angle_pos:.2f} deg (+{dev})")
                save_name = f"{angle_pos:.2f}.tif"
                image_path = self.acquire_at_angle(angle_pos, save_name,
                                                   exposure_ms=cluster_exposure)
                if image_path:
                    acquired[angle_pos] = image_path

            # Test negative deviation (skip 0)
            if dev > 0:
                angle_neg = base_angle - dev
                if 0 <= angle_neg <= 180:
                    self.logger.info(f"[{i*2+2}/{len(deviations)*2}] Testing {angle_neg:.2f} deg (-{dev})")
                    save_name = f"{angle_neg:.2f}.tif"
                    image_path = self.acquire_at_angle(angle_neg, save_name,
                                                       exposure_ms=cluster_exposure)
                    if image_path:
                        acquired[angle_neg] = image_path

        self.logger.info(f"Acquired {len(acquired)} images (including reference)")
        self.logger.info(f"Cluster exposure used: {cluster_exposure:.2f} ms")
        self.test_results[f'deviation_{base_angle}deg'] = acquired
        return acquired

    def run_zero_rotation_baseline_test(self, test_angles: List[float] = None,
                                        n_repeats: int = 3) -> Dict:
        """
        Baseline test: acquire multiple images at the SAME angle WITHOUT rotation.

        This test validates the measurement methodology by comparing images taken
        at the same angle. The intensity difference should be essentially 0%.
        If we see significant differences here, there's a problem with the
        measurement approach, not the rotation precision.

        Args:
            test_angles: Angles to test (default: [7, 0, 90])
            n_repeats: Number of image pairs to acquire at each angle

        Returns:
            Dictionary with baseline measurements
        """
        if test_angles is None:
            test_angles = [7.0, 0.0, 90.0]

        self.logger.info("=" * 60)
        self.logger.info("ZERO-ROTATION BASELINE TEST")
        self.logger.info("(Comparing images at SAME angle - should show ~0% difference)")
        self.logger.info("=" * 60)

        baseline_results = {}

        for angle in test_angles:
            self.logger.info("-" * 40)
            self.logger.info(f"Testing baseline at {angle} degrees")
            self.logger.info("-" * 40)

            # Get exposure for this angle
            exposure = self.get_exposure_for_angle(angle)
            self.logger.info(f"Using exposure: {exposure:.2f} ms")

            # Move to angle once
            self.client.test_move_rotation(angle)
            time.sleep(0.5)

            # Verify position
            actual = self.client.test_get_rotation()
            self.logger.info(f"Position: {actual:.4f} deg (target: {angle})")

            # Acquire multiple image pairs WITHOUT moving
            # NOTE: BGACQUIRE saves files as {angle}.tif, so we must rename after each
            # acquisition to prevent overwriting when acquiring multiple images at same angle
            pairs = []
            for i in range(n_repeats):
                self.logger.info(f"  Pair {i+1}/{n_repeats}: Acquiring image A...")

                # Image A - BGACQUIRE will save as {angle}.tif
                temp_name = f"{angle}.tif"
                save_name_a = f"baseline_{angle:.1f}deg_pair{i+1}_A.tif"
                self.acquire_at_angle(angle, temp_name, exposure_ms=exposure)

                # Rename immediately to prevent overwrite
                src_path = self.output_dir / temp_name
                path_a = self.output_dir / save_name_a
                if src_path.exists():
                    src_path.rename(path_a)
                    self.logger.info(f"    Renamed {temp_name} -> {save_name_a}")
                else:
                    self.logger.warning(f"    Source file not found: {src_path}")
                    path_a = None

                # DO NOT MOVE - acquire image B at same position
                self.logger.info(f"  Pair {i+1}/{n_repeats}: Acquiring image B (NO rotation)...")
                save_name_b = f"baseline_{angle:.1f}deg_pair{i+1}_B.tif"
                self.acquire_at_angle(angle, temp_name, exposure_ms=exposure)

                # Rename immediately
                src_path = self.output_dir / temp_name
                path_b = self.output_dir / save_name_b
                if src_path.exists():
                    src_path.rename(path_b)
                    self.logger.info(f"    Renamed {temp_name} -> {save_name_b}")
                else:
                    self.logger.warning(f"    Source file not found: {src_path}")
                    path_b = None

                if path_a and path_b:
                    pairs.append({
                        'pair': i + 1,
                        'angle': angle,
                        'image_a': str(path_a),
                        'image_b': str(path_b)
                    })

            baseline_results[angle] = {
                'exposure_ms': exposure,
                'pairs': pairs
            }

            self.logger.info(f"Acquired {len(pairs)} image pairs at {angle} deg")

        self.test_results['zero_rotation_baseline'] = baseline_results
        self.logger.info("=" * 60)
        self.logger.info("Zero-rotation baseline test complete")
        self.logger.info("Analyze these pairs to verify ~0% intensity difference")
        self.logger.info("=" * 60)

        return baseline_results

    def run_repeatability_test(self, test_angle: float = 7.0,
                              n_repeats: int = 10,
                              include_large_movements: bool = True) -> List[Dict]:
        """
        Test mechanical repeatability with both small and large movements.

        Tests both small angular movements (e.g., 0 -> 7 deg) and large movements
        (e.g., 0 -> 90 deg, 90 -> 7 deg) to characterize positioning accuracy
        across the full range of operation.

        Args:
            test_angle: Primary angle to test repeatedly
            n_repeats: Number of repetitions for each movement type
            include_large_movements: If True, also test 90+ degree movements

        Returns:
            List of measurement dictionaries
        """
        self.logger.info("=" * 60)
        self.logger.info(f"REPEATABILITY TEST: {test_angle} degrees x {n_repeats}")
        if include_large_movements:
            self.logger.info("Including large movement tests (90+ degrees)")
        self.logger.info("=" * 60)

        all_measurements = []

        # ========== SMALL MOVEMENT TEST (0 -> test_angle) ==========
        self.logger.info("-" * 40)
        self.logger.info(f"SMALL MOVEMENTS: 0 -> {test_angle} deg ({test_angle} deg travel)")
        self.logger.info("-" * 40)

        for i in range(n_repeats):
            self.logger.info(f"Small movement {i+1}/{n_repeats}")

            # Move away first to test return accuracy
            if i > 0:
                self.client.test_move_rotation(0)
                time.sleep(1)

            # Move to test angle
            self.client.test_move_rotation(test_angle)
            time.sleep(0.5)

            # Read actual position
            actual = self.client.test_get_rotation()
            error = actual - test_angle

            measurement = {
                'repetition': i + 1,
                'test_type': 'small_movement',
                'from_angle': 0,
                'set_angle': test_angle,
                'travel_deg': test_angle,
                'read_angle': actual,
                'error': error,
                'timestamp': datetime.now().isoformat()
            }
            all_measurements.append(measurement)

            self.logger.info(f"  Set: {test_angle:.3f}, Read: {actual:.3f}, Error: {error:.4f}")

        # ========== LARGE MOVEMENT TESTS ==========
        if include_large_movements:
            # Test 1: 0 -> 90 degrees (large movement to uncrossed)
            self.logger.info("-" * 40)
            self.logger.info("LARGE MOVEMENTS: 0 -> 90 deg (90 deg travel)")
            self.logger.info("-" * 40)

            for i in range(n_repeats):
                self.logger.info(f"Large movement (0->90) {i+1}/{n_repeats}")

                # Return to 0
                self.client.test_move_rotation(0)
                time.sleep(1)

                # Move to 90 degrees
                self.client.test_move_rotation(90)
                time.sleep(0.5)

                actual = self.client.test_get_rotation()
                error = actual - 90

                measurement = {
                    'repetition': i + 1,
                    'test_type': 'large_movement_0_to_90',
                    'from_angle': 0,
                    'set_angle': 90,
                    'travel_deg': 90,
                    'read_angle': actual,
                    'error': error,
                    'timestamp': datetime.now().isoformat()
                }
                all_measurements.append(measurement)

                self.logger.info(f"  Set: 90.000, Read: {actual:.3f}, Error: {error:.4f}")

            # Test 2: 90 -> test_angle (large movement back)
            self.logger.info("-" * 40)
            self.logger.info(f"LARGE MOVEMENTS: 90 -> {test_angle} deg ({90 - test_angle} deg travel)")
            self.logger.info("-" * 40)

            for i in range(n_repeats):
                self.logger.info(f"Large movement (90->{test_angle}) {i+1}/{n_repeats}")

                # Move to 90 first
                self.client.test_move_rotation(90)
                time.sleep(1)

                # Move to test angle
                self.client.test_move_rotation(test_angle)
                time.sleep(0.5)

                actual = self.client.test_get_rotation()
                error = actual - test_angle

                measurement = {
                    'repetition': i + 1,
                    'test_type': 'large_movement_90_to_target',
                    'from_angle': 90,
                    'set_angle': test_angle,
                    'travel_deg': 90 - test_angle,
                    'read_angle': actual,
                    'error': error,
                    'timestamp': datetime.now().isoformat()
                }
                all_measurements.append(measurement)

                self.logger.info(f"  Set: {test_angle:.3f}, Read: {actual:.3f}, Error: {error:.4f}")

            # Test 3: Full range 0 -> 90 -> 0 (bidirectional)
            self.logger.info("-" * 40)
            self.logger.info("BIDIRECTIONAL: 0 -> 90 -> 0 (hysteresis test)")
            self.logger.info("-" * 40)

            for i in range(n_repeats):
                self.logger.info(f"Bidirectional {i+1}/{n_repeats}")

                # Start at 0
                self.client.test_move_rotation(0)
                time.sleep(0.5)
                start_pos = self.client.test_get_rotation()

                # Move to 90
                self.client.test_move_rotation(90)
                time.sleep(0.5)
                mid_pos = self.client.test_get_rotation()

                # Return to 0
                self.client.test_move_rotation(0)
                time.sleep(0.5)
                end_pos = self.client.test_get_rotation()

                hysteresis = end_pos - start_pos

                measurement = {
                    'repetition': i + 1,
                    'test_type': 'bidirectional_hysteresis',
                    'start_angle': start_pos,
                    'mid_angle': mid_pos,
                    'end_angle': end_pos,
                    'travel_deg': 180,  # Total travel: 0->90->0
                    'hysteresis': hysteresis,
                    'timestamp': datetime.now().isoformat()
                }
                all_measurements.append(measurement)

                self.logger.info(f"  Start: {start_pos:.4f}, Mid: {mid_pos:.3f}, End: {end_pos:.4f}")
                self.logger.info(f"  Hysteresis: {hysteresis:.4f} deg")

        # ========== CALCULATE STATISTICS BY TEST TYPE ==========
        self.logger.info("=" * 60)
        self.logger.info("REPEATABILITY STATISTICS BY TEST TYPE")
        self.logger.info("=" * 60)

        stats_by_type = {}

        # Group measurements by test type
        test_types = set(m.get('test_type', 'unknown') for m in all_measurements)

        for test_type in test_types:
            type_measurements = [m for m in all_measurements if m.get('test_type') == test_type]

            if test_type == 'bidirectional_hysteresis':
                # Special handling for hysteresis test
                hysteresis_values = [m['hysteresis'] for m in type_measurements]
                stats_by_type[test_type] = {
                    'n_measurements': len(type_measurements),
                    'mean_hysteresis': float(np.mean(hysteresis_values)),
                    'max_hysteresis': float(np.max(np.abs(hysteresis_values))),
                    'std_hysteresis': float(np.std(hysteresis_values))
                }
                self.logger.info(f"\n{test_type}:")
                self.logger.info(f"  N measurements: {stats_by_type[test_type]['n_measurements']}")
                self.logger.info(f"  Mean hysteresis: {stats_by_type[test_type]['mean_hysteresis']:.4f} deg")
                self.logger.info(f"  Max hysteresis: {stats_by_type[test_type]['max_hysteresis']:.4f} deg")
            else:
                # Standard error statistics
                errors = [m['error'] for m in type_measurements]
                travel = type_measurements[0].get('travel_deg', 0) if type_measurements else 0

                stats_by_type[test_type] = {
                    'n_measurements': len(type_measurements),
                    'travel_deg': travel,
                    'mean_error': float(np.mean(errors)),
                    'std_error': float(np.std(errors)),
                    'max_error': float(np.max(np.abs(errors))),
                    'repeatability_2sigma': float(2 * np.std(errors))
                }
                self.logger.info(f"\n{test_type} ({travel} deg travel):")
                self.logger.info(f"  N measurements: {stats_by_type[test_type]['n_measurements']}")
                self.logger.info(f"  Mean error: {stats_by_type[test_type]['mean_error']:.4f} deg")
                self.logger.info(f"  Std deviation: {stats_by_type[test_type]['std_error']:.4f} deg")
                self.logger.info(f"  Max error: {stats_by_type[test_type]['max_error']:.4f} deg")
                self.logger.info(f"  2-sigma repeatability: {stats_by_type[test_type]['repeatability_2sigma']:.4f} deg")

        # Overall statistics (all non-hysteresis measurements)
        all_errors = [m['error'] for m in all_measurements if 'error' in m]
        overall_stats = {
            'total_measurements': len(all_measurements),
            'mean_error': float(np.mean(all_errors)) if all_errors else 0,
            'std_error': float(np.std(all_errors)) if all_errors else 0,
            'max_error': float(np.max(np.abs(all_errors))) if all_errors else 0,
            'repeatability_2sigma': float(2 * np.std(all_errors)) if all_errors else 0
        }

        self.logger.info("\n" + "=" * 40)
        self.logger.info("OVERALL REPEATABILITY (all movement types):")
        self.logger.info(f"  Total measurements: {overall_stats['total_measurements']}")
        self.logger.info(f"  Mean error: {overall_stats['mean_error']:.4f} deg")
        self.logger.info(f"  Max error: {overall_stats['max_error']:.4f} deg")
        self.logger.info(f"  2-sigma repeatability: {overall_stats['repeatability_2sigma']:.4f} deg")

        self.test_results['repeatability'] = {
            'measurements': all_measurements,
            'statistics_by_type': stats_by_type,
            'overall_statistics': overall_stats
        }

        return all_measurements

    def run_polarizer_calibration_comparison(self) -> Dict:
        """
        Run polarizer calibration to find crossed positions and test sensitivity.

        Returns:
            Calibration results dictionary
        """
        self.logger.info("=" * 60)
        self.logger.info("POLARIZER CALIBRATION COMPARISON")
        self.logger.info("=" * 60)

        # Use existing polarizer calibration utility
        output_folder = self.output_dir / "polarizer_calibration"
        output_folder.mkdir(exist_ok=True)

        # NOTE: calibrate_hardware_offset_two_stage requires direct hardware access
        # The actual method signature is:
        #   calibrate_hardware_offset_two_stage(hardware, coarse_range_deg, coarse_step_deg,
        #                                       fine_range_deg, fine_step_deg, exposure_ms,
        #                                       channel, logger_instance)
        # This test currently connects via socket to qp_server and doesn't have direct
        # hardware access. To use this calibration, either:
        #   1. Run calibration through qp_server command protocol
        #   2. Or refactor to get hardware object from the server connection

        self.logger.warning(
            "Polarizer calibration skipped - requires direct hardware access. "
            "Use qp_server commands for calibration instead."
        )

        # Placeholder results
        results = {
            'status': 'skipped',
            'reason': 'Direct hardware access required - not available via socket connection',
            'suggestion': 'Run calibration through qp_server CALIBRATE command'
        }

        self.test_results['polarizer_calibration'] = results
        return results

    def analyze_results(self) -> Dict:
        """
        Analyze all acquired images using the PPMRotationAnalyzer.

        Returns:
            Analysis results dictionary
        """
        if not ANALYZER_AVAILABLE:
            self.logger.warning("Analyzer not available, skipping analysis")
            return {}

        self.logger.info("=" * 60)
        self.logger.info("ANALYZING RESULTS")
        self.logger.info("=" * 60)

        try:
            # Initialize analyzer
            analyzer = PPMRotationAnalyzer(
                base_path=self.output_dir,
                output_dir=self.output_dir / "analysis"
            )

            # Try loading standard images first, then deviation images
            images = analyzer.load_images()

            if len(images) < 3:
                self.logger.info("No standard angle images found, trying deviation images...")
                images = analyzer.load_deviation_images()

            if len(images) < 3:
                self.logger.warning("Insufficient images for analysis (need at least 3)")
                return {}

            # Determine reference angle from loaded images
            # Use the center angle (e.g., 45.0 for deviation test, 7.0 for standard test)
            available_angles = sorted(images.keys())
            center_idx = len(available_angles) // 2
            reference_angle = available_angles[center_idx]
            self.logger.info(f"Using reference angle: {reference_angle:.2f} deg")

            # Run fine sensitivity analysis - this is the key analysis!
            self.logger.info("=" * 60)
            self.logger.info("FINE ANGULAR SENSITIVITY ANALYSIS")
            self.logger.info("=" * 60)

            # Analyze sensitivity around PPM-relevant angles (7, 0, -7, 90 degrees)
            fine_sensitivity = analyzer.analyze_fine_sensitivity(base_angles=[7, 0, -7, 90])

            # Analyze zero-rotation baseline (should show ~0% change)
            self.logger.info("\n" + "=" * 60)
            self.logger.info("ZERO-ROTATION BASELINE ANALYSIS")
            self.logger.info("(Images acquired WITHOUT rotation - should be ~0% difference)")
            self.logger.info("=" * 60)
            df_baseline = analyzer.analyze_zero_rotation_baseline()
            if not df_baseline.empty:
                baseline_csv = self.output_dir / "zero_rotation_baseline.csv"
                df_baseline.to_csv(baseline_csv, index=False)
                self.logger.info(f"Saved baseline analysis to: {baseline_csv}")

            # Also compute adjacent differences for the full picture
            self.logger.info("\n" + "=" * 60)
            self.logger.info("ADJACENT ANGLE COMPARISONS")
            self.logger.info("=" * 60)
            df_adjacent = analyzer.compute_adjacent_differences()

            # Show fine-grained adjacent comparisons (delta < 1 degree)
            if not df_adjacent.empty:
                fine_steps = df_adjacent[df_adjacent['delta_deg'] < 1.0]
                if not fine_steps.empty:
                    self.logger.info("\nFine steps (< 1 degree apart):")
                    for _, row in fine_steps.iterrows():
                        self.logger.info(
                            f"  {row['angle1']:.2f} -> {row['angle2']:.2f} "
                            f"(delta={row['delta_deg']:.2f}): "
                            f"MAE={row['mae']:.2f}, {row['pct_change']:.3f}% change, "
                            f"intensity {row['median_intensity_1']:.0f} -> {row['median_intensity_2']:.0f}"
                        )

            # Compute standard differences for comparison (optional)
            self.logger.info("\nComputing standard reference differences...")
            df_differences = analyzer.compute_image_differences(reference_angle=reference_angle)

            # Skip birefringence analysis for deviation test (needs specific angle pairs)
            import pandas as pd
            df_birefringence = pd.DataFrame()

            # Save detailed results to CSV
            if not df_adjacent.empty:
                csv_path = self.output_dir / "adjacent_differences.csv"
                df_adjacent.to_csv(csv_path, index=False)
                self.logger.info(f"\nSaved adjacent differences to: {csv_path}")

            # Generate visualizations
            self.logger.info("Generating visualizations...")
            analyzer.visualize_difference_maps(reference_angle=reference_angle)

            # Generate report with all collected data
            self.logger.info("Generating analysis report...")
            report = analyzer.generate_report(
                df_differences=df_differences,
                df_birefringence=df_birefringence,
                df_adjacent=df_adjacent,
                fine_sensitivity=fine_sensitivity
            )

            self.test_results['analysis'] = report
            return report

        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return {}

    def save_test_summary(self):
        """Save comprehensive test summary."""
        summary_file = self.output_dir / "test_summary.json"

        summary = {
            'test_date': datetime.now().isoformat(),
            'config_file': str(self.config_yaml),
            'output_directory': str(self.output_dir),
            'tests_performed': list(self.test_results.keys()),
            'results': {}
        }

        # Summarize each test
        for test_name, results in self.test_results.items():
            if test_name == 'standard_angles':
                summary['results'][test_name] = {
                    'angles_acquired': len(results),
                    'angles_list': sorted(results.keys())
                }
            elif test_name.startswith('deviation_'):
                if results:
                    # Filter to numeric keys only (exclude reference image string keys)
                    numeric_angles = [k for k in results.keys() if isinstance(k, (int, float))]
                    summary['results'][test_name] = {
                        'images_acquired': len(results),
                        'angle_range': [min(numeric_angles), max(numeric_angles)] if numeric_angles else [],
                        'reference_images': [k for k in results.keys() if isinstance(k, str)]
                    }
                else:
                    summary['results'][test_name] = {
                        'images_acquired': 0,
                        'angle_range': [],
                        'error': 'No images acquired - acquisition may have timed out'
                    }
            elif test_name == 'repeatability':
                # Include both overall and per-type statistics
                summary['results'][test_name] = {
                    'overall': results.get('overall_statistics', {}),
                    'by_type': results.get('statistics_by_type', {})
                }
            elif test_name == 'analysis':
                summary['results'][test_name] = results.get('summary', {})

        # Save JSON summary
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Save human-readable report
        report_file = self.output_dir / "test_report.txt"
        with open(report_file, 'w') as f:
            f.write("PPM ROTATION SENSITIVITY TEST REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Test Date: {summary['test_date']}\n")
            f.write(f"Config: {summary['config_file']}\n")
            f.write(f"Output: {summary['output_directory']}\n\n")

            f.write("TESTS PERFORMED:\n")
            f.write("-" * 40 + "\n")
            for test in summary['tests_performed']:
                f.write(f"  - {test}\n")

            if 'repeatability' in self.test_results:
                overall = self.test_results['repeatability'].get('overall_statistics', {})
                by_type = self.test_results['repeatability'].get('statistics_by_type', {})

                f.write(f"\nMECHANICAL REPEATABILITY:\n")
                f.write("=" * 50 + "\n")

                if overall:
                    f.write(f"\nOVERALL (all movement types combined):\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"  Total measurements: {overall.get('total_measurements', 0)}\n")
                    f.write(f"  Mean error: {overall.get('mean_error', 0):.4f} degrees\n")
                    f.write(f"  2-sigma repeatability: {overall.get('repeatability_2sigma', 0):.4f} degrees\n")
                    f.write(f"  Maximum error: {overall.get('max_error', 0):.4f} degrees\n")

                if by_type:
                    f.write(f"\nBY MOVEMENT TYPE:\n")
                    f.write("-" * 40 + "\n")
                    for test_type, stats in by_type.items():
                        if test_type == 'bidirectional_hysteresis':
                            f.write(f"\n  {test_type} (180 deg total travel):\n")
                            f.write(f"    Mean hysteresis: {stats.get('mean_hysteresis', 0):.4f} degrees\n")
                            f.write(f"    Max hysteresis: {stats.get('max_hysteresis', 0):.4f} degrees\n")
                        else:
                            travel = stats.get('travel_deg', 0)
                            f.write(f"\n  {test_type} ({travel} deg travel):\n")
                            f.write(f"    Mean error: {stats.get('mean_error', 0):.4f} degrees\n")
                            f.write(f"    2-sigma: {stats.get('repeatability_2sigma', 0):.4f} degrees\n")
                            f.write(f"    Max error: {stats.get('max_error', 0):.4f} degrees\n")

                if not overall and not by_type:
                    f.write(f"  Statistics not available\n")

            f.write(f"\nFull details in: {summary_file}\n")

        self.logger.info(f"Test summary saved to {summary_file}")
        self.logger.info(f"Test report saved to {report_file}")

    def save_combined_summary(self):
        """
        Save a combined summary with both motion (repeatability) and intensity analysis.

        Creates a single comprehensive report file that includes:
        - Mechanical repeatability results
        - Intensity sensitivity analysis
        - Zero-rotation baseline measurements
        - Key findings and recommendations
        """
        combined_file = self.output_dir / "combined_summary.txt"

        self.logger.info("Generating combined motion + intensity summary...")

        with open(combined_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PPM ROTATION SENSITIVITY - COMBINED ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Test Date: {datetime.now().isoformat()}\n")
            f.write(f"Config: {self.config_yaml}\n")
            f.write(f"Output: {self.output_dir}\n")
            f.write(f"Images retained: {self.keep_images}\n\n")

            # ================================================================
            # SECTION 1: MECHANICAL MOTION ANALYSIS
            # ================================================================
            f.write("=" * 80 + "\n")
            f.write("PART 1: MECHANICAL MOTION ANALYSIS\n")
            f.write("=" * 80 + "\n\n")

            if 'repeatability' in self.test_results:
                rep = self.test_results['repeatability']
                overall = rep.get('overall_statistics', {})
                by_type = rep.get('statistics_by_type', {})

                f.write("1.1 OVERALL REPEATABILITY\n")
                f.write("-" * 60 + "\n")
                if overall:
                    f.write(f"  Total measurements: {overall.get('total_measurements', 0)}\n")
                    f.write(f"  Mean error: {overall.get('mean_error', 0):.4f} deg\n")
                    f.write(f"  Standard deviation: {overall.get('std_error', 0):.4f} deg\n")
                    f.write(f"  Maximum error: {overall.get('max_error', 0):.4f} deg\n")
                    f.write(f"  2-sigma repeatability: {overall.get('repeatability_2sigma', 0):.4f} deg\n\n")

                    # Interpretation
                    rep_2sigma = overall.get('repeatability_2sigma', 0)
                    if rep_2sigma < 0.05:
                        f.write("  --> EXCELLENT: Repeatability < 0.05 deg (2-sigma)\n\n")
                    elif rep_2sigma < 0.1:
                        f.write("  --> GOOD: Repeatability < 0.1 deg (2-sigma)\n\n")
                    elif rep_2sigma < 0.2:
                        f.write("  --> ACCEPTABLE: Repeatability < 0.2 deg (2-sigma)\n\n")
                    else:
                        f.write(f"  --> WARNING: Repeatability {rep_2sigma:.3f} deg may affect measurements\n\n")
                else:
                    f.write("  No overall statistics available\n\n")

                f.write("1.2 BY MOVEMENT TYPE\n")
                f.write("-" * 60 + "\n")
                if by_type:
                    for test_type, stats in by_type.items():
                        if test_type == 'bidirectional_hysteresis':
                            f.write(f"\n  {test_type}:\n")
                            f.write(f"    Total travel: 180 deg (0 -> 90 -> 0)\n")
                            f.write(f"    Mean hysteresis: {stats.get('mean_hysteresis', 0):.4f} deg\n")
                            f.write(f"    Max hysteresis: {stats.get('max_hysteresis', 0):.4f} deg\n")
                        else:
                            travel = stats.get('travel_deg', 0)
                            f.write(f"\n  {test_type}:\n")
                            f.write(f"    Travel distance: {travel} deg\n")
                            f.write(f"    Mean error: {stats.get('mean_error', 0):.4f} deg\n")
                            f.write(f"    2-sigma: {stats.get('repeatability_2sigma', 0):.4f} deg\n")
                            f.write(f"    Max error: {stats.get('max_error', 0):.4f} deg\n")
                    f.write("\n")
                else:
                    f.write("  No per-type statistics available\n\n")
            else:
                f.write("  No repeatability test data available.\n")
                f.write("  Run repeatability test to measure mechanical precision.\n\n")

            # ================================================================
            # SECTION 2: INTENSITY SENSITIVITY ANALYSIS
            # ================================================================
            f.write("=" * 80 + "\n")
            f.write("PART 2: INTENSITY SENSITIVITY ANALYSIS\n")
            f.write("=" * 80 + "\n\n")

            # Check for analysis results (from PPMRotationAnalyzer)
            analysis_summary_file = self.output_dir / "analysis" / "summary.txt"
            if analysis_summary_file.exists():
                f.write("(See detailed analysis in: analysis/summary.txt)\n\n")

                # Extract key metrics from analysis results
                if 'analysis' in self.test_results:
                    analysis = self.test_results['analysis']
                    f.write("2.1 IMAGES ANALYZED\n")
                    f.write("-" * 60 + "\n")
                    f.write(f"  Total images: {analysis.get('n_images_loaded', 'N/A')}\n")
                    angles = analysis.get('angles_available', [])
                    if angles:
                        f.write(f"  Angle range: {min(angles):.2f} to {max(angles):.2f} deg\n")
                    f.write("\n")
            else:
                f.write("  Analysis not yet run or no results available.\n\n")

            # Look for zero-rotation baseline CSV
            baseline_csv = self.output_dir / "zero_rotation_baseline.csv"
            if baseline_csv.exists():
                import pandas as pd
                try:
                    df_baseline = pd.read_csv(baseline_csv)
                    # Filter out sanity check
                    df_actual = df_baseline[df_baseline['angle'] != -999.0]

                    f.write("2.2 ZERO-ROTATION BASELINE (Measurement Noise Floor)\n")
                    f.write("-" * 60 + "\n")
                    f.write("  Images acquired at SAME angle without rotation.\n")
                    f.write("  Expected difference: ~0%\n\n")

                    if not df_actual.empty:
                        mean_pct = df_actual['pct_change'].mean()
                        max_pct = df_actual['pct_change'].max()
                        mean_ssim = df_actual['ssim'].mean()

                        f.write(f"  Pairs analyzed: {len(df_actual)}\n")
                        f.write(f"  Mean intensity change: {mean_pct:.4f}%\n")
                        f.write(f"  Max intensity change: {max_pct:.4f}%\n")
                        f.write(f"  Mean SSIM: {mean_ssim:.6f}\n\n")

                        if mean_pct < 0.1:
                            f.write("  --> GOOD: Baseline noise is low. Measurements are valid.\n\n")
                        elif mean_pct < 1.0:
                            f.write("  --> MODERATE: Baseline noise present. Consider in analysis.\n\n")
                        else:
                            f.write("  --> WARNING: High baseline noise indicates measurement issues.\n\n")
                except Exception as e:
                    f.write(f"  Error reading baseline data: {e}\n\n")

            # Look for adjacent differences CSV
            adjacent_csv = self.output_dir / "adjacent_differences.csv"
            if adjacent_csv.exists():
                import pandas as pd
                try:
                    df_adj = pd.read_csv(adjacent_csv)

                    f.write("2.3 ANGULAR SENSITIVITY (Intensity Change per Degree)\n")
                    f.write("-" * 60 + "\n")

                    # Group by step size
                    fine_steps = df_adj[df_adj['delta_deg'] < 0.3]
                    medium_steps = df_adj[(df_adj['delta_deg'] >= 0.3) & (df_adj['delta_deg'] < 1.0)]

                    if not fine_steps.empty:
                        mean_pct_fine = fine_steps['pct_change'].mean()
                        mean_delta_fine = fine_steps['delta_deg'].mean()
                        sensitivity_fine = mean_pct_fine / mean_delta_fine if mean_delta_fine > 0 else 0
                        f.write(f"  Fine steps (<0.3 deg): {mean_pct_fine:.4f}% avg change\n")
                        f.write(f"    -> Sensitivity: ~{sensitivity_fine:.3f}% per degree\n\n")

                    if not medium_steps.empty:
                        mean_pct_med = medium_steps['pct_change'].mean()
                        mean_delta_med = medium_steps['delta_deg'].mean()
                        sensitivity_med = mean_pct_med / mean_delta_med if mean_delta_med > 0 else 0
                        f.write(f"  Medium steps (0.3-1.0 deg): {mean_pct_med:.4f}% avg change\n")
                        f.write(f"    -> Sensitivity: ~{sensitivity_med:.3f}% per degree\n\n")

                except Exception as e:
                    f.write(f"  Error reading adjacent differences: {e}\n\n")

            # ================================================================
            # SECTION 3: COMBINED FINDINGS AND RECOMMENDATIONS
            # ================================================================
            f.write("=" * 80 + "\n")
            f.write("PART 3: COMBINED FINDINGS AND RECOMMENDATIONS\n")
            f.write("=" * 80 + "\n\n")

            # Calculate combined metrics
            rep_2sigma = 0
            baseline_noise = 0
            intensity_sensitivity = 0

            if 'repeatability' in self.test_results:
                rep_2sigma = self.test_results['repeatability'].get(
                    'overall_statistics', {}).get('repeatability_2sigma', 0)

            f.write("3.1 COMBINED ERROR BUDGET\n")
            f.write("-" * 60 + "\n")
            f.write(f"  Mechanical repeatability (2-sigma): {rep_2sigma:.4f} deg\n")

            # Estimate intensity error from mechanical error
            if rep_2sigma > 0:
                # Assume ~1% intensity change per degree as rough estimate
                estimated_intensity_error = rep_2sigma * 1.0
                f.write(f"  Estimated intensity error from motion: ~{estimated_intensity_error:.3f}%\n")
            f.write("\n")

            f.write("3.2 RECOMMENDATIONS\n")
            f.write("-" * 60 + "\n")

            if rep_2sigma < 0.1:
                f.write("  [OK] Mechanical precision is sufficient for PPM imaging.\n")
            else:
                f.write("  [!] Consider mechanical improvements to reduce positioning error.\n")

            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("END OF COMBINED REPORT\n")
            f.write("=" * 80 + "\n")

        self.logger.info(f"Combined summary saved to {combined_file}")
        return combined_file

    def cleanup_images(self):
        """
        Remove acquired .tif image files to conserve disk space.

        Called automatically at end of test if keep_images=False.
        Preserves analysis outputs (CSV, JSON, PNG, TXT files).
        """
        if self.keep_images:
            self.logger.info("keep_images=True, preserving all .tif files")
            return 0

        self.logger.info("Cleaning up .tif files to conserve disk space...")

        # Count and remove .tif files in output directory
        tif_files = list(self.output_dir.glob("*.tif"))
        removed_count = 0

        for tif_file in tif_files:
            try:
                tif_file.unlink()
                removed_count += 1
                self.logger.debug(f"Removed: {tif_file.name}")
            except Exception as e:
                self.logger.warning(f"Failed to remove {tif_file.name}: {e}")

        # Also clean up analysis subdirectory if it exists
        analysis_dir = self.output_dir / "analysis"
        if analysis_dir.exists():
            analysis_tifs = list(analysis_dir.glob("*.tif"))
            for tif_file in analysis_tifs:
                try:
                    tif_file.unlink()
                    removed_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to remove {tif_file.name}: {e}")

        self.logger.info(f"Removed {removed_count} .tif files")
        return removed_count

    def run_comprehensive_test(self):
        """Run all tests in sequence."""
        self.logger.info("=" * 70)
        self.logger.info("PPM ROTATION SENSITIVITY - COMPREHENSIVE TEST")
        self.logger.info("=" * 70)

        if not self.connect():
            self.logger.error("Failed to connect to server. Aborting.")
            return

        try:
            # 1. Repeatability test first (quick, doesn't need images)
            self.run_repeatability_test(test_angle=7.0, n_repeats=10)

            # 2. Standard angles acquisition (each angle uses its own fixed exposure)
            self.run_standard_angles_test()

            # 3. Zero-rotation baseline test (should show 0% change)
            self.run_zero_rotation_baseline_test()

            # 4. Fine deviation tests at PPM-relevant angles
            # Each cluster uses the BASE angle's exposure for all images in that cluster
            for base_angle in [7, 0, -7, 90]:
                self.run_fine_deviation_test(base_angle=base_angle)

            # 4. Optional: Polarizer calibration comparison
            # self.run_polarizer_calibration_comparison()

            # 5. Analyze all results
            self.analyze_results()

            # 6. Save summaries
            self.save_test_summary()
            self.save_combined_summary()

            # 7. Cleanup images if requested
            if not self.keep_images:
                self.cleanup_images()

            self.logger.info("=" * 70)
            self.logger.info("COMPREHENSIVE TEST COMPLETE")
            self.logger.info("=" * 70)
            self.logger.info(f"All results saved to: {self.output_dir}")
            if not self.keep_images:
                self.logger.info("Images were deleted to conserve disk space")

        finally:
            self.disconnect()


def run_ppm_sensitivity_test(
    config_yaml: str,
    output_dir: str = None,
    host: str = "127.0.0.1",
    port: int = 5000,
    test_type: str = "comprehensive",
    base_angle: float = 7.0,
    n_repeats: int = 10,
    keep_images: bool = True,
    angle_exposures: Dict[float, float] = None
) -> Optional[Path]:
    """
    Run PPM rotation sensitivity test programmatically.

    This function can be called from QuPath or other applications to run
    the sensitivity test without CLI interaction.

    Args:
        config_yaml: Path to microscope configuration YAML file
        output_dir: Output directory for results (auto-generated if None)
        host: qp_server host address
        port: qp_server port
        test_type: Type of test ('comprehensive', 'standard', 'deviation',
                   'repeatability', 'calibration')
        base_angle: Base angle for deviation testing
        n_repeats: Number of repetitions for repeatability test
        keep_images: If True, keep acquired .tif images. If False, delete
                    them after analysis to conserve disk space.
        angle_exposures: Optional dict of {angle: exposure_ms} to override
                        default exposures

    Returns:
        Path to output directory on success, None on failure
    """
    # Initialize tester
    tester = PPMRotationSensitivityTester(
        config_yaml=config_yaml,
        output_dir=output_dir,
        host=host,
        port=port,
        angle_exposures=angle_exposures,
        keep_images=keep_images
    )

    # Run selected test
    if test_type == 'comprehensive':
        tester.run_comprehensive_test()
        return tester.output_dir

    # For non-comprehensive tests, handle connection manually
    if not tester.connect():
        print("Failed to connect to server")
        return None

    try:
        if test_type == 'standard':
            tester.run_standard_angles_test()
        elif test_type == 'deviation':
            tester.run_fine_deviation_test(base_angle=base_angle)
        elif test_type == 'repeatability':
            tester.run_repeatability_test(test_angle=base_angle, n_repeats=n_repeats)
        elif test_type == 'calibration':
            tester.run_polarizer_calibration_comparison()

        # Analyze if we acquired images
        if test_type in ['standard', 'deviation']:
            tester.analyze_results()

        # Save summaries
        tester.save_test_summary()
        tester.save_combined_summary()

        # Cleanup if requested
        if not keep_images:
            tester.cleanup_images()

        return tester.output_dir

    finally:
        tester.disconnect()


def main():
    """Main entry point for the rotation sensitivity test."""
    import argparse

    parser = argparse.ArgumentParser(
        description='PPM Rotation Sensitivity Testing'
    )
    parser.add_argument('config_yaml',
                       help='Path to microscope configuration YAML file')
    parser.add_argument('--output', '-o',
                       help='Output directory for test results')
    parser.add_argument('--host', default='127.0.0.1',
                       help='qp_server host address')
    parser.add_argument('--port', type=int, default=5000,
                       help='qp_server port')
    parser.add_argument('--test', choices=['comprehensive', 'standard', 'deviation',
                                          'repeatability', 'calibration'],
                       default='comprehensive',
                       help='Test type to run')
    parser.add_argument('--base-angle', type=float, default=7.0,
                       help='Base angle for deviation testing')
    parser.add_argument('--repeats', type=int, default=10,
                       help='Number of repetitions for repeatability test')
    parser.add_argument('--clean', action='store_true',
                       help='Delete existing output directory before running')
    parser.add_argument('--keep-images', dest='keep_images', action='store_true',
                       default=True,
                       help='Keep acquired .tif images after analysis (default)')
    parser.add_argument('--no-keep-images', dest='keep_images', action='store_false',
                       help='Delete .tif images after analysis to conserve disk space')
    parser.add_argument('--angle-exposures', type=str, default=None,
                       help='JSON dict of {angle: exposure_ms} overrides, e.g. \'{"7.0": 25.0}\'')

    args = parser.parse_args()

    # Handle --clean flag
    if args.clean and args.output:
        import shutil
        output_path = Path(args.output)
        if output_path.exists():
            print(f"Cleaning output directory: {output_path}")
            shutil.rmtree(output_path)
            print("Done - starting fresh")

    # Parse angle exposures if provided
    angle_exposures = None
    if args.angle_exposures:
        try:
            angle_exposures = json.loads(args.angle_exposures)
            # Convert string keys to float
            angle_exposures = {float(k): float(v) for k, v in angle_exposures.items()}
        except Exception as e:
            print(f"Error parsing --angle-exposures: {e}")
            return

    # Run the test using the programmatic interface
    result = run_ppm_sensitivity_test(
        config_yaml=args.config_yaml,
        output_dir=args.output,
        host=args.host,
        port=args.port,
        test_type=args.test,
        base_angle=args.base_angle,
        n_repeats=args.repeats,
        keep_images=args.keep_images,
        angle_exposures=angle_exposures
    )

    if result:
        print(f"\nTest complete. Results saved to: {result}")
        if not args.keep_images:
            print("Note: .tif images were deleted to conserve disk space")
    else:
        print("\nTest failed")


if __name__ == "__main__":
    main()