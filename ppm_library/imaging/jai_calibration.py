"""
JAI Camera Calibration Utilities.

This module provides calibration tools specific to JAI prism cameras,
including white balance calibration via per-channel exposure adjustment.

The JAI camera uses a 3-sensor prism design (no Bayer filter), which means
white balance must be achieved through per-channel exposure/gain adjustment
rather than software demosaicing corrections.

Calibration Algorithm Overview
------------------------------
1. Capture image with current settings
2. Analyze per-channel histogram (R, G, B)
3. Iteratively adjust per-channel exposure to balance channels
4. If exposure ratio exceeds 2x, compensate with per-channel gain
5. Optionally calibrate black level using dark frame subtraction
6. Save calibration results for use during acquisition

Usage
-----
    from smart_wsi_scanner.imaging.jai_calibration import JAIWhiteBalanceCalibrator

    calibrator = JAIWhiteBalanceCalibrator(hardware)
    results = calibrator.calibrate(target_value=200, tolerance=5)

    # Results contain per-channel exposure and gain settings
    print(results.exposures)  # {'R': 10.5, 'G': 8.2, 'B': 12.1} ms
    print(results.gains)      # {'R': 1.0, 'G': 1.0, 'B': 1.0}

Note
----
This module is JAI camera-specific and requires the JAI camera to be
configured in Micro-Manager. It will not work with other camera types.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WhiteBalanceResult:
    """Results from white balance calibration."""

    # Per-channel exposure times in milliseconds
    exposures_ms: Dict[str, float]

    # Per-channel gain multipliers (1.0 = no gain adjustment)
    gains: Dict[str, float]

    # Black level offsets per channel (for dark frame subtraction)
    black_levels: Dict[str, float]

    # Whether calibration converged successfully
    converged: bool

    # Number of iterations to converge
    iterations: int

    # Final channel means after calibration
    final_means: Dict[str, float]

    # Target value that was used
    target_value: float


@dataclass
class CalibrationConfig:
    """Configuration for white balance calibration."""

    # Target mean value for all channels (0-255 for 8-bit, 0-65535 for 16-bit)
    target_value: float = 200.0

    # Acceptable deviation from target (channels within tolerance are considered balanced)
    tolerance: float = 5.0

    # Maximum iterations before giving up
    max_iterations: int = 20

    # Minimum exposure time in milliseconds
    min_exposure_ms: float = 0.1

    # Maximum exposure time in milliseconds
    max_exposure_ms: float = 1000.0

    # Exposure ratio threshold before applying gain compensation
    # If brightest_channel_exposure / darkest_channel_exposure > this, use gain
    gain_threshold_ratio: float = 2.0

    # Whether to perform black level calibration
    calibrate_black_level: bool = True

    # Number of dark frames to average for black level
    dark_frame_count: int = 5


class JAIWhiteBalanceCalibrator:
    """
    White balance calibrator for JAI prism cameras.

    This calibrator adjusts per-channel exposure times (and optionally gains)
    to achieve balanced color response from the JAI 3-sensor prism camera.

    The JAI camera exposes each color channel independently, so white balance
    is achieved by finding the exposure time for each channel that produces
    equal mean values when imaging a neutral gray or white target.
    """

    def __init__(self, hardware: Any):
        """
        Initialize the calibrator.

        Args:
            hardware: PycromanagerHardware instance with JAI camera configured
        """
        self.hardware = hardware
        self._validate_camera()

    def _validate_camera(self) -> None:
        """Verify that JAI camera is available."""
        # TODO: Check that camera is JAI type
        # camera = self.hardware.core.get_property("Core", "Camera")
        # if camera != "JAICamera":
        #     raise ValueError(f"JAI calibration requires JAI camera, got: {camera}")
        pass

    def calibrate(
        self,
        config: Optional[CalibrationConfig] = None,
        output_path: Optional[Path] = None,
    ) -> WhiteBalanceResult:
        """
        Run white balance calibration.

        This should be run with a neutral gray or white target in the field of view.
        The calibrator will iteratively adjust per-channel exposures until all
        channels produce similar mean values.

        Args:
            config: Calibration configuration. Uses defaults if None.
            output_path: Optional path to save diagnostic output (histograms, log)

        Returns:
            WhiteBalanceResult with calibrated settings
        """
        if config is None:
            config = CalibrationConfig()

        logger.info("Starting JAI white balance calibration")
        logger.info(f"Target value: {config.target_value}, tolerance: {config.tolerance}")

        # TODO: Implement calibration algorithm
        # 1. Get current exposure settings
        # 2. Capture initial image
        # 3. Analyze per-channel means
        # 4. Iteratively adjust exposures
        # 5. Apply gain compensation if needed
        # 6. Optionally calibrate black level

        raise NotImplementedError(
            "JAI white balance calibration not yet implemented. "
            "See TODO_LIST.md for implementation plan."
        )

    def calibrate_black_level(
        self,
        num_frames: int = 5,
    ) -> Dict[str, float]:
        """
        Calibrate black level using dark frames.

        Captures images with the light path blocked to measure sensor dark current
        and fixed pattern noise. The resulting black levels can be subtracted
        from acquired images.

        Args:
            num_frames: Number of dark frames to average

        Returns:
            Dictionary of per-channel black level offsets
        """
        logger.info(f"Calibrating black level with {num_frames} dark frames")

        # TODO: Implement black level calibration
        # 1. Prompt user to block light path (or use shutter if available)
        # 2. Capture num_frames dark images
        # 3. Average and compute per-channel mean
        # 4. Return black level offsets

        raise NotImplementedError("Black level calibration not yet implemented")

    def _capture_and_analyze(self) -> Dict[str, float]:
        """
        Capture an image and return per-channel mean values.

        Returns:
            Dictionary with 'R', 'G', 'B' keys and mean values
        """
        img, tags = self.hardware.snap_image()

        if img is None:
            raise RuntimeError("Failed to capture image")

        # JAI camera returns RGB image (no Bayer)
        if len(img.shape) != 3 or img.shape[2] < 3:
            raise ValueError(f"Expected RGB image, got shape: {img.shape}")

        return {
            'R': float(np.mean(img[:, :, 0])),
            'G': float(np.mean(img[:, :, 1])),
            'B': float(np.mean(img[:, :, 2])),
        }

    def _set_channel_exposure(self, channel: str, exposure_ms: float) -> None:
        """
        Set exposure time for a specific channel.

        Args:
            channel: 'R', 'G', or 'B'
            exposure_ms: Exposure time in milliseconds
        """
        # TODO: Implement JAI per-channel exposure control
        # This requires discovering the correct property names
        # See: dev_tests/jai_property_discovery.py
        raise NotImplementedError("Per-channel exposure control not yet implemented")

    def _set_channel_gain(self, channel: str, gain: float) -> None:
        """
        Set gain for a specific channel.

        Args:
            channel: 'R', 'G', or 'B'
            gain: Gain multiplier (1.0 = no gain)
        """
        # TODO: Implement JAI per-channel gain control
        raise NotImplementedError("Per-channel gain control not yet implemented")

    def save_calibration(
        self,
        result: WhiteBalanceResult,
        output_path: Path,
    ) -> None:
        """
        Save calibration results to YAML file.

        Args:
            result: Calibration results to save
            output_path: Path to save YAML file
        """
        import yaml

        data = {
            'white_balance_calibration': {
                'camera': 'JAICamera',
                'exposures_ms': result.exposures_ms,
                'gains': result.gains,
                'black_levels': result.black_levels,
                'target_value': result.target_value,
                'converged': result.converged,
                'iterations': result.iterations,
            }
        }

        with open(output_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

        logger.info(f"Saved white balance calibration to: {output_path}")

    def load_calibration(self, input_path: Path) -> WhiteBalanceResult:
        """
        Load calibration results from YAML file.

        Args:
            input_path: Path to YAML file

        Returns:
            WhiteBalanceResult loaded from file
        """
        import yaml

        with open(input_path, 'r') as f:
            data = yaml.safe_load(f)

        cal = data['white_balance_calibration']

        return WhiteBalanceResult(
            exposures_ms=cal['exposures_ms'],
            gains=cal['gains'],
            black_levels=cal['black_levels'],
            converged=cal['converged'],
            iterations=cal['iterations'],
            final_means={},  # Not saved
            target_value=cal['target_value'],
        )
