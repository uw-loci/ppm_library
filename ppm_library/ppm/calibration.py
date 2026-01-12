"""
Polarizer calibration utilities for PPM imaging.

This module contains utilities for calibrating polarized light microscopy (PPM)
rotation stage, including finding crossed polarizer positions.
"""

from typing import Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PolarizerCalibrationUtils:
    """
    Utilities for calibrating polarized light microscopy (PPM) rotation stage.

    This class provides methods for determining crossed polarizer positions,
    which are critical for PPM imaging. These calibration functions should be
    run infrequently - only when optical components or rotation stage are
    physically repositioned or replaced.

    Note on Angle Conventions:
        - OPTICAL ANGLES: User-facing angles in degrees (0-360 deg)
        - HARDWARE POSITIONS: Motor encoder counts (device-specific units)

        For PI Stage: hardware_position = (optical_angle * 1000) + ppm_pizstage_offset
        For Thor Stage: hardware_position = -2 * optical_angle + 276

        The calibration function calculates the hardware offset needed to align
        the optical reference (0 deg) with a physical crossed polarizer position.
    """

    @staticmethod
    def find_crossed_polarizer_positions(
        hardware,
        start_angle: float = 0.0,
        end_angle: float = 360.0,
        step_size: float = 5.0,
        exposure_ms: float = 10.0,
        channel: int = 1,
        min_prominence: float = 0.1,
        logger_instance = None
    ) -> Dict[str, Any]:
        """
        Calibrate polarizer by sweeping rotation angles and finding crossed positions.

        This function rotates the polarization stage through a range of angles,
        captures images at each position, measures intensity, fits the data to
        a sinusoidal function, and identifies local minima corresponding to
        crossed polarizer orientations.

        **When to use this:**
            - After installing or repositioning polarization optics
            - After reseating or replacing the rotation stage
            - When validating polarizer alignment
            - To verify/update rotation_angles in config_PPM.yml

        **NOT needed for:**
            - Regular imaging sessions
            - Between samples
            - After software updates

        Args:
            hardware: PycromanagerHardware instance with PPM methods.
            start_angle: Starting optical angle in degrees (default: 0.0).
            end_angle: Ending optical angle in degrees (default: 360.0).
            step_size: Angular step size in degrees (default: 5.0).
            exposure_ms: Camera exposure time in milliseconds (default: 10.0).
            channel: Which RGB channel to analyze (0=R, 1=G, 2=B, None=mean).
            min_prominence: Minimum prominence for peak detection (default: 0.1).
            logger_instance: Logger for output messages.

        Returns:
            Dictionary containing:
                - 'angles': np.ndarray of optical angles tested
                - 'intensities': np.ndarray of mean intensities
                - 'minima_angles': List of crossed polarizer angles
                - 'minima_intensities': List of intensities at minima
                - 'fit_params': Fitted sine parameters
                - 'fit_curve': Fitted intensity values

        Raises:
            AttributeError: If hardware lacks PPM methods.
            RuntimeError: If image acquisition fails.
            ValueError: If no minima detected.
        """
        # Use scipy imports locally to avoid import issues if not available
        from scipy.optimize import curve_fit
        from scipy.signal import find_peaks

        if logger_instance is None:
            logger_instance = logger

        # Verify hardware has PPM methods
        if not hasattr(hardware, 'set_psg_ticks') or not hasattr(hardware, 'get_psg_ticks'):
            raise AttributeError(
                "Hardware does not have PPM methods initialized. "
                "Check that ppm_optics is set in configuration and not 'NA'."
            )

        # Generate angle array
        angles = np.arange(start_angle, end_angle + step_size, step_size)
        intensities = []

        # Set exposure
        hardware.set_exposure(exposure_ms)

        logger_instance.info(
            f"Starting polarizer calibration sweep: "
            f"{start_angle} deg to {end_angle} deg in {step_size} deg steps"
        )
        logger_instance.info(f"Exposure: {exposure_ms} ms, Channel: {channel if channel is not None else 'mean'}")
        logger_instance.info(f"Expected duration: ~{len(angles) * 0.5:.0f} seconds")

        # Sweep through optical angles
        for i, angle in enumerate(angles):
            # Set rotation angle
            hardware.set_psg_ticks(angle)

            # Capture image
            try:
                img, tags = hardware.snap_image()
            except Exception as e:
                raise RuntimeError(f"Image acquisition failed at angle {angle} deg: {e}")

            # Calculate mean intensity
            if channel is not None and len(img.shape) == 3:
                intensity = float(np.mean(img[:, :, channel]))
            else:
                intensity = float(np.mean(img))

            intensities.append(intensity)

            # Progress indicator
            if (i + 1) % 10 == 0 or i == 0:
                logger_instance.info(f"  Angle {angle:.1f} deg: intensity = {intensity:.1f}")

        intensities = np.array(intensities)
        logger_instance.info(f"Intensity range: {intensities.min():.1f} to {intensities.max():.1f}")

        # Normalize intensities
        intensities_norm = (intensities - intensities.min()) / (intensities.max() - intensities.min())

        # Define sine function
        def sine_func(x, amplitude, frequency, phase, offset):
            return amplitude * np.sin(2 * np.pi * frequency * x + phase) + offset

        # Initial guess (180 deg period for polarizers)
        initial_guess = [0.5, 1.0/180.0, 0.0, 0.5]

        # Fit sine function
        try:
            popt, pcov = curve_fit(sine_func, angles, intensities_norm, p0=initial_guess)
            fit_curve_norm = sine_func(angles, *popt)
            fit_curve = fit_curve_norm * (intensities.max() - intensities.min()) + intensities.min()

            logger_instance.info("Sine fit successful:")
            logger_instance.info(f"  Amplitude: {popt[0]:.4f}")
            logger_instance.info(f"  Frequency: {popt[1]:.6f} (period: {1/popt[1]:.1f} deg)")
            logger_instance.info(f"  Phase: {popt[2]:.4f} rad ({np.degrees(popt[2]):.1f} deg)")
            logger_instance.info(f"  Offset: {popt[3]:.4f}")
        except Exception as e:
            logger_instance.warning(f"Sine fit failed: {e}. Using initial guess.")
            popt = initial_guess
            fit_curve_norm = sine_func(angles, *popt)
            fit_curve = fit_curve_norm * (intensities.max() - intensities.min()) + intensities.min()

        # Find local minima
        inverted = -intensities_norm
        peaks, properties = find_peaks(inverted, prominence=min_prominence, distance=len(angles)/10)

        if len(peaks) == 0:
            raise ValueError(
                "No minima detected. Try adjusting: "
                "exposure_ms, step_size, min_prominence, or angular_range"
            )

        minima_angles = angles[peaks].tolist()
        minima_intensities = intensities[peaks].tolist()

        logger_instance.info(f"Found {len(minima_angles)} crossed polarizer positions:")
        for angle, intensity in zip(minima_angles, minima_intensities):
            logger_instance.info(f"  {angle:.1f} deg: intensity = {intensity:.1f}")

        return {
            'angles': angles,
            'intensities': intensities,
            'minima_angles': minima_angles,
            'minima_intensities': minima_intensities,
            'fit_params': popt,
            'fit_curve': fit_curve
        }

    @staticmethod
    def calibrate_hardware_offset_two_stage(
        hardware,
        coarse_range_deg: float = 360.0,
        coarse_step_deg: float = 5.0,
        fine_range_deg: float = 10.0,
        fine_step_deg: float = 0.1,
        exposure_ms: float = 10.0,
        channel: int = 1,
        logger_instance = None
    ) -> Dict[str, Any]:
        """
        Two-stage calibration to determine exact hardware offset for PPM rotation stage.

        This function performs precise hardware position calibration in two stages:
        1. Coarse sweep: Find approximate locations of crossed polarizer minima
        2. Fine sweep: Determine exact hardware encoder positions for each minimum

        The result provides the exact hardware position that should be set as
        ppm_pizstage_offset in config_PPM.yml.

        **CRITICAL**: This calculates the hardware offset itself, not optical angles.
        Run this ONLY when:
            - Installing or repositioning rotation stage hardware
            - After optical component changes
            - When ppm_pizstage_offset needs recalibration

        Args:
            hardware: PycromanagerHardware instance with PPM methods.
            coarse_range_deg: Full range to sweep in coarse stage (default: 360.0 deg).
            coarse_step_deg: Step size for coarse sweep (default: 5.0 deg).
            fine_range_deg: Range around each minimum for fine sweep (default: 10.0 deg, increased for optical stability).
            fine_step_deg: Step size for fine sweep (default: 0.1 deg).
            exposure_ms: Camera exposure time in milliseconds (default: 10.0).
            channel: Which RGB channel to analyze (0=R, 1=G, 2=B, None=mean).
            logger_instance: Logger for output messages.

        Returns:
            Dictionary containing:
                - 'rotation_device': Name of rotation device (PIZStage or Thor)
                - 'coarse_hardware_positions': Hardware positions tested in coarse sweep
                - 'coarse_intensities': Intensities from coarse sweep
                - 'approximate_minima': Approximate hardware positions of minima
                - 'fine_results': List of dicts with fine sweep data for each minimum
                - 'exact_minima': List of exact hardware positions at intensity minima
                - 'recommended_offset': Hardware position to use as ppm_pizstage_offset
                - 'optical_angles': Optical angles corresponding to exact minima

        Raises:
            AttributeError: If hardware lacks PPM methods or rotation device.
            RuntimeError: If image acquisition fails.
            ValueError: If fewer than 2 minima detected (expected for 360 deg sweep).
        """
        from scipy.optimize import curve_fit
        from scipy.signal import find_peaks

        if logger_instance is None:
            logger_instance = logger

        # Verify hardware has PPM methods
        if not hasattr(hardware, 'rotation_device'):
            raise AttributeError(
                "Hardware does not have rotation_device attribute. "
                "Check that PPM is properly initialized."
            )

        rotation_device = hardware.rotation_device
        logger_instance.info(f"=== TWO-STAGE HARDWARE OFFSET CALIBRATION ===")
        logger_instance.info(f"Rotation device: {rotation_device}")

        # Get current hardware position as reference
        current_hw_pos = hardware.core.get_position(rotation_device)
        logger_instance.info(f"Current hardware position: {current_hw_pos:.1f}")

        # Determine conversion factor based on device type
        if rotation_device == "PIZStage":
            # For PI: 1 deg optical = 1000 encoder counts
            hw_per_deg = 1000.0
            logger_instance.info("PI Stage detected: 1 deg = 1000 encoder counts")
        elif rotation_device == "KBD101_Thor_Rotation":
            # For Thor: Uses ppm_psgticks_to_thor conversion (-2x + 276)
            # For sweep purposes, we treat it as 2 counts per degree
            hw_per_deg = 2.0
            logger_instance.info("Thor Stage detected: 1 deg = 2 encoder counts (approx)")
        else:
            raise ValueError(f"Unknown rotation device: {rotation_device}")

        # ===== STAGE 1: COARSE SWEEP =====
        logger_instance.info("\n--- STAGE 1: COARSE SWEEP ---")

        # Calculate hardware range for coarse sweep
        coarse_hw_range = coarse_range_deg * hw_per_deg
        coarse_hw_step = coarse_step_deg * hw_per_deg

        # Center sweep on current position
        coarse_start_hw = current_hw_pos - (coarse_hw_range / 2)
        coarse_end_hw = current_hw_pos + (coarse_hw_range / 2)

        coarse_hw_positions = np.arange(coarse_start_hw, coarse_end_hw + coarse_hw_step, coarse_hw_step)
        coarse_intensities = []

        hardware.set_exposure(exposure_ms)

        logger_instance.info(
            f"Sweeping {coarse_start_hw:.0f} to {coarse_end_hw:.0f} "
            f"in steps of {coarse_hw_step:.0f} ({len(coarse_hw_positions)} positions)"
        )
        logger_instance.info(f"Expected duration: ~{len(coarse_hw_positions) * 0.5:.0f} seconds")

        for i, hw_pos in enumerate(coarse_hw_positions):
            # Set hardware position directly
            hardware.core.set_position(rotation_device, hw_pos)
            hardware.core.wait_for_device(rotation_device)

            # Capture image
            try:
                img, tags = hardware.snap_image()
            except Exception as e:
                raise RuntimeError(f"Image acquisition failed at hardware position {hw_pos:.0f}: {e}")

            # Calculate mean intensity
            if channel is not None and len(img.shape) == 3:
                intensity = float(np.mean(img[:, :, channel]))
            else:
                intensity = float(np.mean(img))

            coarse_intensities.append(intensity)

            # Progress indicator
            if (i + 1) % 10 == 0 or i == 0:
                logger_instance.info(f"  Position {hw_pos:.0f}: intensity = {intensity:.1f}")

        coarse_intensities = np.array(coarse_intensities)
        logger_instance.info(
            f"Coarse sweep complete. Intensity range: {coarse_intensities.min():.1f} to "
            f"{coarse_intensities.max():.1f}"
        )

        # Fit sine curve to coarse data
        intensities_norm = (coarse_intensities - coarse_intensities.min()) / (
            coarse_intensities.max() - coarse_intensities.min()
        )

        def sine_func(x, amplitude, frequency, phase, offset):
            return amplitude * np.sin(2 * np.pi * frequency * x + phase) + offset

        # Initial guess for sine fit (180 deg period)
        period_hw = 180.0 * hw_per_deg
        initial_guess = [0.5, 1.0/period_hw, 0.0, 0.5]

        try:
            popt, _ = curve_fit(sine_func, coarse_hw_positions, intensities_norm, p0=initial_guess)
            logger_instance.info("Sine fit successful")
        except Exception as e:
            logger_instance.warning(f"Sine fit failed: {e}. Using initial guess.")
            popt = initial_guess

        # Find local minima in coarse sweep
        inverted = -intensities_norm
        min_distance = int(len(coarse_hw_positions) / 4)  # At least 90 deg apart

        # Try with default prominence first
        peaks, properties = find_peaks(inverted, prominence=0.1, distance=min_distance)

        # If we didn't find 2 minima, try with lower prominence
        if len(peaks) < 2:
            logger_instance.warning(
                f"Only found {len(peaks)} minimum with prominence=0.1. "
                "Retrying with lower prominence threshold..."
            )
            peaks, properties = find_peaks(inverted, prominence=0.05, distance=min_distance)

        # If still only one, try finding the global minimum in opposite half
        if len(peaks) < 2:
            logger_instance.warning(
                f"Still only found {len(peaks)} minimum. "
                "Searching for second minimum in opposite 180deg region..."
            )
            # If we have one peak, look for minimum in opposite half of sweep
            if len(peaks) == 1:
                first_min_idx = peaks[0]
                half_len = len(coarse_intensities) // 2

                # Search in opposite half
                if first_min_idx < half_len:
                    # First minimum in first half, search second half
                    second_half_min_idx = half_len + np.argmin(coarse_intensities[half_len:])
                else:
                    # First minimum in second half, search first half
                    second_half_min_idx = np.argmin(coarse_intensities[:half_len])

                peaks = np.array([peaks[0], second_half_min_idx])
                logger_instance.info(f"  Found second minimum at index {second_half_min_idx}")

        approximate_minima = coarse_hw_positions[peaks].tolist()
        logger_instance.info(f"Found {len(approximate_minima)} approximate minima:")
        for hw_pos in approximate_minima:
            logger_instance.info(f"  Hardware position: {hw_pos:.0f}")

        # ===== STAGE 2: FINE SWEEP AROUND EACH MINIMUM =====
        logger_instance.info("\n--- STAGE 2: FINE SWEEP ---")

        fine_hw_range = fine_range_deg * hw_per_deg
        fine_hw_step = fine_step_deg * hw_per_deg

        fine_results = []
        exact_minima = []

        for min_idx, approx_hw_pos in enumerate(approximate_minima):
            logger_instance.info(f"\nFine sweep {min_idx + 1}/{len(approximate_minima)}:")
            logger_instance.info(f"  Centered on hardware position: {approx_hw_pos:.0f}")

            # Calculate fine sweep range
            fine_start_hw = approx_hw_pos - (fine_hw_range / 2)
            fine_end_hw = approx_hw_pos + (fine_hw_range / 2)

            fine_hw_positions = np.arange(fine_start_hw, fine_end_hw + fine_hw_step, fine_hw_step)
            fine_intensities = []

            logger_instance.info(
                f"  Sweeping {fine_start_hw:.1f} to {fine_end_hw:.1f} "
                f"in steps of {fine_hw_step:.1f} ({len(fine_hw_positions)} positions)"
            )

            for hw_pos in fine_hw_positions:
                hardware.core.set_position(rotation_device, hw_pos)
                hardware.core.wait_for_device(rotation_device)

                try:
                    img, tags = hardware.snap_image()
                except Exception as e:
                    raise RuntimeError(f"Fine sweep image acquisition failed at {hw_pos:.1f}: {e}")

                if channel is not None and len(img.shape) == 3:
                    intensity = float(np.mean(img[:, :, channel]))
                else:
                    intensity = float(np.mean(img))

                fine_intensities.append(intensity)

            fine_intensities = np.array(fine_intensities)

            # Find exact minimum
            min_idx_local = np.argmin(fine_intensities)
            exact_hw_pos = fine_hw_positions[min_idx_local]
            exact_intensity = fine_intensities[min_idx_local]

            exact_minima.append(exact_hw_pos)

            logger_instance.info(f"  Exact minimum found:")
            logger_instance.info(f"    Hardware position: {exact_hw_pos:.1f}")
            logger_instance.info(f"    Intensity: {exact_intensity:.1f}")

            fine_results.append({
                'approximate_position': approx_hw_pos,
                'fine_hw_positions': fine_hw_positions,
                'fine_intensities': fine_intensities,
                'exact_position': exact_hw_pos,
                'exact_intensity': exact_intensity
            })

        # ===== CALCULATE RECOMMENDATIONS =====
        logger_instance.info("\n--- CALIBRATION RESULTS ---")

        # Sort minima by hardware position
        exact_minima_sorted = sorted(exact_minima)

        # Recommend the minimum closest to current offset as reference (0 deg)
        # For PI stage, current offset is approximately current_hw_pos
        recommended_offset = exact_minima_sorted[0]

        # Calculate optical angles for all minima relative to recommended offset
        optical_angles = []
        for hw_pos in exact_minima_sorted:
            if rotation_device == "PIZStage":
                optical_angle = (hw_pos - recommended_offset) / hw_per_deg
            elif rotation_device == "KBD101_Thor_Rotation":
                # Thor uses: hw_pos = -2 * angle + 276
                # Need to account for this in angle calculation
                optical_angle = (hw_pos - recommended_offset) / hw_per_deg
            else:
                optical_angle = 0.0
            optical_angles.append(optical_angle)

        logger_instance.info(f"Recommended ppm_pizstage_offset: {recommended_offset:.1f}")
        logger_instance.info(f"Exact minima positions (hardware):")
        for i, (hw_pos, opt_angle) in enumerate(zip(exact_minima_sorted, optical_angles)):
            logger_instance.info(f"  Minimum {i+1}: {hw_pos:.1f} ({opt_angle:.2f} deg optical)")

        return {
            'rotation_device': rotation_device,
            'coarse_hardware_positions': coarse_hw_positions,
            'coarse_intensities': coarse_intensities,
            'approximate_minima': approximate_minima,
            'fine_results': fine_results,
            'exact_minima': exact_minima_sorted,
            'recommended_offset': recommended_offset,
            'optical_angles': optical_angles,
            'hw_per_deg': hw_per_deg
        }

    @staticmethod
    def calibrate_hardware_offset_with_stability_check(
        hardware,
        num_runs: int = 3,
        stability_threshold_counts: float = 50.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run hardware offset calibration multiple times to check optical stability.

        Performs the two-stage calibration multiple times in succession and validates
        that results are consistent. This helps identify optical instability issues
        such as loose mounts, thermal drift, or mechanical backlash.

        Args:
            hardware: PycromanagerHardware instance
            num_runs: Number of calibration runs to perform (default: 3)
            stability_threshold_counts: Maximum acceptable variation in encoder counts (default: 50.0 = 0.05deg)
            **kwargs: Additional arguments passed to calibrate_hardware_offset_two_stage

        Returns:
            Dictionary with calibration results plus stability metrics:
                - 'all_runs': List of all calibration results
                - 'recommended_offset': Average offset from all runs
                - 'offset_std': Standard deviation of offsets (stability metric)
                - 'offset_range': Max - min offsets (stability metric)
                - 'is_stable': Boolean indicating if variation < threshold
                - 'stability_warning': Warning message if unstable

        Raises:
            RuntimeError: If optical instability exceeds threshold
        """
        logger_instance = kwargs.get('logger_instance', logger)

        logger_instance.info("="*70)
        logger_instance.info("POLARIZER CALIBRATION WITH STABILITY CHECK")
        logger_instance.info("="*70)
        logger_instance.info(f"Running {num_runs} calibrations to validate optical stability")
        logger_instance.info(f"Stability threshold: +/-{stability_threshold_counts:.1f} encoder counts")

        all_results = []
        all_offsets = []

        for run_num in range(1, num_runs + 1):
            logger_instance.info(f"\n{'='*70}")
            logger_instance.info(f"CALIBRATION RUN {run_num}/{num_runs}")
            logger_instance.info(f"{'='*70}")

            result = PolarizerCalibrationUtils.calibrate_hardware_offset_two_stage(
                hardware, **kwargs
            )

            all_results.append(result)
            all_offsets.append(result['recommended_offset'])

            logger_instance.info(f"Run {run_num} completed: offset = {result['recommended_offset']:.1f}")

            # Brief pause between runs to allow hardware to settle
            if run_num < num_runs:
                import time
                time.sleep(2.0)

        # Calculate stability metrics
        all_offsets = np.array(all_offsets)

        # Get hardware conversion factor from first run
        hw_per_deg = all_results[0]['hw_per_deg']
        half_rotation_counts = 180.0 * hw_per_deg  # e.g., 180000 for PI stage

        # Normalize offsets to a single 180-degree range
        # IMPORTANT: Crossed polarizers have TWO equivalent minima per 360 degrees
        # (at 0 deg and 180 deg), so we normalize to 180 deg, not 360 deg.
        # This ensures we compare equivalent positions correctly.
        normalized_offsets = all_offsets % half_rotation_counts

        # Calculate statistics on normalized offsets
        std_offset = np.std(normalized_offsets)
        range_offset = np.max(normalized_offsets) - np.min(normalized_offsets)

        # For the recommended offset, use the first run's value normalized to 0-360 deg
        # (user expects a value in the standard range)
        full_rotation_counts = 360.0 * hw_per_deg
        recommended_offset = all_offsets[0] % full_rotation_counts

        logger_instance.info(f"\n{'='*70}")
        logger_instance.info("STABILITY ANALYSIS")
        logger_instance.info(f"{'='*70}")
        logger_instance.info(f"Raw offsets from {num_runs} runs: {all_offsets}")
        logger_instance.info(f"Normalized offsets (mod 180 deg for equivalence): {normalized_offsets}")
        logger_instance.info(f"Note: Crossed polarizers repeat every 180 deg, so 0 deg = 180 deg")
        logger_instance.info(f"Recommended offset (from run 1): {recommended_offset:.1f}")
        logger_instance.info(f"Std deviation: {std_offset:.2f} counts ({std_offset/hw_per_deg:.4f} deg)")
        logger_instance.info(f"Range (max-min): {range_offset:.1f} counts ({range_offset/hw_per_deg:.4f} deg)")

        is_stable = range_offset <= stability_threshold_counts

        if is_stable:
            logger_instance.info(f"RESULT: STABLE - Variation {range_offset:.1f} counts ({range_offset/hw_per_deg:.4f} deg) within threshold")
        else:
            warning_msg = (
                f"WARNING: OPTICAL INSTABILITY DETECTED!\n"
                f"  Variation: {range_offset:.1f} counts ({range_offset/hw_per_deg:.3f} deg)\n"
                f"  Threshold: {stability_threshold_counts:.1f} counts ({stability_threshold_counts/hw_per_deg:.3f} deg)\n"
                f"  Possible causes:\n"
                f"    - Loose polarizer/analyzer mounts\n"
                f"    - Thermal drift in optical components\n"
                f"    - Mechanical backlash in rotation stage\n"
                f"    - Vibration or external disturbances\n"
                f"  Recommendation: Check hardware before proceeding with acquisitions"
            )
            logger_instance.warning(warning_msg)

        return {
            'all_runs': all_results,
            'recommended_offset': float(recommended_offset),
            'offset_std': float(std_offset),
            'offset_range': float(range_offset),
            'is_stable': is_stable,
            'stability_warning': None if is_stable else warning_msg,
            'individual_offsets': all_offsets.tolist(),
            'normalized_offsets': normalized_offsets.tolist(),
            'rotation_device': all_results[0]['rotation_device'],
            'hw_per_deg': all_results[0]['hw_per_deg']
        }
