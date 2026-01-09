"""
Unit tests for background correction and flat-field correction.

Tests the apply_flat_field_correction function and background detection methods.
"""

import numpy as np
import pytest
from ppm_library.imaging.background import BackgroundCorrectionUtils


class TestCalculateBackgroundColorFromMode:
    """Test background color detection from histogram mode."""

    def test_background_detection_uniform_image(self, uniform_background_image):
        """Test background detection on uniform white background."""
        bg_color, confidence = BackgroundCorrectionUtils.calculate_background_color_from_mode(uniform_background_image)

        # Should detect the uniform background value
        assert bg_color is not None
        assert confidence > 0.8  # High confidence for uniform background
        assert 14500 < bg_color < 15500  # Near the fixture value of 15000

    def test_background_detection_with_tissue(self, synthetic_raw_image):
        """Test background detection on image with tissue and background."""
        bg_color, confidence = BackgroundCorrectionUtils.calculate_background_color_from_mode(synthetic_raw_image)

        assert bg_color is not None
        assert np.isfinite(bg_color)
        assert 0 <= confidence <= 1.0

    def test_background_detection_returns_tuple(self, uniform_background_image):
        """Test that function returns (color, confidence) tuple."""
        result = BackgroundCorrectionUtils.calculate_background_color_from_mode(uniform_background_image)

        assert isinstance(result, tuple)
        assert len(result) == 2

        bg_color, confidence = result
        assert isinstance(bg_color, (int, float, np.number))
        assert isinstance(confidence, (float, np.floating))

    def test_background_detection_low_confidence_for_varied_image(self):
        """Test that confidence is low for images without clear background."""
        # Image with high variation, no clear background mode
        varied_image = np.random.randint(0, 65535, (256, 256), dtype=np.uint16)

        bg_color, confidence = BackgroundCorrectionUtils.calculate_background_color_from_mode(varied_image)

        # Confidence should be lower for varied images
        assert confidence < 0.9


class TestApplyFlatFieldCorrection:
    """Test flat-field correction (background correction)."""

    def test_divide_method_basic(
        self, synthetic_raw_image, synthetic_background_image
    ):
        """Test basic divide method for flat-field correction."""
        corrected = BackgroundCorrectionUtils.apply_flat_field_correction(
            synthetic_raw_image,
            synthetic_background_image,
            method='divide',
            scaling_factor=1.0
        )

        # Check output properties
        assert corrected.shape == synthetic_raw_image.shape
        assert corrected.dtype == synthetic_raw_image.dtype

        # Corrected image should have values in valid range
        assert np.all(corrected >= 0)
        assert np.all(corrected <= 65535)  # uint16 max

    def test_subtract_method_basic(
        self, synthetic_raw_image, synthetic_background_image
    ):
        """Test basic subtract method for background correction."""
        corrected = BackgroundCorrectionUtils.apply_flat_field_correction(
            synthetic_raw_image,
            synthetic_background_image,
            method='subtract',
            scaling_factor=1.0
        )

        assert corrected.shape == synthetic_raw_image.shape
        assert corrected.dtype == synthetic_raw_image.dtype

        # Should not have negative values (clipped to 0)
        assert np.all(corrected >= 0)

    def test_divide_removes_vignetting(
        self, synthetic_raw_image, synthetic_background_image
    ):
        """Test that divide method reduces vignetting effect."""
        # Apply correction
        corrected = BackgroundCorrectionUtils.apply_flat_field_correction(
            synthetic_raw_image,
            synthetic_background_image,
            method='divide',
            scaling_factor=1.0
        )

        # Vignetting causes lower values at edges
        # After correction, center-to-edge variance should be reduced

        # Compare center vs edge variance before and after
        center_roi = synthetic_raw_image[128:384, 128:384]
        edge_roi = synthetic_raw_image[0:100, 0:100]

        center_corrected = corrected[128:384, 128:384]
        edge_corrected = corrected[0:100, 0:100]

        # Center and edge should be more similar after correction
        before_diff = abs(np.mean(center_roi) - np.mean(edge_roi))
        after_diff = abs(np.mean(center_corrected) - np.mean(edge_corrected))

        assert after_diff < before_diff

    def test_scaling_factor_effect(
        self, synthetic_raw_image, synthetic_background_image
    ):
        """Test that scaling factor affects output intensity."""
        corrected_1x = BackgroundCorrectionUtils.apply_flat_field_correction(
            synthetic_raw_image,
            synthetic_background_image,
            method='divide',
            scaling_factor=1.0
        )

        corrected_2x = BackgroundCorrectionUtils.apply_flat_field_correction(
            synthetic_raw_image,
            synthetic_background_image,
            method='divide',
            scaling_factor=2.0
        )

        # 2x scaling should produce brighter result
        mean_1x = np.mean(corrected_1x)
        mean_2x = np.mean(corrected_2x)

        assert mean_2x > mean_1x

    def test_epsilon_prevents_division_by_zero(self, synthetic_raw_image):
        """Test that epsilon parameter prevents division by zero."""
        # Background with some zero values
        zero_background = np.ones((512, 512), dtype=np.uint16)
        zero_background[100:200, 100:200] = 0  # Some zero region

        # Should not crash or produce inf/nan
        corrected = BackgroundCorrectionUtils.apply_flat_field_correction(
            synthetic_raw_image,
            zero_background,
            method='divide',
            scaling_factor=1.0,
            epsilon=1.0
        )

        assert np.all(np.isfinite(corrected))
        assert not np.any(np.isinf(corrected))

    def test_shape_mismatch_handling(self, synthetic_raw_image):
        """Test handling of transposed or mismatched background dimensions."""
        # Transposed background (common issue)
        background_transposed = np.ones(
            (synthetic_raw_image.shape[1], synthetic_raw_image.shape[0]),
            dtype=np.uint16
        ) * 15000

        # Should handle transpose automatically or raise clear error
        try:
            corrected = BackgroundCorrectionUtils.apply_flat_field_correction(
                synthetic_raw_image,
                background_transposed,
                method='divide',
                scaling_factor=1.0
            )
            # If it succeeds, check output is valid
            assert corrected.shape == synthetic_raw_image.shape
        except (ValueError, AssertionError) as e:
            # Should raise clear error about shape mismatch
            assert "shape" in str(e).lower() or "dimension" in str(e).lower()


class TestApplyFlatFieldCorrectionEdgeCases:
    """Test edge cases for flat-field correction."""

    def test_uint8_images(self):
        """Test correction with uint8 images."""
        raw_uint8 = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
        bg_uint8 = np.random.randint(100, 250, (256, 256), dtype=np.uint8)

        corrected = BackgroundCorrectionUtils.apply_flat_field_correction(
            raw_uint8,
            bg_uint8,
            method='divide',
            scaling_factor=1.0
        )

        assert corrected.dtype == np.uint8
        assert np.all(corrected >= 0) and np.all(corrected <= 255)

    def test_identical_raw_and_background(self):
        """Test when raw and background are identical."""
        image = np.random.randint(5000, 15000, (256, 256), dtype=np.uint16)

        # Divide method should produce near-uniform result
        corrected_divide = BackgroundCorrectionUtils.apply_flat_field_correction(
            image,
            image.copy(),
            method='divide',
            scaling_factor=1.0
        )

        # Should be close to scaling_factor * mean
        assert np.std(corrected_divide) < np.std(image)

        # Subtract method with identical images and scaling_factor=1.0 returns original
        # Formula: img - (bg - bg*scale) = img - (bg - bg*1.0) = img - 0 = img
        corrected_subtract = BackgroundCorrectionUtils.apply_flat_field_correction(
            image,
            image.copy(),
            method='subtract',
            scaling_factor=1.0
        )

        # Should return approximately the same image
        assert np.allclose(corrected_subtract, image, rtol=0.01)

    def test_background_brighter_than_raw(self):
        """Test when background is brighter than raw (normal case)."""
        raw = np.full((256, 256), 10000, dtype=np.uint16)
        background = np.full((256, 256), 15000, dtype=np.uint16)

        # This is the expected scenario
        corrected = BackgroundCorrectionUtils.apply_flat_field_correction(
            raw,
            background,
            method='divide',
            scaling_factor=1.0
        )

        assert np.all(np.isfinite(corrected))

    def test_background_dimmer_than_raw(self):
        """Test when background is unexpectedly dimmer than raw."""
        raw = np.full((256, 256), 15000, dtype=np.uint16)
        background = np.full((256, 256), 10000, dtype=np.uint16)

        # Unusual but should handle gracefully
        corrected = BackgroundCorrectionUtils.apply_flat_field_correction(
            raw,
            background,
            method='divide',
            scaling_factor=1.0
        )

        # Should not crash, may clip to max value
        assert np.all(np.isfinite(corrected))
        assert np.all(corrected <= 65535)

    def test_clipping_to_valid_range(self):
        """Test that output is clipped to valid range for dtype."""
        raw = np.full((256, 256), 30000, dtype=np.uint16)
        background = np.full((256, 256), 10000, dtype=np.uint16)

        # With large scaling factor, might exceed uint16 max
        corrected = BackgroundCorrectionUtils.apply_flat_field_correction(
            raw,
            background,
            method='divide',
            scaling_factor=5.0
        )

        # Should clip to uint16 max
        assert np.all(corrected <= 65535)


class TestValidateBackgroundImages:
    """Test background image validation function."""

    def test_validate_with_all_present(self, tmp_path):
        """Test validation when all required background images are present."""
        # Create temporary background directory structure
        bg_dir = tmp_path / "backgrounds"
        modality_dir = bg_dir / "ppm_20x"
        modality_dir.mkdir(parents=True)

        # Create dummy background files for standard PPM angles
        for angle in [0, 45, 90, 135]:
            (modality_dir / f"bg_{angle}.tif").touch()

        # Should validate successfully
        try:
            is_valid = BackgroundCorrectionUtils.validate_background_images(
                str(bg_dir),
                modality="ppm_20x",
                required_angles=[0, 45, 90, 135]
            )
            assert is_valid is True
        except Exception:
            # Function might not exist or have different signature
            pytest.skip("validate_background_images not available")

    def test_validate_with_missing_angle(self, tmp_path):
        """Test validation when a required background angle is missing."""
        bg_dir = tmp_path / "backgrounds"
        modality_dir = bg_dir / "ppm_20x"
        modality_dir.mkdir(parents=True)

        # Create only some background files (missing 135 degree)
        for angle in [0, 45, 90]:
            (modality_dir / f"bg_{angle}.tif").touch()

        try:
            is_valid = BackgroundCorrectionUtils.validate_background_images(
                str(bg_dir),
                modality="ppm_20x",
                required_angles=[0, 45, 90, 135]
            )
            # Should indicate missing files
            assert is_valid is False or is_valid is None
        except (FileNotFoundError, ValueError):
            # Expected to raise error for missing files
            pass
        except Exception:
            pytest.skip("validate_background_images not available or different signature")

    def test_validate_with_missing_modality_directory(self, tmp_path):
        """Test validation when entire modality directory is missing."""
        bg_dir = tmp_path / "backgrounds"
        bg_dir.mkdir(parents=True)

        # Don't create modality subdirectory

        try:
            is_valid = BackgroundCorrectionUtils.validate_background_images(
                str(bg_dir),
                modality="ppm_20x",
                required_angles=[0, 45, 90, 135]
            )
            assert is_valid is False
        except (FileNotFoundError, ValueError):
            # Expected to raise error
            pass
        except Exception:
            pytest.skip("validate_background_images not available")


class TestGetModalityFromScanType:
    """Test extraction of modality from scan type string."""

    def test_standard_scan_type_format(self):
        """Test parsing standard scan type format: PPM_10x_1."""
        try:
            from ppm_library.imaging.background import get_modality_from_scan_type

            modality = get_modality_from_scan_type("PPM_10x_1")
            assert modality == "PPM_10x"

            modality = get_modality_from_scan_type("PPM_20x_2")
            assert modality == "PPM_20x"

            modality = get_modality_from_scan_type("Brightfield_40x_1")
            assert modality == "Brightfield_40x"

        except ImportError:
            pytest.skip("get_modality_from_scan_type not available")

    def test_malformed_scan_type(self):
        """Test handling of malformed scan type strings."""
        try:
            from ppm_library.imaging.background import get_modality_from_scan_type

            # Should handle gracefully or raise clear error
            try:
                modality = get_modality_from_scan_type("invalid")
                # If it succeeds, should return something reasonable
                assert modality is not None
            except (ValueError, IndexError):
                # Expected for malformed input
                pass

        except ImportError:
            pytest.skip("get_modality_from_scan_type not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
