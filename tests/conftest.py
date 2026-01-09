"""
Shared pytest fixtures for ppm_library tests.

Provides synthetic test data for debayering, background correction, and PPM processing.
"""

import numpy as np
import pytest


@pytest.fixture
def synthetic_bayer_rggb():
    """
    Generate a synthetic Bayer pattern image (RGGB) with known RGB output.

    The image is constructed so that debayering produces predictable RGB values.

    Returns:
        np.ndarray: 512x512 uint8 Bayer image (RGGB pattern)
    """
    size = 512
    bayer = np.zeros((size, size), dtype=np.uint8)

    # RGGB pattern:
    # R G
    # G B

    # Create blocks of uniform color for easy validation
    # Top-left quadrant: Red region
    bayer[0:256:2, 0:256:2] = 200  # R positions
    bayer[0:256:2, 1:256:2] = 50   # G positions
    bayer[1:256:2, 0:256:2] = 50   # G positions
    bayer[1:256:2, 1:256:2] = 50   # B positions

    # Top-right quadrant: Green region
    bayer[0:256:2, 256:512:2] = 50   # R positions
    bayer[0:256:2, 257:512:2] = 200  # G positions
    bayer[1:256:2, 256:512:2] = 200  # G positions
    bayer[1:256:2, 257:512:2] = 50   # B positions

    # Bottom-left quadrant: Blue region
    bayer[256:512:2, 0:256:2] = 50   # R positions
    bayer[256:512:2, 1:256:2] = 50   # G positions
    bayer[257:512:2, 0:256:2] = 50   # G positions
    bayer[257:512:2, 1:256:2] = 200  # B positions

    # Bottom-right quadrant: White region (all channels high)
    bayer[256:512:2, 256:512:2] = 200  # R positions
    bayer[256:512:2, 257:512:2] = 200  # G positions
    bayer[257:512:2, 256:512:2] = 200  # G positions
    bayer[257:512:2, 257:512:2] = 200  # B positions

    return bayer


@pytest.fixture
def expected_rgb_from_bayer_rggb():
    """
    Expected RGB output after debayering the synthetic_bayer_rggb image.

    Returns:
        dict: Dictionary with expected RGB values for each quadrant
    """
    return {
        'top_left': {'R': 200, 'G': 50, 'B': 50},      # Red quadrant
        'top_right': {'R': 50, 'G': 200, 'B': 50},     # Green quadrant
        'bottom_left': {'R': 50, 'G': 50, 'B': 200},   # Blue quadrant
        'bottom_right': {'R': 200, 'G': 200, 'B': 200} # White quadrant
    }


@pytest.fixture
def synthetic_raw_image():
    """
    Generate a synthetic raw image for background correction testing.

    Returns:
        np.ndarray: 512x512 uint16 image with vignetting and uneven illumination
    """
    size = 512

    # Create base tissue signal
    tissue = np.random.randint(8000, 12000, (size, size), dtype=np.uint16)

    # Add vignetting (darker corners) - common optical artifact
    y, x = np.ogrid[:size, :size]
    center_y, center_x = size // 2, size // 2

    # Gaussian-like falloff from center
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_distance = np.sqrt(2 * (size / 2)**2)
    vignette_factor = 1.0 - 0.3 * (distance / max_distance)**2

    # Apply vignetting
    image = (tissue * vignette_factor).astype(np.uint16)

    return image


@pytest.fixture
def synthetic_background_image():
    """
    Generate a synthetic background/flatfield image.

    This represents illumination with no sample present.

    Returns:
        np.ndarray: 512x512 uint16 background image
    """
    size = 512

    # Background should be brighter (no sample absorption)
    background = np.random.randint(14000, 16000, (size, size), dtype=np.uint16)

    # Same vignetting pattern as raw image
    y, x = np.ogrid[:size, :size]
    center_y, center_x = size // 2, size // 2
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_distance = np.sqrt(2 * (size / 2)**2)
    vignette_factor = 1.0 - 0.3 * (distance / max_distance)**2

    background = (background * vignette_factor).astype(np.uint16)

    return background


@pytest.fixture
def uniform_background_image():
    """
    Generate a uniform background image (perfect flatfield).

    Useful for testing background mode detection.

    Returns:
        np.ndarray: 512x512 uint16 uniform background
    """
    size = 512
    background_value = 15000

    # Nearly uniform with minimal camera noise
    background = np.full((size, size), background_value, dtype=np.uint16)
    noise = np.random.randint(-50, 50, (size, size), dtype=np.int32)
    background = np.clip(background.astype(np.int32) + noise, 0, 65535).astype(np.uint16)

    return background


@pytest.fixture
def ppm_angles_standard():
    """
    Standard PPM rotation angles for testing.

    Returns:
        list: List of rotation angles in degrees
    """
    return [0.0, 45.0, 90.0, 135.0]


@pytest.fixture
def synthetic_ppm_intensities():
    """
    Generate synthetic PPM intensity values for birefringence calculation testing.

    Simulates intensity measurements at different polarizer angles.

    Returns:
        np.ndarray: Array of intensity values
    """
    angles = np.array([0, 45, 90, 135])  # degrees
    angles_rad = np.radians(angles)

    # Simulate sinusoidal variation (characteristic of birefringence)
    # I = A + B * cos(2 * (angle - angle_offset))
    amplitude = 100.0
    offset = 150.0
    phase_shift = np.pi / 4  # 45 degrees

    intensities = offset + amplitude * np.cos(2 * (angles_rad - phase_shift))

    # Add some noise
    noise = np.random.normal(0, 5, len(angles))
    intensities = intensities + noise

    return intensities.astype(np.float64)


@pytest.fixture
def sample_background_config():
    """
    Sample background correction configuration for testing.

    Returns:
        dict: Background correction settings
    """
    return {
        'background': {
            'method': 'divide',  # or 'subtract'
            'scaling_factor': 1.0,
            'epsilon': 1.0  # Prevents division by zero
        }
    }
