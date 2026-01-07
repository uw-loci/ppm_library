import numpy as np
from scipy import ndimage
from multiprocessing import Pool
import os
import tifffile


class CPUDebayer:
    """CPU-optimized Bayer pattern demosaicing"""

    def __init__(
        self,
        pattern="RGGB",
        image_bit_clipmax=65535,
        image_dtype=np.uint16,
        convolution_mode="reflect",
    ):
        """Initialize with Bayer pattern type"""
        self.pattern = pattern
        self.image_bit_clipmax = image_bit_clipmax
        self.image_dtype = image_dtype
        self.convolution_mode = convolution_mode
        self._setup_masks()

    def _setup_masks(self):
        """Create color channel masks for the Bayer pattern"""
        patterns = {
            "RGGB": {"R": (0, 0), "G": [(0, 1), (1, 0)], "B": (1, 1)},
            "GRBG": {"R": (0, 1), "G": [(0, 0), (1, 1)], "B": (1, 0)},
            "GBRG": {"R": (1, 0), "G": [(0, 0), (1, 1)], "B": (0, 1)},
            "BGGR": {"R": (1, 1), "G": [(0, 1), (1, 0)], "B": (0, 0)},
        }
        self.masks = patterns[self.pattern]

    def debayer(self, bayer_img):
        """Perform CPU debayering using bilinear interpolation"""
        img = bayer_img.astype(np.float32)
        h, w = img.shape

        # Initialize RGB output
        rgb = np.zeros((h, w, 3), dtype=np.float32)

        # Extract channels directly using slicing (faster than masks)
        r_y, r_x = self.masks["R"]
        rgb[r_y::2, r_x::2, 0] = img[r_y::2, r_x::2]

        for g_y, g_x in self.masks["G"]:
            rgb[g_y::2, g_x::2, 1] = img[g_y::2, g_x::2]

        b_y, b_x = self.masks["B"]
        rgb[b_y::2, b_x::2, 2] = img[b_y::2, b_x::2]

        # Bilinear interpolation kernels
        kernels = {
            "r": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 4,
            "g": np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype=np.float32) / 4,
            "b": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 4,
        }

        # Interpolate each channel
        for i, kernel in enumerate([kernels["r"], kernels["g"], kernels["b"]]):
            rgb[:, :, i] = ndimage.convolve(
                rgb[:, :, i], kernel, mode=self.convolution_mode
            )

        # Restore original values
        rgb[r_y::2, r_x::2, 0] = img[r_y::2, r_x::2]
        for g_y, g_x in self.masks["G"]:
            rgb[g_y::2, g_x::2, 1] = img[g_y::2, g_x::2]
        rgb[b_y::2, b_x::2, 2] = img[b_y::2, b_x::2]

        return np.clip(rgb, 0, self.image_bit_clipmax).astype(self.image_dtype)


def process_image(filepath, pattern="GRBG"):
    """Process single image - used for multiprocessing"""
    # filepath, pattern = args
    # Load image (assuming raw bayer data)
    image = tifffile.imread(filepath)
    assert image.ndim == 2
    # bayer = np.fromfile(filepath, dtype=np.uint8).reshape(2000, 2000)

    # Process
    debayer = CPUDebayer(pattern=pattern)
    rgb = debayer.debayer(image)

    # Save result
    output_path = filepath.replace(".raw", "_rgb.npy")
    np.save(output_path, rgb)
    return output_path


def process_data(image, pattern="GRBG"):
    assert image.ndim == 2
    debayer = CPUDebayer(pattern=pattern)
    rgb = debayer.debayer(image)
    return rgb


def batch_process(file_list, pattern="GRBG", n_workers=None):
    """Process multiple images in parallel"""
    if n_workers is None:
        n_workers = os.cpu_count()

    args = [(f, pattern) for f in file_list]

    with Pool(n_workers) as pool:
        results = pool.map(process_image, args)

    return results


# Usage example
if __name__ == "__main__":
    # Single image
    # bayer = np.random.randint(0, 256, (2000, 2000), dtype=np.uint8)
    # debayer = CPUDebayer(pattern="RGGB")
    # rgb = debayer.debayer(bayer)
    pass
    # Batch processing
    # files = ['image1.raw', 'image2.raw', ...]
    # results = batch_process(files, pattern='RGGB', n_workers=8)
