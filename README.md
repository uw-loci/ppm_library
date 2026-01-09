# PPM Library

Image processing library for polarized light microscopy (PPM) and general microscopy imaging.

> **Part of the QPSC (QuPath Scope Control) system**
> For complete installation instructions, see: https://github.com/uw-loci/QPSC
>
> **Note:** This library can also be used standalone for general microscopy image processing.

## Features

- **PPM Processing**: Polarizer calibration, birefringence analysis, sensitivity testing
- **Image Processing**: Background correction, flatfield correction, tissue detection
- **Debayering**: CPU and GPU-based Bayer pattern demosaicing
- **Camera Calibration**: JAI camera white balance calibration
- **TIFF I/O**: TIFF writing with metadata support

## Installation

**Part of [QPSC (QuPath Scope Control)](https://github.com/uw-loci/QPSC)**

This library has no dependencies on other QPSC packages and can be used standalone.
See the [QPSC Installation Guide](https://github.com/uw-loci/QPSC#automated-installation-windows) for complete QPSC setup.

### Quick Install (from GitHub)

**Standard installation:**
```bash
pip install git+https://github.com/uw-loci/ppm_library.git
```

**With GPU support:**
```bash
pip install "git+https://github.com/uw-loci/ppm_library.git#egg=ppm-library[gpu]"
```

### Development Install (editable mode)

```bash
git clone https://github.com/uw-loci/ppm_library.git
cd ppm_library
pip install -e .

# Or with GPU support:
pip install -e ".[gpu]"
```

**For automated setup**, use the [QPSC setup script](https://github.com/uw-loci/QPSC/blob/main/PPM-QuPath.ps1).

## Quick Start

```python
from ppm_library.imaging.background import BackgroundCorrectionUtils
from ppm_library.debayering import CPUDebayer

# Background correction
corrector = BackgroundCorrectionUtils()
corrected = corrector.apply_flatfield(raw_image, background_image)

# Debayering
debayer = CPUDebayer(pattern='RGGB')
rgb_image = debayer.debayer(bayer_image)
```

## License

MIT License
