# PPM Library

Image processing library for polarized light microscopy (PPM) and general microscopy imaging.

## Features

- **PPM Processing**: Polarizer calibration, birefringence analysis, sensitivity testing
- **Image Processing**: Background correction, flatfield correction, tissue detection
- **Debayering**: CPU and GPU-based Bayer pattern demosaicing
- **Camera Calibration**: JAI camera white balance calibration
- **TIFF I/O**: TIFF writing with metadata support

## Installation

```bash
pip install ppm-library
```

For GPU support:
```bash
pip install ppm-library[gpu]
```

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
