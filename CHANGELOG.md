# Changelog

All notable changes to the PPM Library will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.3] - 2026-05-13

### Added

- `biref_blur_sigma` parameter to `analyze_perpendicularity()` for optional Gaussian blur of biref image before threshold gating
- `hsv_blur_sigma` parameter to `analyze_perpendicularity()` for optional Gaussian blur of RGB image before HSV/value validity testing (independent of angle computation)

## [1.3.0] - 2026-03-09

### Added

- Analysis CLI for command-line access to PPM analysis tools
- Region analysis module for spatial analysis of PPM data
- Surface perpendicularity module for orientation measurements
- pandas as optional dependency under `[analysis]` extra

### Changed

- Silenced startup warning when optional dependencies are not installed

## [1.2.0] - 2026-02-27

### Added

- Per-channel white balance calibration in birefringence calibrate mode
- Intensity validation and white balance support for birefringence test
- Stage move callback for socket-based calibrate mode coordination
- Progress callback support for polarizer calibration
- Progress updates during Phase 1 background calibration
- Calibration plotting, quality checking, and mask generation moved into library
- Radial calibrator with circular hue stats, per-spoke refinement, and center refinement
- Timestamped subfolder creation for birefringence test output
- Diagnostic logging for image file loading failures
- codemap/ added to .gitignore

### Changed

- Replaced smart_wsi_scanner imports with modular packages in birefringence_test
- Updated sensitivity_test imports from smart_wsi_scanner to modular packages
- Renamed rectangle terminology to spokes in calibration code
- Adjusted background variation threshold for 2-part scan types
- Updated README to remove GPU debayering references

### Fixed

- Center detection using bounding box instead of centroid
- Center detection for high-magnification sunburst images
- Objective/detector lookup for white balance in birefringence test
- Handling of 2-part scan types in background processing

### Removed

- SunburstCalibrator (replaced by improved radial calibrator)

## [1.1.0] - 2026-01-21

### Added

- Merged PSTACS ppmlibrary into ppm_library
- Comprehensive documentation with example images
- Test images and tissue hero image
- Unit tests for image processing

### Changed

- Updated README hero image to show original + processed side-by-side

### Fixed

- Unicode characters replaced for Windows compatibility
- Hardcoded LZW compression in ome_writer now configurable
- Documentation: removed migration section, clarified phantom images

## [1.0.0] - 2026-01-08

### Added

- Initial release of PPM Library
- Core PPM image processing pipeline
- README with link to QPSC installation instructions
- Python version requirement documentation
- Installation instructions for GitHub-based install
- Troubleshooting section in README

### Changed

- Moved tissue_detection to microscope_control package

### Fixed

- Packaging structure for editable installs
- Moved Python code into ppm_library/ subdirectory for correct packaging
