# Changelog

All notable changes to pyAVS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Complete MEG processing pipeline with Maxwell filtering, bandpass filtering, and resampling
- ICA artifact removal with automatic eye movement and cardiac component detection
- MEG-ET temporal alignment and synchronized epoch creation
- Source reconstruction with beamformer and minimum norm estimation methods
- Population code analysis for experimental conditions
- ROI-based analysis with FreeSurfer and Glasser atlas integration
- Command line interface for common workflows
- Comprehensive documentation with ReadTheDocs integration
- Batch processing capabilities for multiple subjects/sessions
- BIDS-compliant data handling with legacy format support

### Changed
- Enhanced main API functions to support full MEG+ET pipeline
- Updated `load_and_preprocess()` to include MEG preprocessing options
- Improved `get_epochs()` to create MEG epochs from eye tracking events
- Reorganized package structure for better modularity

### Fixed
- Eye tracking event temporal alignment with MEG data
- BIDS path handling for both legacy and standard formats
- Configuration management across different server environments

## [0.1.0] - 2024-07-07

### Added
- Initial release of pyAVS package
- Eye tracking data loading and preprocessing
- MSCOCO object detection and fixation mapping
- Basic MEG data loading functionality
- BIDS dataset integration
- Configuration management for multiple server environments
- Data validation and quality assessment
- Scene cropping and visualization utilities
- Comprehensive examples and tutorials

### Features
- **Eye Tracking Processing**
  - Load and preprocess eye tracking events
  - Fixation sequence analysis  
  - Cross-event information (saccade-fixation relationships)
  - Multi-saccade detection and correction
  - Quality metrics and artifact detection

- **Scene Analysis**
  - MSCOCO object detection integration
  - Fixation-to-object mapping
  - Scene image cropping around fixations
  - Visualization tools for eye tracking patterns

- **Data Management**
  - BIDS-compliant data organization
  - Flexible configuration system
  - Data integrity validation
  - Multi-environment support (UOS, MPI, local)

- **API Design**
  - Clean, intuitive Python API
  - Comprehensive docstring documentation
  - Error handling and validation
  - Extensible architecture for future features

### Technical Details
- Python 3.8+ support
- Dependencies: numpy, pandas, scipy, matplotlib, mne, pycocotools
- Modular package structure with clear separation of concerns
- Comprehensive test coverage (planned)
- CI/CD integration (planned)

### Known Limitations
- MEG preprocessing is basic (enhanced in unreleased version)
- Source reconstruction not yet implemented (added in unreleased version)
- Limited visualization options (expanded in unreleased version)

## Development Roadmap

### Version 0.2.0 (Planned)
- [ ] Advanced statistical analysis tools
- [ ] Interactive visualization widgets
- [ ] Performance optimizations
- [ ] Extended documentation and tutorials

### Version 0.3.0 (Planned)  
- [ ] Real-time processing capabilities
- [ ] Cloud computing integration
- [ ] Advanced machine learning features
- [ ] Plugin architecture for extensions

### Version 1.0.0 (Planned)
- [ ] Stable API with backward compatibility guarantees
- [ ] Complete feature set for AVS dataset analysis
- [ ] Production-ready performance and reliability
- [ ] Comprehensive validation and benchmarking

## Contributors

- **P. Sulewski** - Lead Developer

## Acknowledgments

- MNE-Python project for MEG/EEG analysis tools
- MSCOCO dataset for object annotations
- BIDS community for data standardization
- FreeSurfer team for anatomical processing tools