# pyAVS: Python package for Active Visual Semantics

[![PyPI version](https://badge.fury.io/py/pyavs.svg)](https://badge.fury.io/py/pyavs)
[![Documentation Status](https://readthedocs.org/projects/pyavs/badge/?version=latest)](https://pyavs.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A streamlined Python package for loading and preprocessing MEG + eye-tracking data from the Active Visual Semantics (AVS) BIDS dataset. This package provides a modern, organized interface with enhanced functionality for neuroscience research.

## 🧠 Features

### Core Functionality
- **MEG Data Processing**: Complete pipeline from raw data to source reconstruction
- **Eye Tracking Analysis**: Advanced fixation and saccade detection with scene integration
- **MEG-ET Synchronization**: Temporal alignment and synchronized analysis
- **BIDS Compatibility**: Support for BIDS-formatted datasets with legacy compatibility
- **Source Reconstruction**: Beamformer and minimum norm estimation methods
- **Population Code Analysis**: Advanced encoding analysis for experimental conditions

### Key Capabilities
- ✅ **Automated Preprocessing**: Maxwell filtering, bandpass filtering, ICA artifact removal
- ✅ **Object Detection Integration**: MSCOCO object mapping for visual scenes
- ✅ **ROI-based Analysis**: FreeSurfer and Glasser atlas integration
- ✅ **Command Line Interface**: Easy-to-use CLI for common workflows
- ✅ **Parallel Processing**: Batch processing for multiple subjects/sessions
- ✅ **Comprehensive Examples**: Full workflow demonstrations and tutorials

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- MNE-Python >= 1.0.0
- FreeSurfer (for source reconstruction)

### Install from PyPI
```bash
pip install pyavs
```

### Install from source
```bash
git clone https://github.com/your-org/pyavs.git
cd pyavs
pip install -e .
```

### Development installation
```bash
git clone https://github.com/your-org/pyavs.git
cd pyavs
pip install -e ".[dev,full]"
```

## 🚀 Quick Start

### Basic Usage

```python
import pyavs

# Set up data path
pyavs.set_data_path('/path/to/avs/dataset')

# Load and preprocess MEG + eye tracking data
subject_data = pyavs.load_and_preprocess(
    subject_id=1, 
    session=1,
    include_meg=True,
    include_eye=True,
    apply_ica=True
)

# Create epochs from eye tracking events
epochs, events = pyavs.get_epochs(
    subject_data, 
    event_type='fixation',
    sensor_type='meg',
    tmin=-0.2, 
    tmax=0.5
)

# Perform source reconstruction
forward_model = pyavs.load_forward_model(subject_id=1, session=1)
source_data = pyavs.apply_source_reconstruction(
    epochs, 
    forward_model, 
    method='beamformer'
)

print(f"Created {len(epochs)} epochs")
print(f"Source data shape: {source_data.shape}")
```

### Command Line Interface

```bash
# Check data availability
pyavs check-data --subject 1 --session 1 --data-path /path/to/data

# Preprocess MEG + eye tracking data
pyavs preprocess --subject 1 --session 1 --blocks 1 2 3 --apply-ica

# Create epochs from eye tracking events
pyavs create-epochs --subject 1 --session 1 --event-type fixation --sensor-type meg

# Run source reconstruction
pyavs source-reconstruction --subject 1 --session 1 --method beamformer

# Batch process multiple subjects
pyavs batch --subjects 1 2 3 --sessions 1 2 --workflow preprocess --parallel
```

## Data Structure

pyAVS works with BIDS-formatted AVS dataset containing:

- **MEG data**: Raw .fif files per block/run
- **Eye tracking events**: Preprocessed CSV files with fixations, saccades, and blinks
- **Experiment logs**: Trial information, scene IDs, and experimental conditions
- **Scene images**: MSCOCO scene images used as stimuli
- **Object masks**: Segmentation masks for objects in scenes

## Key Functions

### Data Loading
- `load_eye_events()`: Load eye tracking events and messages
- `load_experiment_log()`: Load experimental trial information
- `load_scenes()`: Load scene image paths
- `load_anatomical()`: Load anatomical MRI data

### Eye Tracking Processing
- `load_and_enrich_eye_events()`: Main function for loading and enriching ET data
- `add_fixation_sequence_position()`: Add sequence positions to fixations
- `add_cross_event_information()`: Add saccade-fixation relationships
- `preprocess_eye_events()`: Remove artifacts and outliers

### Object Analysis
- `get_fixated_objects()`: Map fixations to MSCOCO objects
- `create_fixation_crops()`: Extract image crops around fixations
- `visualize_fixations_on_scene()`: Visualize fixations overlaid on scenes

### Configuration
- `set_data_path()`: Set base data directory
- `setup_data_directory()`: Auto-detect and set data path
- `check_data_availability()`: Validate data integrity

## Package Structure

```
pyavs/
├── __init__.py              # Main API functions
├── dataloader/              # Data loading modules
│   ├── loaders.py          # Basic data loading functions
│   └── eye.py              # Eye tracking specific functions
├── preprocessing/           # Data preprocessing
│   └── eye.py              # Eye tracking preprocessing
├── scenes/                  # Scene-related utilities
│   ├── objects.py          # Object detection and mapping
│   └── crops.py            # Image cropping utilities
├── utils/                   # Utility functions
│   ├── config.py           # Configuration management
│   ├── paths.py            # Path utilities and naming conventions
│   └── validation.py       # Data validation functions
└── examples/                # Usage examples
    └── basic_eye_tracking_workflow.py
```

## Configuration

pyAVS supports multiple server environments:

```python
# Auto-detection
pyavs.setup_data_directory()

# Manual configuration
pyavs.set_data_path('/share/klab/datasets/avs/')  # UOS
pyavs.set_data_path('/data/p_02644/act_vis_sem/') # MPI

# Environment variable
export PYAVS_DATA_PATH=/path/to/avs/data
```

## Examples

See the `examples/` directory for complete workflows:

- `basic_eye_tracking_workflow.py`: Complete eye tracking processing pipeline

Run examples:

```bash
python pyavs/examples/basic_eye_tracking_workflow.py
```

## BIDS Integration

pyAVS handles the translation between BIDS terminology and AVS conventions:

| BIDS Term | AVS Term | Description |
|-----------|----------|-------------|
| `run-XX` | `block` | Experimental block/run |
| `ses-XX` | `session` | Recording session |
| `sub-XX` | `subject` | Participant ID |

## Development Status

### ✅ Completed (Eye Tracking Focus)
- Eye tracking data loading and preprocessing
- Fixation sequence analysis
- Object mapping with MSCOCO integration
- Scene cropping and visualization
- Data validation and quality assessment
- BIDS path handling
- Configuration management

### 🚧 In Development
- MEG data loading and preprocessing
- MEG-ET temporal alignment
- Source reconstruction
- Population code analysis

### 📋 Planned
- Advanced artifact rejection
- Statistical analysis tools
- Interactive visualization
- Documentation and tutorials

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use pyAVS in your research, please cite:

```bibtex
@software{pyavs,
  title={pyAVS: Python Package for Active Visual Semantics Dataset},
  author={Sulewski, P. and Meinert, C.},
  year={2024},
  url={https://github.com/your-org/pyavs}
}
```

## Support

- 📖 [Documentation](https://pyavs.readthedocs.io/)
- 🐛 [Bug Reports](https://github.com/your-org/pyavs/issues)
- 💬 [Discussions](https://github.com/your-org/pyavs/discussions)

## Acknowledgments

- Active Visual Semantics dataset contributors
- MNE-Python project for neuroimaging tools
- MSCOCO dataset for object annotations
- BIDS community for data standardization