# pyAVS: Python package for Active Visual Semantics

[![PyPI version](https://badge.fury.io/py/pyavs.svg)](https://badge.fury.io/py/pyavs)
[![Documentation Status](https://readthedocs.org/projects/pyavs/badge/?version=latest)](https://pyavs.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A streamlined Python package for loading and preprocessing MEG + eye-tracking data from the Active Visual Semantics (AVS) BIDS dataset. This package provides a modern, organized interface with enhanced functionality for neuroscience research.

## Features

### Core Functionality
- **AVSComposer Workflow**: Streamlined high-level interface for complete MEG + eye tracking analysis
- **MEG Data Processing**: Complete pipeline from raw data to source reconstruction  
- **Eye Tracking Analysis**: Advanced fixation and saccade detection with scene integration
- **MEG-ET Synchronization**: Temporal alignment and synchronized analysis
- **Samples-Level Analysis**: Individual eye tracking sample processing with scene assignment
- **Source Reconstruction**: Beamformer and minimum norm estimation methods
- **Population Code Analysis**: Advanced encoding analysis for experimental conditions

### Key Capabilities
- **Composer-Based Workflows**: High-level API that handles complex processing pipelines
- **Automated Preprocessing**: Maxwell filtering, bandpass filtering, ICA artifact removal
- **Sample Scene Assignment**: Assign stimulus scene information to individual eye tracking samples
- **Object Detection Integration**: MSCOCO object mapping for visual scenes
- **ROI-based Analysis**: FreeSurfer and Glasser atlas integration
- **Advanced Visualizations**: Eye tracking heatmaps, sample trajectories, and scene overlays
- **BIDS Compatibility**: Support for BIDS-formatted datasets with legacy compatibility
- **Command Line Interface**: Easy-to-use CLI for composer workflows
- **Comprehensive Examples**: Complete workflow demonstrations with composer tutorials

## Installation

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

## Quick Start

### pyAVS Composer Workflow (Recommended)

The **AVSComposer** provides a high-level interface for MEG + eye tracking data analysis, handling data loading, preprocessing, and analysis in a unified workflow:

```python
import pyavs

# Set up data path
pyavs.set_data_path('/path/to/avs/dataset')

# Create composer instance - handles all data loading and preprocessing
composer = pyavs.AVSComposer(
    subject=1, 
    session_num=1,
    preprocessed=True,
    use_precomputed_ica=True,
    verbose=True
)

# Load MEG data with automatic preprocessing
composer.load_meg_data()

# Apply ICA artifact removal across all blocks
composer.apply_ica_to_blocks()

# Load and enrich eye tracking events with scene information
composer.load_eye_events()

# Align MEG and eye tracking data temporally
composer.align_meg_eye_events()

# Create epochs from eye tracking events
composer.create_epochs(
    event_type='fixation',
    tmin=-0.2,
    tmax=0.5,
    baseline=(-0.2, 0)
)

# Perform source reconstruction
composer.compute_source_reconstruction(
    method='beamformer',
    roi_atlas='glasser'
)

# Extract ROI data for analysis
roi_data = composer.extract_roi_data()

print(f"Loaded {len(composer.meg_blocks)} MEG blocks")
print(f"Created {len(composer.epochs)} epochs")
print(f"ROI data shape: {roi_data.shape}")
```

### Alternative: Functional API

For more granular control, you can use the functional API:

```python
import pyavs

# Load and preprocess data step by step
subject_data = pyavs.load_and_preprocess(
    subject_id=1, session=1,
    include_meg=True, include_eye=True, apply_ica=True
)

# Create epochs from eye tracking events
epochs, events = pyavs.get_epochs(
    subject_data, event_type='fixation', sensor_type='meg',
    tmin=-0.2, tmax=0.5
)

# Perform source reconstruction
forward_model = pyavs.load_forward_model(subject_id=1, session=1)
source_data = pyavs.apply_source_reconstruction(
    epochs, forward_model, method='beamformer'
)
```

### Command Line Interface

```bash
# Run complete composer workflow
pyavs composer --subject 1 --session 1 --data-path /path/to/data --apply-ica

# Check data availability
pyavs check-data --subject 1 --session 1 --data-path /path/to/data

# Process samples with scene assignment
pyavs process-samples --subject 1 --session 1 --output samples_with_scenes.csv

# Create eye tracking visualizations
pyavs visualize --subject 1 --session 1 --scene 123 --show-heatmap

# Run source reconstruction with composer
pyavs source-reconstruction --subject 1 --session 1 --method beamformer --roi-atlas glasser

# Batch process multiple subjects with composer
pyavs batch --subjects 1 2 3 --sessions 1 2 --workflow composer --parallel
```

## Data Structure

pyAVS works with BIDS-formatted AVS dataset containing MEG data, eye tracking events, experiment logs, scene images, and object masks.

## Key Functions

### High-Level Composer Workflow (Recommended)
- `AVSComposer`: Main class for streamlined MEG + eye tracking analysis
- `load_meg_data()`: Load and preprocess MEG data across blocks
- `apply_ica_to_blocks()`: Apply artifact removal with ICA
- `load_eye_events()`: Load and enrich eye tracking events
- `align_meg_eye_events()`: Temporal alignment of MEG and eye tracking
- `create_epochs()`: Create epochs from eye tracking events
- `compute_source_reconstruction()`: Beamformer and source analysis
- `extract_roi_data()`: Extract region-of-interest time series

### Eye Tracking Analysis
- `load_and_enrich_eye_events()`: Main function for loading and enriching ET data
- `attach_scene_ids_to_samples()`: Assign scene information to individual samples
- `load_samples_with_scenes()`: Load samples data with scene assignments
- `validate_samples_scene_assignment()`: Validate scene assignment quality
- `add_fixation_sequence_position()`: Add sequence positions to fixations
- `preprocess_eye_events()`: Remove artifacts and outliers

### MEG Processing
- `load_meg_raw()`: Load raw MEG data
- `apply_maxwell_filter()`: Maxwell filtering for noise reduction
- `compute_ica()`: Independent component analysis for artifacts
- `create_et_event_epochs()`: Create epochs aligned to eye tracking events

### Source Reconstruction
- `create_forward_model()`: Build forward model for source localization
- `apply_source_reconstruction()`: Beamformer and minimum norm methods
- `compute_beamformer_filters()`: LCMV beamformer implementation
- `extract_roi_data()`: Extract data from anatomical regions
- `compute_population_codes()`: Population-level encoding analysis

### Object and Scene Analysis
- `get_fixated_objects()`: Map fixations to MSCOCO objects
- `create_fixation_crops()`: Extract image crops around fixations
- `EyeTrackingPlotter`: Visualize fixations and heatmaps on scenes

### Configuration
- `set_data_path()`: Set base data directory
- `setup_data_directory()`: Auto-detect and set data path
- `check_data_availability()`: Validate data integrity

## Package Structure

```
pyavs/
├── __init__.py              # Main API functions
├── cli.py                   # Command line interface
├── config/                  # Configuration management
│   ├── config.py           # Configuration classes
│   └── manager.py          # Configuration manager
├── dataloader/              # Data loading modules
│   ├── loaders.py          # Basic data loading functions
│   ├── eye.py              # Eye tracking specific functions
│   └── meg.py              # MEG data loading
├── preprocessing/           # Data preprocessing
│   ├── alignment.py        # MEG-ET temporal alignment
│   ├── composer.py         # AVSComposer high-level interface
│   ├── eye.py              # Eye tracking preprocessing
│   ├── ica.py              # Independent component analysis
│   ├── meg.py              # MEG preprocessing
│   ├── samples.py          # Eye tracking samples processing
│   └── trigger_tools.py    # Trigger and timing utilities
├── source/                  # Source reconstruction
│   ├── forward.py          # Forward modeling
│   ├── reconstruction.py   # Source localization methods
│   ├── spaces.py           # Source spaces and ROIs
│   └── filters.py          # Spatial filters
├── scenes/                  # Scene-related utilities
│   ├── objects.py          # Object detection and mapping
│   └── crops.py            # Image cropping utilities
├── utils/                   # Utility functions
│   ├── config.py           # Configuration utilities
│   ├── logging.py          # Logging system
│   ├── paths.py            # Path utilities and naming conventions
│   └── validation.py       # Data validation functions
├── visualization/           # Visualization tools
│   ├── events_on_scene.py  # Eye tracking visualization
│   └── meg.py              # MEG visualization
├── io/                      # Input/output operations
│   ├── read.py             # Data reading functions
│   └── write.py            # Data writing functions
└── examples/                # Usage examples
    ├── avs_composer_example.py
    ├── samples_scene_assignment_example.py
    ├── source_reconstruction_example.py
    └── basic_eye_tracking_workflow.py
```

## Configuration

```python
# Set data path
pyavs.set_data_path('/path/to/avs/data')

# Or use environment variable
export PYAVS_DATA_PATH=/path/to/avs/data
```

## Examples and Tutorials

### pyAVS Composer Examples

The `examples/` directory contains workflow demonstrations using the AVSComposer:

- **`avs_composer_example.py`**: Complete MEG + eye tracking pipeline with composer
  - Data loading and preprocessing
  - ICA artifact removal
  - MEG-ET temporal alignment  
  - Source reconstruction and ROI analysis
  - Population code computation

- **`samples_scene_assignment_example.py`**: Eye tracking samples analysis
  - Load individual eye tracking samples (not just events)
  - Assign stimulus scene information to each sample
  - Sample-level analysis and visualization

### Additional Examples

- **`basic_eye_tracking_workflow.py`**: Eye tracking processing using functional API
- **Source reconstruction tutorials**: Complete beamformer and population code workflows

### Running Examples

```bash
# Run the main composer workflow
python pyavs/examples/avs_composer_example.py

# Run samples scene assignment demo
python pyavs/examples/samples_scene_assignment_example.py

# Run basic eye tracking workflow
python pyavs/examples/basic_eye_tracking_workflow.py
```

For complete documentation, see [ReadTheDocs](https://pyavs.readthedocs.io/).

## BIDS Integration

pyAVS handles the translation between BIDS terminology and AVS conventions:

| BIDS Term | AVS Term | Description |
|-----------|----------|-------------|
| `run-XX` | `block` | Experimental block/run |
| `ses-XX` | `session` | Recording session |
| `sub-XX` | `subject` | Participant ID |


## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Support

- [Documentation](https://pyavs.readthedocs.io/)
- [Bug Reports](https://github.com/your-org/pyavs/issues)
- [Discussions](https://github.com/your-org/pyavs/discussions)