# pyAVS

[![PyPI version](https://badge.fury.io/py/pyavs.svg)](https://badge.fury.io/py/pyavs)
[![Documentation Status](https://readthedocs.org/projects/pyavs/badge/?version=latest)](https://pyavs.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**pyAVS** is the companion Python package for the **Active Visual Semantics (AVS) dataset**:
MEG, eye tracking, and structural MRI recorded while participants freely explored natural
scenes — active vision, rather than the passive, fixation-enforced viewing used in most
existing neuroimaging datasets.

Full dataset documentation, methods pages, and example analyses live on the companion website:
**https://www.kietzmannlab.uni-osnabrueck.de/avs/** (also mirrored on
[ReadTheDocs](https://pyavs.readthedocs.io/)).

## Dataset at a Glance

| | |
|---|---|
| Participants | 5 |
| Sessions | 10 MEG + eye-tracking sessions per participant, plus one anatomical session |
| Stimuli | 4,080 natural scenes from the Natural Scenes Dataset (NSD) |
| Task | Free viewing (4 s/scene) with a verbal scene-captioning task on 25% of trials |
| MEG | 306-channel Elekta Neuromag TRIUX, 1000 Hz |
| Eye tracking | EyeLink 1000, 1000 Hz |
| Fixation epochs | 200,000+ across the dataset, shipped fixation- and saccade-locked with per-epoch metadata |
| Object labels | Per-fixation MS-COCO / COCO-Stuff category labels, 171 categories |
| Anatomy | Defaced individual T1 plus a ready-to-use FreeSurfer `SUBJECTS_DIR` |

AVS is described in a manuscript in preparation (Sulewski, Amme, König, Hebart & Kietzmann) —
see [Citation](#citation). The dataset itself is not yet publicly downloadable; see
[Data Access](https://www.kietzmannlab.uni-osnabrueck.de/avs/data_access.html) for the release
plan and current status.

## Installation

### Prerequisites
- Python 3.8+
- MNE-Python >= 1.0.0
- FreeSurfer (optional, only needed for source reconstruction)

### From PyPI
```bash
pip install pyavs
```

### From source
```bash
git clone https://github.com/KietzmannLab/pyavs.git
cd pyavs
pip install -e .
```

### Development installation
```bash
git clone https://github.com/KietzmannLab/pyavs.git
cd pyavs
pip install -e ".[dev,full]"
```

### Configuration

Point pyAVS at your local copy of the dataset once per machine:

```bash
pyavs configure --data-path /path/to/avs/dataset
```

This writes `~/.config/pyavs/config.json`, which `pyavs.get_data_path()` and the rest of the
package read from automatically. Equivalently, from Python:

```python
import pyavs
pyavs.set_data_path('/path/to/avs/dataset')
```

## Quick Start

### AVSComposer workflow (recommended)

`AVSComposer` is the high-level entry point for MEG + eye-tracking fusion: it loads MEG
blocks, applies ICA and filtering, concatenates blocks per session, finds MEG trigger events,
and aligns eye-tracking events to build epoched MEG data with rich per-epoch metadata.

```python
import pyavs

composer = pyavs.AVSComposer(subject=1, session_num=1, use_precomputed_ica=True)

composer.load_meg_data()
composer.apply_ica_to_blocks()
composer.concatenate_raws_per_session()
composer.find_events_in_raw()
composer.get_et_annotations(event_type="fixation")
composer.make_et_event_epochs(tmin=-0.2, tmax=0.5, event_type="fixation")

epochs = composer.et_epochs
print(composer.get_data_summary())
```

See the [AVSComposer guide](https://www.kietzmannlab.uni-osnabrueck.de/avs/package/composer_guide.html)
for the full range of options (data paths, ICA source, filtering/resampling) and how composer
epochs feed into source reconstruction.

### Functional API

For lower-level control, an older functional API is still available (`AVSComposer` is the
actively developed path — prefer it for new code):

```python
import pyavs

subject_data = pyavs.load_and_preprocess(
    subject_id=1, session=1, include_meg=True, include_eye=True, apply_ica=True
)

epochs, events = pyavs.get_epochs(
    subject_data, event_type='fixation', sensor_type='meg', tmin=-0.2, tmax=0.5
)
```

### Command line interface

```bash
# Check what data is available for a subject/session
pyavs check-data --subject 1 --session 1 --data-path /path/to/data

# Preprocess MEG + eye tracking data
pyavs preprocess --subject 1 --session 1 --blocks 1 2 3 --apply-ica

# Create fixation-locked MEG epochs
pyavs create-epochs --subject 1 --session 1 \
    --event-type fixation --sensor-type meg --tmin -0.2 --tmax 0.5 --save

# Run beamformer source reconstruction
pyavs source-reconstruction --subject 1 --session 1 --method beamformer

# Batch process multiple subjects/sessions
pyavs batch --subjects 1 2 3 --sessions 1 2 --workflow preprocess
```

Run `pyavs --help` or `pyavs <command> --help` for the full set of options.

## Package Structure

```
pyavs/
├── config/          # PyAVSConfig / ConfigManager — data paths & analysis parameters
├── dataloader/       # Loading MEG raws, experiment logs, eye-tracking events, anatomy
├── preprocessing/    # AVSComposer, ICA, MEG filtering, ET preprocessing/alignment, triggers
├── source/           # Forward modeling, BEM, LCMV beamformer filters, ROI/atlas handling
├── scenes/            # Fixation→MS-COCO/COCO-Stuff object mapping, scene crops, embeddings
├── captions/          # Transcribed + official MS-COCO captions, caption embeddings
├── utils/             # Derivatives paths, path/naming conventions, validation, logging
├── visualization/     # ERF/sensor-space plots, eye-tracking-on-scene plotting
├── io/                # HDF5 population-code read/write, reproducibility helpers
├── pilot/             # Loading/enrichment for the pilot-phase eye-tracking dataset
└── cli.py             # `pyavs` command-line entry point
```

At the repository root, alongside the `pyavs/` package:

```
scripts/    # Research analysis pipelines built on the library (decoding, encoding, RSA,
            # source reconstruction, ET/MEG quality, ICA, ...) — one subfolder per analysis
examples/   # Teaching-oriented demonstrations of the library API
docs/       # Sphinx source for the companion website
tests/      # pytest suite
```

## Key Functions

- **Composer workflow**: `AVSComposer`, `MEGETComposer`, `create_et_event_epochs`
- **Eye tracking**: `load_and_enrich_eye_events`, `attach_scene_ids_to_samples`,
  `load_samples_with_scenes`, `validate_samples_scene_assignment`,
  `add_fixation_sequence_position`, `preprocess_eye_events`
- **MEG processing**: `load_meg_raw`, `load_meg_preprocessed`, `apply_maxwell_filter`,
  `compute_ica`, `apply_ica`, `find_eye_components_xy_correlation`,
  `repair_meg_trigger_events`
- **Source reconstruction**: `create_forward_model`, `apply_source_reconstruction`,
  `compute_beamformer_filters`, `extract_roi_data`, `compute_population_codes`,
  `get_glasser_roi_labels`
- **Objects and scenes**: `get_fixated_objects`, `create_fixation_crops`, `EyeTrackingPlotter`
- **Configuration**: `set_data_path`, `get_data_path`, `configure`, `check_data_availability`

See the [full API reference](https://www.kietzmannlab.uni-osnabrueck.de/avs/api/index.html)
for the complete, current top-level surface (`pyavs/__init__.py` is ground truth).

## BIDS Integration

pyAVS handles the translation between BIDS terminology and AVS conventions:

| BIDS Term | AVS Term | Description |
|-----------|----------|-------------|
| `run-XX` | `block` | Experimental block/run |
| `ses-XX` | `session` | Recording session |
| `sub-XX` | `subject` | Participant ID |

## Documentation

- **Companion website**: https://www.kietzmannlab.uni-osnabrueck.de/avs/ — dataset structure,
  known issues, methods pages, example analyses, tutorials, and the full API reference.
- **ReadTheDocs mirror**: https://pyavs.readthedocs.io/

## Citation

If you use the AVS dataset or pyAVS, please cite the dataset paper:

> Sulewski, P., Amme, C., König, P., Hebart, M. N., & Kietzmann, T. C. *Active Visual
> Semantics: A large-scale MEG and eye-tracking dataset for understanding visual intelligence
> in action.* Manuscript in preparation.

See the [citation page](https://www.kietzmannlab.uni-osnabrueck.de/avs/reference/citation.html)
for the full BibTeX entry and how to cite the software itself.

## Contributors

Philip Sulewski, Carmen Amme, Peter König, Martin N. Hebart, and Tim C. Kietzmann.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Support

- [Documentation](https://www.kietzmannlab.uni-osnabrueck.de/avs/)
- [Bug Reports](https://github.com/KietzmannLab/pyavs/issues)
