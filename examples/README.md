# pyAVS Examples

This directory contains example scripts demonstrating various pyAVS functionalities.

## Source Reconstruction Examples (NEW)

### 🔥 `simple_source_reconstruction.py`
**Minimalistic source reconstruction example that works without real AVS data**

- ✅ **Self-contained**: Works without requiring actual AVS dataset
- 🧠 **Demonstrates**: Complete source reconstruction workflow
- 💾 **Shows**: New unified HDF5 data saving system
- 🚀 **Perfect for**: Getting started, understanding the workflow

**What it does:**
1. Creates synthetic MEG data (102 channels, 50 epochs)
2. Performs beamformer source reconstruction  
3. Creates mock ROIs (visual cortex, motor cortex, etc.)
4. Saves data using new pyAVS I/O system
5. Outputs data to `/tmp/pyavs_example/`

**Run it:**
```bash
cd pyavs/examples
python simple_source_reconstruction.py
```

### 📊 `data_loading_example.py`
**Demonstrates how to load and analyze saved pyAVS data**

- 📂 **Loads**: Population codes and epochs from HDF5 files  
- 🔍 **Explores**: Data structure, metadata, and attributes
- 📈 **Analyzes**: Basic population code analysis
- 🎨 **Visualizes**: Creates plots and saves them

**Prerequisites:** Run `simple_source_reconstruction.py` first

**Run it:**
```bash
python data_loading_example.py
```

### 🏗️ `source_reconstruction_example.py` 
**Comprehensive example using real AVS data workflow**

- 🎯 **Full workflow**: MEG loading → preprocessing → source reconstruction → saving
- 📡 **Uses**: AVSComposer for MEG-ET data fusion
- 🧠 **Includes**: Real ROI extraction, population codes computation
- ⚠️ **Requires**: Actual AVS dataset and preprocessed data

## Other Examples

### `avs_composer_example.py`
MEG-ET data fusion using the AVS Composer

### `meg_et_workflow.py` & `meg_et_workflow_simple.py`  
Complete MEG + eye tracking analysis workflows

### `logging_example.py`
Demonstrates pyAVS logging system configuration

## Quick Start

1. **Try the synthetic example first:**
   ```bash
   python simple_source_reconstruction.py
   ```

2. **Then explore the saved data:**
   ```bash
   python data_loading_example.py
   ```

3. **Check the output:**
   - Data saved to: `/tmp/pyavs_example/`
   - Plot saved to: `/tmp/pyavs_analysis_example.png`

## New I/O System Features

These examples showcase the new unified I/O system:

- **🗂️ Unified HDF5 format** for all data types
- **📁 Clean directory structure** (`derivatives/pyavs/...`) 
- **🔧 Simple API**: `pyavs.io.write` and `pyavs.io.read`
- **🔗 Compatible** with original avs-machine-room format
- **📊 Rich metadata** preservation

### Key Functions Demonstrated:

**Writing:**
- `save_population_codes_h5()` - Core saving function
- `save_epochs()` - Save epoched data
- `save_source_data()` - Save source reconstructions

**Reading:**
- `load_population_codes()` - Load population codes
- `find_population_codes_files()` - Discover available files  
- `load_epochs_h5()` - Load epoched data

## File Structure After Running Examples

```
/tmp/pyavs_example/
└── derivatives/
    └── pyavs/
        ├── population_codes/
        │   └── synthetic_500hz_[hash]/
        │       └── sub95-99/
        │           └── as99a_population_codes_synthetic_...h5
        └── sub-99/
            └── ses-01/
                └── epochs/
                    └── sub-99_ses-01_task-avs_synthetic_epochs.h5
```

## Tips

- 🐛 **Debugging**: Check MNE log level with `mne.set_log_level('INFO')`
- 🔧 **Customize**: Modify synthetic data parameters in the examples
- 📁 **Output**: Change `output_dir` to save data elsewhere
- 🎨 **Plotting**: Install matplotlib for visualization features