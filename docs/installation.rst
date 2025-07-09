Installation
============

Prerequisites
-------------

Before installing pyAVS, ensure you have:

- **Python 3.8 or higher**
- **MNE-Python >= 1.0.0** for MEG/EEG analysis
- **FreeSurfer** (optional, for source reconstruction)

Basic Installation
------------------

Install from PyPI
~~~~~~~~~~~~~~~~~

The easiest way to install pyAVS is via pip:

.. code-block:: bash

   pip install pyavs

Install from Source
~~~~~~~~~~~~~~~~~~~

For the latest development version:

.. code-block:: bash

   git clone https://github.com/your-org/pyavs.git
   cd pyavs
   pip install -e .

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For development with all optional dependencies:

.. code-block:: bash

   git clone https://github.com/your-org/pyavs.git
   cd pyavs
   pip install -e ".[dev,full]"

Dependencies
------------

Required Dependencies
~~~~~~~~~~~~~~~~~~~~~

- numpy >= 1.19.0
- pandas >= 1.3.0  
- scipy >= 1.7.0
- matplotlib >= 3.3.0
- h5py >= 3.1.0
- pillow >= 8.0.0
- scikit-image >= 0.18.0
- pycocotools >= 2.0.0
- mne >= 1.0.0

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

For full functionality:

- **autoreject >= 0.3.0**: Advanced artifact rejection
- **scikit-learn >= 1.0.0**: Machine learning algorithms
- **seaborn >= 0.11.0**: Statistical visualization
- **jupyter >= 1.0.0**: Interactive notebooks

Development Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

For development and testing:

- pytest >= 6.0
- pytest-cov >= 2.0
- black >= 21.0
- flake8 >= 3.8
- isort >= 5.0
- sphinx >= 4.0
- sphinx-rtd-theme >= 1.0

FreeSurfer Setup
----------------

For source reconstruction features, install FreeSurfer:

1. Download from https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall
2. Set environment variables:

.. code-block:: bash

   export FREESURFER_HOME=/usr/local/freesurfer
   export SUBJECTS_DIR=/usr/local/freesurfer/subjects
   source $FREESURFER_HOME/SetUpFreeSurfer.sh

Verification
------------

Test your installation:

.. code-block:: python

   import pyavs
   print(pyavs.__version__)
   
   # Check available modules
   print("Available modules:")
   for module in ['data', 'preprocessing', 'scenes', 'source', 'utils']:
       try:
           exec(f"import pyavs.{module}")
           print(f"  ✓ pyavs.{module}")
       except ImportError as e:
           print(f"  ✗ pyavs.{module}: {e}")

Configuration
-------------

Set up your data path:

.. code-block:: python

   import pyavs
   
   # Method 1: Direct configuration
   pyavs.set_data_path('/path/to/avs/dataset')
   
   # Method 2: Environment variable
   # export PYAVS_DATA_PATH=/path/to/avs/dataset
   
   # Method 3: Auto-detection
   pyavs.setup_data_directory()

Create a configuration file (optional):

.. code-block:: bash

   pyavs setup --data-path /path/to/avs/dataset --create-config

This creates ``~/.pyavs_config.json`` with default settings.

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'mne'**

Install MNE-Python:

.. code-block:: bash

   pip install mne

**ModuleNotFoundError: No module named 'pycocotools'**

Install COCO API:

.. code-block:: bash

   pip install pycocotools

**FreeSurfer not found**

Set the environment variable:

.. code-block:: bash

   export FREESURFER_HOME=/usr/local/freesurfer

**Permission errors during installation**

Use virtual environment:

.. code-block:: bash

   python -m venv pyavs_env
   source pyavs_env/bin/activate  # Linux/Mac
   # or
   pyavs_env\Scripts\activate  # Windows
   pip install pyavs

Platform-Specific Notes
~~~~~~~~~~~~~~~~~~~~~~~

**macOS**

If you encounter issues with matplotlib backends:

.. code-block:: bash

   export MPLBACKEND=TkAgg

**Windows**

Use Anaconda/Miniconda for easier dependency management:

.. code-block:: bash

   conda install -c conda-forge pyavs

**Linux**

Some distributions may require additional packages:

.. code-block:: bash

   # Ubuntu/Debian
   sudo apt-get install python3-dev python3-tk
   
   # CentOS/RHEL
   sudo yum install python3-devel tkinter

Getting Help
------------

If you encounter installation issues:

1. Check the `troubleshooting guide <https://pyavs.readthedocs.io/en/latest/troubleshooting.html>`_
2. Search existing `GitHub issues <https://github.com/your-org/pyavs/issues>`_
3. Create a new issue with:
   - Operating system and version
   - Python version
   - Full error message
   - Installation method used