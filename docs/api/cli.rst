Command Line Interface (pyavs.cli)
===================================

The CLI module provides command-line tools for common pyAVS workflows.

CLI Commands
------------

.. automodule:: pyavs.cli
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Check Data Availability
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pyavs check-data --subject 1 --session 1 --data-path /path/to/data

Preprocess Data
~~~~~~~~~~~~~~~

.. code-block:: bash

   # MEG + eye tracking preprocessing
   pyavs preprocess --subject 1 --session 1 --blocks 1 2 3 --apply-ica
   
   # Eye tracking only
   pyavs preprocess --subject 1 --session 1 --include-meg false

Create Epochs
~~~~~~~~~~~~~

.. code-block:: bash

   # Fixation-locked MEG epochs
   pyavs create-epochs --subject 1 --session 1 \
       --event-type fixation --sensor-type meg \
       --tmin -0.2 --tmax 0.5 --baseline -0.2 0 --save

Source Reconstruction
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Beamformer source reconstruction
   pyavs source-reconstruction --subject 1 --session 1 \
       --method beamformer --roi-labels V1 V2 V4 --save-source-data

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Process multiple subjects
   pyavs batch --subjects 1 2 3 4 5 --sessions 1 2 \
       --workflow preprocess --parallel --n-jobs 4

Setup Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Create configuration file
   pyavs setup --data-path /path/to/avs/dataset \
       --freesurfer-dir /usr/local/freesurfer/subjects \
       --create-config