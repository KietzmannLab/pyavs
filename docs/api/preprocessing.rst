Preprocessing (pyavs.preprocessing)
====================================

The preprocessing module provides functions for cleaning and preparing MEG and eye tracking
data for analysis.

AVSComposer
---------------

:class:`~pyavs.preprocessing.composer.AVSComposer` is the recommended high-level entry point
for MEG + eye-tracking data fusion -- see :doc:`../package/composer_guide` for a guided
walkthrough with a full worked example.

.. autoclass:: pyavs.preprocessing.composer.AVSComposer
   :members:
   :undoc-members:
   :show-inheritance:

Eye Tracking Preprocessing
---------------------------

.. automodule:: pyavs.preprocessing.eye
   :members:
   :undoc-members:
   :show-inheritance:

MEG Preprocessing
-----------------

.. automodule:: pyavs.preprocessing.meg
   :members:
   :undoc-members:
   :show-inheritance:

ICA Artifact Removal
---------------------

.. automodule:: pyavs.preprocessing.ica
   :members:
   :undoc-members:
   :show-inheritance:

MEG-ET Alignment
----------------

.. automodule:: pyavs.preprocessing.alignment
   :members:
   :undoc-members:
   :show-inheritance:

Sample-Scene Attachment
----------------------------

.. automodule:: pyavs.preprocessing.samples
   :members:
   :undoc-members:
   :show-inheritance:

MEG Trigger Tools
----------------------

Ported from the original AVS-machine-room trigger repair/analysis code.

.. automodule:: pyavs.preprocessing.trigger.tools
   :members:
   :undoc-members:
   :show-inheritance:

Maxwell-Filter Calibration Data
-------------------------------------

``pyavs.preprocessing.calibration`` is not a code module -- it bundles the MNE Maxwell-filter
calibration data files (``sss_cal*.dat``, ``ct_sparse*.fif``, ``bad_channels.csv``) that
:func:`pyavs.apply_maxwell_filter` reads by default. These files are **not currently
included in the built package distribution** (only ``pyavs/*.py`` and select data files are
packaged) -- if ``apply_maxwell_filter()`` fails to find calibration files after a ``pip
install``, pass explicit calibration file paths, or install from source.