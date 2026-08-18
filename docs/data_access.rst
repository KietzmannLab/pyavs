Data Access
===============

.. warning::

   **TODO (pending publication):** the AVS dataset's public hosting location has not yet
   been finalized. The dataset paper itself currently states only that the dataset "will be
   made publicly available upon publication." This page will be updated with the concrete
   download location and access mechanism once that's settled -- do not assume a specific
   repository or URL from any other source.

What Will Be Released
--------------------------

Per the dataset paper (see :doc:`reference/citation`), the release is expected to include:

- Raw and preprocessed MEG data, in BIDS format
- Eye-tracking data with fixation, saccade, and blink annotations
- Per-fixation object category labels
- Transcribed participant scene captions
- Stimulus presentation logs with timing information
- Individual anatomical MRIs (defaced)

Terms of use for the released data are also pending finalization -- see
:doc:`reference/terms_of_use`.

Code
--------

The pyAVS package and the analysis code used in the dataset paper are already public:

.. code-block:: bash

   pip install pyavs
   # or, for the latest development version:
   git clone https://github.com/KietzmannLab/pyavs.git

See :doc:`installation` and :doc:`quickstart` to get started once you have data access.

Staying Updated
--------------------

Watch or star the `pyavs GitHub repository <https://github.com/KietzmannLab/pyavs>`_ for
updates, or check back on this page.
