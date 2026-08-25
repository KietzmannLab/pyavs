Python Package
==================

pyAVS is the Python package for loading and preprocessing the AVS dataset -- MEG, eye
tracking, and structural MRI, ported from the analysis pipelines behind the AVS dataset
paper into a documented, reusable library.

.. toctree::
   :maxdepth: 2
   :hidden:

   composer_guide
   ../tutorials/index
   ../examples/index
   ../api/index

Three Steps: Configure, Compose, Analyze
---------------------------------------------

1. **Configure** -- point pyAVS at your local copy of the dataset once, with
   :func:`pyavs.set_data_path` or the ``pyavs configure`` CLI command (see
   :doc:`../installation`).
2. **Compose** -- use :class:`~pyavs.preprocessing.composer.AVSComposer` (see
   :doc:`composer_guide`) to load MEG data, run ICA/filtering, and align eye-tracking events
   into epoched MEG data with rich metadata.
3. **Analyze** -- source-reconstruct, extract ROI/population codes, or hand epochs off to
   your own analysis code. See :doc:`../tutorials/index` and :doc:`../examples/index` for
   worked examples across preprocessing, source reconstruction, object detection, and
   configuration/reproducibility.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: AVSComposer Guide
      :link: composer_guide
      :link-type: doc

      The recommended entry point for MEG + eye-tracking data fusion, with a full worked
      example and advanced configuration options.

   .. grid-item-card:: Tutorials
      :link: ../tutorials/index
      :link-type: doc

      Narrative, end-to-end walkthroughs: MEG + eye tracking, and source reconstruction +
      population codes.

   .. grid-item-card:: Examples
      :link: ../examples/index
      :link-type: doc

      Focused, runnable examples for source reconstruction, cross-session filters,
      configuration, object detection, and reproducibility.

   .. grid-item-card:: API Reference
      :link: ../api/index
      :link-type: doc

      Full API documentation, organized by submodule (preprocessing, source, scenes,
      dataloader, and more).
