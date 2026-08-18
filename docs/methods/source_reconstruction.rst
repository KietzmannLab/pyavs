Source Reconstruction
=========================

Individual forward models were computed using three-layer boundary element models (BEMs)
based on FreeSurfer cortical reconstructions (Fischl, 2012), with source spaces at 4,098
vertices per hemisphere (ico4 spacing). Session-specific noise covariance was estimated from
400 ms pre-scene baseline periods, pooled across all 10 sessions, using Oracle Approximating
Shrinkage (OAS). The inverse solution used loose orientation constraint 0.2, depth weighting
0.8, and an assumed SNR of 3 (lambda-squared = 1/9). Source estimates were computed with
dSPM (Dale et al., 2000) as implemented in MNE-Python, and morphed to the ``fsaverage``
template (ico5 spacing, smoothing = 5).

Category-averaged event-related fields (ERFs) were averaged over a +/-10 ms window centered
on the group-level sensor-space representational-similarity peak latency (114 ms
post-fixation onset -- see :doc:`../analyses/index`). A geodesic searchlight (20 mm radius)
was applied across the cortical surface using MNE-RSA (van Vliet et al., 2025). Visual
regions of interest (early, lateral, ventral, parietal) were defined following the NSD
cortical ROI scheme (Allen et al., 2022); frontal regions (dlPFC, FEF, OFC, mPFC, infFC) and
hippocampus were defined as compound regions from the Glasser parcellation (Glasser et al.,
2016).

Running This With pyAVS
----------------------------

pyAVS's source module (:mod:`pyavs.source`, see :doc:`../api/source`) implements this
pipeline: :func:`pyavs.create_bem_model` / :func:`pyavs.create_source_space` /
:func:`pyavs.setup_coregistration` / :func:`pyavs.load_forward_model` for forward modeling,
:func:`pyavs.compute_beamformer_filters` and :func:`pyavs.apply_source_reconstruction` for
beamformer-based reconstruction, and :func:`pyavs.get_roi_labels` /
:func:`pyavs.get_glasser_roi_labels` / :func:`pyavs.extract_roi_data` for ROI-level
summarization. See :doc:`../tutorials/source_reconstruction_population_codes` for a guided
walkthrough and :doc:`../examples/source_reconstruction_examples` for runnable code.

.. note::

   pyAVS's example pipelines use LCMV beamforming rather than dSPM by default (both are
   supported); the dSPM/BEM/fsaverage configuration described above is specifically the one
   used for the source-projected fixation ERFs reported in the dataset paper.
