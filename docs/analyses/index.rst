Example Analyses
=====================

This page showcases the kind of analysis pyAVS's fixation-locked MEG data supports, using a
figure from a manuscript currently in preparation that analyzes the AVS dataset (distinct
from the AVS dataset paper itself, see :doc:`../reference/citation`). It combines a
fixation-aligned dynamic representational similarity analysis (dRSA), a per-fixation
ANN-to-MEG encoding analysis, and source-projected fixation event-related fields (ERFs).

.. important::

   This section describes a real figure from an in-preparation manuscript, reproduced here
   as a *scientific showcase* of what the dataset and pyAVS support -- not as a tutorial to
   be followed step by step. The corresponding analysis code lives in
   ``pyavs/scripts/rsa_analysis/``, ``pyavs/scripts/encoding/``, and
   ``pyavs/scripts/meg_viz/compute_source_erp.py`` (research pipelines, not part of the
   documented package API).

Dynamic Representational Similarity Analysis (dRSA)
----------------------------------------------------------

For each fixation, the MEG time course and the corresponding image crop were extracted. MEG
activity was averaged across fixations targeting the same COCO-Stuff object category (see
:doc:`../methods/object_labeling`), yielding category-level neural representations for 171
object categories. Pairwise correlation distances between category-average MEG patterns were
computed at each time point to form a time-resolved MEG representational dissimilarity
matrix (RDM). In parallel, image crops were passed through a ResNet50 network trained on a
cropped version of Ecoset, and pairwise distances between category-average layer embeddings
formed a model RDM. Neural and model RDMs were compared using Spearman rank correlation.

Inter-Subject Reliability
------------------------------

The grand-average Spearman correlation between MEG RDMs of held-out subject pairs rises
rapidly after fixation onset, peaking within approximately 100-150 ms -- benchmarked against
an inter-subject noise ceiling computed the same way.

Layer-Wise Alignment
-------------------------

Comparing MEG RDMs against RDMs built from different ResNet50 layers shows that higher,
later layers (``avgpool``, ``layer3``) achieve higher peak representational similarity than
earlier layers, consistent with fixation-locked MEG activity reflecting increasingly
object-level (rather than purely low-level visual) representations shortly after fixation
onset.

Per-Fixation ANN-to-MEG Encoding
--------------------------------------

A complementary, model-driven approach: the ResNet50 (Ecoset-cropped) embedding of each
individual fixation crop was used to predict the simultaneously recorded per-sensor MEG time
course via a linear encoding model, fit and evaluated per fixation (not averaged across
repeated presentations, since each fixation in active viewing is a unique, unrepeated
event). Sensor-wise encoding performance (Pearson r) rises steeply after fixation onset,
peaking near 100 ms.

Source-Projected Fixation ERFs
------------------------------------

Fixation-locked responses were also projected into source space (see
:doc:`../methods/source_reconstruction`). Grand-average (n = 5) cortical source amplitude at
100 ms post-fixation onset shows bilateral occipital activation dominating this early time
window; the whole-brain average source amplitude time course peaks shortly after fixation
onset before decaying.

Try It Yourself
--------------------

- :doc:`../tutorials/source_reconstruction_population_codes` and
  :doc:`../examples/source_reconstruction_examples` for the source-reconstruction pipeline
- :doc:`../methods/object_labeling` for the object-category labels used to build the model
  RDM
- :doc:`../api/source` and :doc:`../api/scenes` for the underlying API
