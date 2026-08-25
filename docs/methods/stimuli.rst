Stimuli
===========

Scene Set
-------------

Stimuli comprised 4,080 natural scenes drawn from the Natural Scenes Dataset (NSD; Allen et
al., 2022), including NSD's shared-1000 subset (presented to all NSD participants) and the
special-100 subset (chosen to maximally span the NSD semantic space).

Semantically Balanced Sampling
-----------------------------------

Because internet-sourced image collections such as NSD/MS-COCO over-represent commonly
photographed scene types, stimuli were sampled to give AVS uniform coverage across semantic
content. Sentence-BERT embeddings (``paraphrase-multilingual-mpnet-base-v2``) were computed
for the five MS-COCO captions available per scene and averaged; the resulting embedding
space was partitioned into 60 clusters via K-means, with the number of clusters selected via
a cross-validated silhouette criterion. 68 scenes were then sampled uniformly from each of
the 60 clusters, giving 4,080 scenes with balanced semantic coverage.

.. figure:: ../_static/images/avs-pipeline.png
   :alt: Balanced semantic scene sampling approach
   :width: 100%

   Balanced semantic scene sampling approach. Adapted from Sulewski et al., 2025.

All participants viewed the same stimulus set, each in an individually randomized order.

Presentation
----------------

Scenes were cropped and resized to 947 x 710 pixels (preserving aspect ratio) and displayed
on a 41.6 x 31.2 cm screen (1024 x 768 pixels) at a viewing distance of 70 cm, subtending
28.5 x 21.6 degrees of visual angle. Each scene was shown for 4 seconds, followed by a
1-2 second inter-stimulus interval.

On 25% of trials (selected pseudorandomly but fixed across participants), a microphone icon
appeared for 1 second after scene offset, cueing the verbal scene-description task (see
:doc:`semantic_captioning`), followed by an 8-second response window.
