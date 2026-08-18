Semantic Captioning Task
=============================

To encourage active viewing that serves scene understanding -- and to link gaze behavior to
memory-based report -- participants were asked to verbally describe the preceding scene
(in German) on 25% of trials (see :doc:`stimuli` for the trial-selection details).

Caption Quality
-------------------

Participant-generated captions were compared to the original English MS-COCO annotations
(Lin et al., 2014) using multilingual sentence embeddings
(``distiluse-base-multilingual-cased``; Reimers & Gurevych, 2019). Mean cross-lingual cosine
similarity was 0.53 (95% CI [0.51, 0.55]), compared to a within-COCO self-similarity of 0.59
(95% CI [0.59, 0.59]). Cross-lingual similarity scaled with within-COCO caption agreement
(slope b = 0.61, 95% CI [0.58, 0.64], p < .001), confirming that the German AVS captions
reliably trace the semantic content captured by the original English annotations.

Linking Fixations to Captions
-----------------------------------

To explicitly link gaze behavior to the downstream captioning task, two independent raters
classified each fixation target (n = 45,193 per rater; 5 subjects, 1,020 scenes) as
referenced in the participant's own caption ("self"), in another participant's caption only
("other"), or absent from all captions. On average across raters, 82.81% +/- 12.53%
(mean +/- SD) of fixation targets were classified as "self", 13.06% +/- 12.91% as "other",
and 4.13% +/- 0.38% as absent -- indicating that most fixated content was subsequently
reported by the same participant.

Using This With pyAVS
--------------------------

Caption loading and embedding utilities live in :mod:`pyavs.captions` (see
:doc:`../api/captions`): :func:`pyavs.captions.load.load_captions` for the transcribed
participant captions, and :func:`pyavs.captions.embedding.encode_captions` for computing
multilingual embeddings as used above.
