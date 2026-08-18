Eye Tracking
================

Eye movements were recorded with an EyeLink 1000 system (SR Research Ltd., Ottawa, Canada)
at 1000 Hz, calibrated using a 9-point grid at the start of each session and after each
break. A fixation cross preceding each scene served as a drift-correction target; the next
scene appeared only after successful correction. Gaze position data were aligned to the MEG
recording via triggers sent at trial onset.

Precision
-------------

- Mean 9-point calibration error: 0.268 degrees (95% CI [0.227, 0.306])
- Pre-scene drift correction magnitude: median 0.440 degrees (IQR 0.270-0.670 degrees),
  with 92.9% of corrections below 1.0 degree (95% CI [89.4, 96.3])

Event Detection
--------------------

Saccades and fixations were detected using a velocity-based algorithm (Engbert &
Mergenthaler, 2006): for each session, samples exceeding five standard deviations of the
sample-wise velocity were classified as saccades, and all remaining non-blink samples were
classified as fixations. Fixations shorter than 50 ms or longer than 1000 ms, and the last
fixation of each trial, were excluded from analysis.

Gaze Event Counts
----------------------

Across all participants and sessions, scene viewing produced 203,356 fixations (mean 40,671
per subject, 95% CI [37,966, 44,548]), 213,810 saccades (mean 42,762 per subject, 95% CI
[39,684, 46,777]), and 9,954 blinks (mean 1,991 per subject, 95% CI [1,173, 2,841]). The
captioning task (see :doc:`semantic_captioning`) produced an additional 61,624 fixations,
56,047 saccades, and 18,097 blinks.

Fixation durations (n = 199,289, top 2% outliers removed) had a median of 254.4 ms (95% CI
[230.8, 273.6]; IQR 196.0-345.6 ms). Saccade durations (n = 209,541) had a median of 25.2 ms
(95% CI [23.4, 28.0]; IQR 14.2-35.0 ms) -- both consistent with established values for free
viewing of natural scenes.

Loading Eye-Tracking Data
------------------------------

pyAVS loads and enriches eye-tracking events via :func:`pyavs.load_eye_events` and
:func:`pyavs.load_and_enrich_eye_events`, which apply the corrections described in
:doc:`../dataset/known_issues` (e.g. multi-saccade cleanup, trigger-delay correction). See
:doc:`../package/composer_guide` for the recommended combined MEG + eye-tracking workflow.
