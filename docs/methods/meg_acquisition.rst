MEG Acquisition
====================

Brain activity was recorded using a 306-channel whole-head MEG system (Elekta Neuromag
TRIUX, Elekta Oy, Helsinki, Finland), comprising 102 magnetometers and 204 planar
gradiometers, sampled at 1000 Hz with an online bandpass of 0.1-330 Hz.

Head Tracking and Stabilization
-------------------------------------

Five head-position indicator (HPI) coils tracked head position continuously throughout each
session. Before each session, the participant's head shape was digitized using a Polhemus
FASTRAK system. Head stabilization was achieved with individually fitted foam casts milled
from 3D head scans, filling the space between the participant's head and the MEG helmet
while still allowing natural viewing behavior. See :doc:`participants` for the resulting
head-repositioning precision.

Loading MEG Data
---------------------

pyAVS loads raw MEG data via :func:`pyavs.load_meg_raw` (or, more commonly, through
:class:`pyavs.AVSComposer` -- see :doc:`../package/composer_guide`), which expects data in
the naming/session conventions described in :doc:`../dataset/overview`.
