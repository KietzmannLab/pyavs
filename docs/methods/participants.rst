Participants
================

Five healthy, right-handed participants (3 female, 2 male; mean age 27.8 years, SD 2.6),
all native German speakers with normal or corrected-to-normal vision, took part in the
study. All participants provided written informed consent. The study was approved by the
ethics committee of the University of Leipzig and conducted in accordance with the
Declaration of Helsinki.

Session Schedule
---------------------

Each participant completed 10 MEG + eye-tracking sessions plus one anatomical (MRI)
session. Sessions were spread out over time: the median gap between sessions was 2 days
(IQR [1, 5]; range 1-93 days).

Head Stabilization
-----------------------

To support reliable source reconstruction across sessions recorded on different days,
individualized foam head casts were milled from 3D head scans for each participant, filling
the space between the head and the MEG helmet while still allowing natural, unconstrained
eye and head movement during viewing.

This substantially reduced head-repositioning error relative to conventional surface- or
fiducial-based coregistration:

.. list-table:: Head repositioning error (mean, 95% CI)
   :header-rows: 1

   * - Axis
     - Between-session
     - Within-session (between blocks)
   * - X
     - 1.92 mm [1.57, 2.27]
     - 1.20 mm [0.88, 1.52]
   * - Y
     - 2.07 mm [1.86, 2.27]
     - 1.11 mm [0.87, 1.42]
   * - Z
     - 2.87 mm [2.27, 3.66]
     - 1.75 mm [1.47, 2.12]

These errors are well below the 4-5 mm typical of conventional coregistration approaches,
supporting reliable cross-session source reconstruction at the single-participant level (see
:doc:`source_reconstruction`).
