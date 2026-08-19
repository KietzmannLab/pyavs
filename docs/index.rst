pyAVS
=========

.. image:: https://badge.fury.io/py/pyavs.svg
   :target: https://badge.fury.io/py/pyavs
   :alt: PyPI version

.. image:: https://readthedocs.org/projects/pyavs/badge/?version=latest
   :target: https://pyavs.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

**pyAVS** is the companion Python package for the **Active Visual Semantics (AVS) dataset**:
MEG, eye tracking, and structural MRI recorded while participants freely explored natural
scenes -- active vision, rather than the passive, fixation-enforced viewing used in most
existing neuroimaging datasets.

:bdg-primary:`5 participants` :bdg-primary:`10 sessions each` :bdg-primary:`4,080 scenes`
:bdg-primary:`200,000+ fixation epochs` :bdg-primary:`306-channel MEG`
:bdg-primary:`1000 Hz eye tracking`

What Is AVS?
----------------

.. note::

   AVS is described in a manuscript currently in preparation (Sulewski, Amme, König,
   Hebart & Kietzmann) -- see :doc:`reference/citation`. The summary below is drawn from
   that manuscript's abstract.

Unlike existing neuroimaging datasets that rely on passive viewing with enforced central
fixation, AVS captures brain activity during active scene exploration, including
self-generated saccades and fixations, across five participants who freely explored 4,080
natural scenes over 10 sessions each. A semantic captioning task on 25% of trials links gaze
to scene understanding and memory. Alongside neural and behavioural data, AVS includes
per-fixation object category labels, human-rated caption-relevance annotations, pupil
dynamics, and individual head-stabilization casts paired with structural MRI scans for
precise cross-session source reconstruction.

.. figure:: _static/images/avs-overview.png
   :alt: AVS dataset overview combining MEG recordings and eye tracking with natural scene understanding
   :width: 100%

   Dataset design combining MEG recordings and eye tracking with natural scene understanding.
   Adapted from Sulewski et al., 2025.

Getting Started
--------------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Installation
      :link: installation
      :link-type: doc

      Install pyAVS and configure your local copy of the dataset.

   .. grid-item-card:: Quickstart
      :link: quickstart
      :link-type: doc

      Load MEG + eye-tracking data and build your first epochs in a few minutes.

   .. grid-item-card:: Dataset at a Glance
      :link: dataset_at_a_glance
      :link-type: doc

      Key numbers, modalities, and a task-oriented "what do I need?" guide.

   .. grid-item-card:: Data Access
      :link: data_access
      :link-type: doc

      What's in the release, how it will be hosted, and how to download only the parts
      you need.

Citation
------------

If you use the AVS dataset or pyAVS, please cite the dataset paper:

    Sulewski, P., Amme, C., König, P., Hebart, M. N., & Kietzmann, T. C. *Active Visual
    Semantics: A large-scale MEG and eye-tracking dataset for understanding visual
    intelligence in action.* Manuscript in preparation.

See :doc:`reference/citation` for the full citation, BibTeX, and how to cite the software.

Contributors
----------------

Philip Sulewski, Carmen Amme, Peter König, Martin N. Hebart, and Tim C. Kietzmann.
Corresponding: phsulewski@gmail.com, tim.kietzmann@uni-osnabrueck.de.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   installation
   quickstart
   dataset_at_a_glance
   data_access

.. toctree::
   :maxdepth: 1
   :caption: Dataset
   :hidden:

   dataset/overview
   dataset/known_issues

.. toctree::
   :maxdepth: 1
   :caption: Methods
   :hidden:

   methods/index

.. toctree::
   :maxdepth: 1
   :caption: Example Analyses
   :hidden:

   analyses/index

.. toctree::
   :maxdepth: 2
   :caption: Python Package
   :hidden:

   package/index
   tutorials/index
   examples/index
   api/index

.. toctree::
   :maxdepth: 1
   :caption: Reference
   :hidden:

   reference/citation
   reference/faq
   reference/terms_of_use
   reference/contributing
   reference/changelog
   reference/license

Indices and Tables
-----------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
