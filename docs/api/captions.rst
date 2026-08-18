Captions (pyavs.captions)
==============================

Loading transcribed participant captions and official MS-COCO captions, and computing
multilingual caption embeddings -- see :doc:`../methods/semantic_captioning`.

.. note::

   Like :mod:`pyavs.scenes.embeddings`, this module is intentionally **not** re-exported
   from top-level ``pyavs`` -- it's imported directly by the scripts that use it (e.g.
   ``captions/analyze_caption_similarity.py``), not part of the core top-level API surface.

Loading Captions
---------------------

.. automodule:: pyavs.captions.load
   :members:
   :undoc-members:
   :show-inheritance:

MS-COCO Caption Loading
----------------------------

.. automodule:: pyavs.captions.coco_loader
   :members:
   :undoc-members:
   :show-inheritance:

Caption Embeddings
-----------------------

.. automodule:: pyavs.captions.embedding
   :members:
   :undoc-members:
   :show-inheritance:
