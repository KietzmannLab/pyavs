Remote Access (pyavs.remote)
==================================

.. note::

   Loads data directly from the public AWS S3 release bucket, on demand, without downloading
   the whole dataset first. See :doc:`../data_access` for details and :doc:`../quickstart` for
   a walkthrough.

``AVSRemote`` mirrors the local loaders (:func:`pyavs.load_meg_raw`,
:func:`pyavs.load_experiment_log`, etc.): each method fetches the file(s) it needs from S3 into
a local cache, then delegates to the existing loader function unchanged -- "the same loaders, a
different byte source."

Client
------

.. automodule:: pyavs.remote.client
   :members:
   :undoc-members:
   :show-inheritance:

Content-Indexed Epoch Queries
----------------------------------

.. automodule:: pyavs.remote.query
   :members:
   :undoc-members:
   :show-inheritance:

S3 Store
--------

.. automodule:: pyavs.remote.store
   :members:
   :undoc-members:
   :show-inheritance:
