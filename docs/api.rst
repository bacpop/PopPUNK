Reference documentation
========================

Documentation for module functions (for developers)

.. warning::
    This doesn't build properly on readthedocs. To view, clone and run
    ``cd docs && make html`` then see ``_build/api.html``.

.. contents::
   :local:

assign.py
---------
``poppunk_assign`` main function

.. automodule:: PopPUNK.assign
   :members:

bgmm.py
--------

Functions used to fit the mixture model to a database. Access using
:class:`~PopPUNK.models.BGMMFit`.

.. automodule:: PopPUNK.bgmm
   :members:

dbscan.py
---------

Functions used to fit DBSCAN to a database. Access using
:class:`~PopPUNK.models.DBSCANFit`.

.. automodule:: PopPUNK.dbscan
   :members:

mandrake.py
-----------

.. automodule:: PopPUNK.mandrake
   :members:

models.py
---------

.. automodule:: PopPUNK.models
   :members:

network.py
----------

Functions used to construct the network, and update with new queries. Main
entry point is :func:`~PopPUNK.network.constructNetwork` for new reference
databases, and :func:`~PopPUNK.network.findQueryLinksToNetwork` for querying
databases.

.. automodule:: PopPUNK.network
   :members:

refine.py
---------

Functions used to refine an existing model. Access using
:class:`~PopPUNK.models.RefineFit`.

.. automodule:: PopPUNK.refine
   :members:

plot.py
--------

.. automodule:: PopPUNK.plot
   :members:

sparse_mst.py
-------------

.. automodule:: PopPUNK.sparse_mst
   :members:

sketchlib.py
------------

.. automodule:: PopPUNK.sketchlib
   :members:

utils.py
--------

.. automodule:: PopPUNK.utils
   :members:

visualise.py
------------
``poppunk_visualise`` main function

.. automodule:: PopPUNK.visualise
   :members:

web.py
--------

Functions used by the web API to convert a sketch to an h5 database, then generate visualisations and post results to PopPUNK-web.

.. automodule:: PopPUNK.web
   :members:
