Fitting new models
==================

If you cannot find an existing model for your species in the
`list <https://poppunk.net/pages/databases.html>`__ you will want to fit your own.
This process is flexible, and there are five different models you can use depending
on the population structure of your dataset.

First, you must use ``poppunk --create-db`` to sketch your input data and calculate distances
between all samples. This is detailed in :doc:`sketching`.

Then, you wil use ``poppunk --fit-model <model_name>`` with one of the following:

- ``bgmm`` -- Bayesian Gaussian Mixture Model. Best for small sample collections
  with strain-structure. Works best when distance distribution components are clearly
  separated.
- ``dbscan`` -- HDBSCAN. A good general method for larger sample collections with
  strain-structure. Some points will always be designated as noise, so a subsequent run
  of model refinement may help improve the fit.
- ``refine`` -- Model refinement. Requires a model already fitted with ``bgmm`` or ``dbscan``
  and attempts to improve it by maximising the network score. Particularly useful when
  components overlap significantly (often due to recombination), or when the strain boundary
  is thought to lie somewhere within a component.
- ``lineage`` -- Lineage clustering. To find lineages within a strain (subclustering), or
  find clusters in a population without strain structure. Uses a simple nearest neighbour approach
  so is more of a heuristic. Network scores are not meaningful in this mode.
- ``threshold`` -- Apply a given core or accessory distance threshold to define clusters. Useful if
  a cutoff threshold is already known/calculated, is estimated from a plot, or to compare a threshold
  between datasets or species.

The most useful guide to deciding which model to use is the ``_distanceDistribution.png`` file
showing the core and accessory distances.

bgmm
----


dbscan
------


refine
------

.. _manual-start:

Using fit refinement when mixture model totally fails
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If the mixture model does not give any sort of reasonable fit to the points,
you can manually provide a file with ``--manual-start`` to give the starting parameters to
``--refine-fit`` mode. The format of this file is as follows::

    mean0 0,0
    mean1 0.5,0.6
    start_point 0.3

A key, followed by its value (space separated).

``mean0`` and ``mean1`` define the points (x,y) to draw the line between, and
``start_point`` is the distance along this line to draw the initial boundary
(which is normal to the line).

lineage
-------


threshold
---------

Use an existing model with new data
-----------------------------------

There is also one further mode, ``--use-model``, which may be useful in limited circumstances. This
applies any of the above models to a new dataset without refitting it. This may be useful if a reference
dataset has changed (been added to or removed from) and you do not wish to refit the model, for example
because it is already in use. However, typically you would use :doc:`query_assignment` with ``--update-db``
to add to a model.
