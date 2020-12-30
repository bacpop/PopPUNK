Distributing PopPUNK models
===========================
If you have fitted a model yourself, you may be interested in distributing it so that
others can use it for your species. This will give consistent cluster names across datasets,
mean the high-quality tested fit can be reused, and speeds up future analysis.

Please contact us at poppunk@poppunk.net. We would be happy to add your sketches and
fitted model to our other `databases <https://poppunk.net/pages/databases.html>`__.

Database contents
-----------------
A database requires the following files:

- ``.h5``. The sketch database, a HDF5 file.
- ``.dists.pkl`` and ``.dists.npy`` files. Distances for all vs all samples in the sketch database.
- ``_fit.npz`` and ``_fit.pkl`` files. Python files which describe the model fit.
- ``_graph.gt``. The network relating distances, fit and strain assignment for all samples in the sketch database.
- ``_clusters.csv``. The strain assignment of all samples in the sketch database.

If you used a :ref:`lineage-fit` you will also need:

- ``rank_k_fit.npz``. Distances for each rank :math:`k` fit.
- ``_lineages.csv``. Combined lineage assignments for each rank.

You may also have ``.refs`` versions of these files, which are pruned to contain just the
reference samples (see below). We would highly recommend including the ``output/output.refs`` file
with any database, even though it is not strictly required, as it will speed up query assignment.
Lineage models do not use references.

.. note::
    If the database is very large, you may consider just distributing the ``.refs`` files. This will
    enable query assignment, but visualisation and subclustering within strains will no longer be
    possible, as full information within each strain will be missing.

Picking references
------------------
PopPUNK automatically prunes redundant sequence information from databases by removing
samples from cliques (where every sample is in the same strain as every other sample). This
algorithm has changed slightly from the originally published one:

#. Split the graph into connected components (strains), which are analysed in parallel.
#. Identify a clique. If no samples in the clique are already references, add one sample as a reference.
#. Prune the clique from the graph, creating a subgraph.
#. Recursively apply steps 2-3 until only two samples or fewer remain.
#. Add the remaining samples as references
#. Create the reference graph, and find connected components again.
#. For any samples which are no longer in the same connected component, find a minimum path
   between them in the full graph, and add all samples in this path as references.

This makes the algorithm scale better, and ensures clusters remain connected. You may find
that more references are picked than before using this method, which is a small cost for the
increase robustness.

This process occurs automatically after the model fit. In the *Listeria* example::

    Removing 97 sequences

31 strains are represented by :math:`128 - 97 = 31` references, exactly one reference
per cluster, which is the minimum. The refined fit removes 93 sequences with 29 strains,
so some larger clusters need to be represented by multiple references. The names of the chosen
references are written to the .refs file. In addition, the distances, sketch database and graph
have the non-reference sequences pruned and saved with .refs suffixes. This gives a complete database
suitable for assignment with references only, should the full database be prohibitively large.

.. note::
    Previous fans (users) of PopPUNK may remember the ``--full-db`` option which switched off
    reference picking. This was useful, as reference-only databases always lost information. This
    option has now been removed, and reference picking will always be run. Both full and reference
    databases are always produced (apart from in lineage mode). The default assignment uses
    just references, but has the full database available for strain visualisation and subclustering.

If you interrupt the reference picking the output will still be valid. If you wish to
run reference picking on a database where it is missing (due to being from an older version,
or interrupted) you can do this with the ``poppunk_references`` script.
