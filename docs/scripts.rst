.. _scripts:

Scripts
=======

Brief documentation on the helper scripts included in the package in the ``/scripts`` directory.
To use these scripts you will need to have a clone of the git repository, or they should also be
installed with the prefix 'poppunk' (e.g to run ``extract_distances.py``, run the command
``poppunk_extract_distances.py``).

.. contents::
   :local:

Easy run mode
-------------
Previous versions of the software had an ``--easy-run`` mode which would run a pipeline of:

- ``--create-db`` to sketch genomes
- ``--fit-model --dbscan`` to fit a flexible model
- ``--refine-model`` to improve this model

This is now available as ``poppunk_easy_run.py`` which will chain calls to ``poppunk``
and ``poppunk_visualise`` to replicate this functionality.

.. _poppunk-iterate:

Iterative PopPUNK
-----------------
You can combine the output from multiple to produce further analysis. For an easy
way to create multiple clusters, try the ``--multi-boundary`` option (:ref:`multi-boundary`).

The script to analyse these is ``poppunk_iterate.py``. Basic use is to provide the
output directory as ``--db``, but run ``--help`` for other common options. This relies on
finding files named ``<db>/<db>_boundary<n>_clusters.csv``, where ``<n>`` is the boundary
iteration number (continuous integers increasing from zero). Clusters must contain at least
two samples.

This script will do the following:

1. Starting from the most specific clusters (nearest the origin), it will iteratively
   add new clusters which are either:

    a) totally new clusters

    b) subsets of existing clusters

    c) existing clusters are subsets of the new cluster.

2. Remove duplicate clusters.

3. Calculate average core distance within this cluster set.

4. Create a tree by nesting smaller clusters within larger clusters they are subsets of.

5. Output the combined clusters, average core distances, and tree.

Adding weights to the network
-----------------------------
Converts binary within-cluster edge weights to the Euclidean core-accessory distance.
This is equivalent to running with ``--graph-weights``::

  poppunk_add_weights <name>_graph.gt <name>.dists <output>

Default output is a graph-tool file. Add ``--graphml`` to save as .graphml instead.

Writing the pairwise distances to an output file
------------------------------------------------
By default PopPUNK does not write the calculated :math:`\pi_n` and :math:`a` distances out, as this
contains :math:`\frac{1}{2}n*(n-1)` rows, which gives a multi Gb file for large datasets.

However, if needed, there is a script available to extract these distances as a text file::

  poppunk_extract_distances.py --distances strain_db.dists --output strain_db.dists.out

Writing network components to an output file
--------------------------------------------
Visualisation of large networks with cytoscape may become challenging. It is possible to extract
individual components/clusters for visualisation as follows::

  poppunk_extract_components.py strain_db_graph.gpickle strain_db

Calculating Rand indices
------------------------
This script allows the clusters formed by different runs/fits/modes of PopPUNK to be compared to each
other. 0 indicates the clusterings are totally discordant, and 1 indicates they are identical.

Run::

  poppunk_calculate_rand_indices.py --input poppunk_gmm_clusters.csv,poppunk_dbscan_cluster.csv

The script will calculate the `Rand index <https://en.wikipedia.org/wiki/Rand_index#Rand_index>`__
and the `adjusted Rand index <https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index>`__
between all pairs of files provided (comma separated) to the ``--input`` argument.
These will be written to the file ``rand.out``, which can be changed using ``--output``.

The ``--subset`` argument can be used to restrict comparisons to include only specific samples
listed in the provided file.

Calculating silhouette indices
------------------------------
This script can be used to find how well the clusters project into core-accessory space by
calculating the `silhoutte index <https://en.wikipedia.org/wiki/Silhouette_(clustering)>`__,
which measures how close samples are to others in their own cluster compared to samples from other
clusters. The silhoutte index is calculated for every sample and takes a value between -1 (poorly matched)
to +1 (well matched). The script reports the average of these indices across all samples, using Euclidean
distances between the (normalised) core and accessory divergences calculated by PopPUNK.

To run::

  poppunk_calculate_silhouette.py --distances strain_db.dists --cluster-csv strain_db_clusters.csv

The following additonal options are available for use with external clusterings (e.g. from hierBAPS):

- ``--cluster-col`` the (1-indexed) column index containing the cluster assignment
- ``--id-col`` the (1-indexed) column index containing the sample names
- ``--sub`` a string to remove from sample names to match them to those in ``--distances``
