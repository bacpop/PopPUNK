.. _scripts:

Scripts
=======

Brief documentation on the helper scripts included in the package in the ``/scripts`` directory.
To use these scripts you will need to have a clone of the git repository, or they should also be
installed with the prefix 'poppunk' (e.g to run ``extract_distances.py``, run the command
``poppunk_extract_distances.py``).

.. contents::
   :local:

Writing the pairwise distances to an output file
------------------------------------------------
By default PopPUNK does not write the calculated :math:`\pi_n` and :math:`a` distances out, as this
contains :math:`\frac{1}{2}n*(n-1)` rows, which gives a multi Gb file for large datasets.

However, if needed, there is a script available to extract these distances as a text file::

  python scripts/extract_distances.py --distances strain_db.dists --output strain_db.dists.out

Writing network components to an output file
--------------------------------------------
Visualisation of large networks with cytoscape may become challenging. It is possible to extract
individual components/clusters for visualisation as follows::

  python scripts/extract_components.py strain_db_graph.gpickle strain_db

Calculating Rand indices
------------------------
This script allows the clusters formed by different runs/fits/modes of PopPUNK to be compared to each
other. 0 indicates the clusterings are totally discordant, and 1 indicates they are identical.

Run::

  python scripts/calculate_rand_indices.py --input poppunk_gmm_clusters.csv,poppunk_dbscan_cluster.csv

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

  python scripts/calculate_silhouette.py --distances strain_db.dists --cluster-csv strain_db_clusters.csv

The following additonal options are available for use with external clusterings (e.g. from hierBAPS):

- ``--cluster-col`` the (1-indexed) column index containing the cluster assignment
- ``--id-col`` the (1-indexed) column index containing the sample names
- ``--sub`` a string to remove from sample names to match them to those in ``--distances``
