Data quality control
====================
TODO: new QC options

Removing samples from a database
--------------------------------
You can use the ``prune_poppunk`` command to remove samples from a database,
for example those found to be of poor quality (see :ref:`qc`). Create a file
``remove.txt`` with the names of the samples you wish to remove, one per line,
and run::

   prune_poppunk --remove remove.txt --distances strain_db/strain_db.dists --output pruned_db

This will remove the samples from the ``strain_db.dists`` files, from which
``--model-fit`` can be run again.

If you would like to create the mash sketches again, which is recommended if
you plan to use ``--full-db`` and/or assign future query sequences, add the
``--resketch`` argument::

   prune_poppunk --remove remove.txt --distances strain_db/strain_db.dists --output pruned_db --resketch --ref-db strain_db --threads 4

Dealing with poor quality data
------------------------------
In this example we analyse 76 *Haemophilus influenzae* isolates. One isolate, 14412_4_15,
is contaminated with 12% of reads being *Haemophilus parainfluenzae* and a total
assembly length of 3.8Mb. It should be removed before input, but its presence
can also be found with ``PopPUNK``.

With the distances
^^^^^^^^^^^^^^^^^^
A fit with three mixture components overestimates the number of between strain
links, and gives a network with a poor score (0.6849) and only five components:

.. image:: images/contam_DPGMM_fit.png
   :alt:  A bad fit to pairwise distances
   :align: center

Distances in the top left of the plot, with low core distances and high
accessory distances, are due to the contaminated contigs in the isolate.
Finding which isolates contribute to these distances reveals a clear culprit::

    awk '$3<0.02 && $4 > 0.3 {print $1}' contam_db/contam_db.search.out | cut -f 1 | sort | uniq -c
       1 14412_3_81
       1 14412_3_82
       1 14412_3_83
       1 14412_3_84
       1 14412_3_88
       1 14412_3_89
       1 14412_3_91
       1 14412_3_92
       1 14412_4_1
       1 14412_4_10
      28 14412_4_15

In this case it is sufficient to increase the number of mixture components to four,
which no longer includes these inflated distances. This gives a score of 0.9401 and 28 components:

.. image:: images/contam_DPGMM_better_fit.png
   :alt:  A better fit to pairwise distances
   :align: center

The best thing to do is to remove the poor quality isolate, or if possible
remove the contaminated reads/contigs from the assembly.

With the network
^^^^^^^^^^^^^^^^
Alternatively, the network itself can be inspected with ``--cytoscape``. Using
the approach detailed in :ref:`cytoscape-view` gives the following view:

.. image:: images/cytoscape_contaminant.png
   :alt:  A better fit to pairwise distances
   :align: center

The contaminated node appears when ordering by ClusteringCoefficient, Stress or
TopologicalCoefficient, and its edges appear when ordering by EdgeBetweeness.
It can be seen highlighted in the top right component, connecting two clusters
which otherwise have no links. It can be removed, and components recalculated in
cytoscape directly, though removal from the PopPUNK database is best.

The second largest cluster is also suspicious, where there are few triangles
(low transitivity) and the nodes involved have high Stress. This is indicative
of a bad fit overall, rather than a single problem sample.

