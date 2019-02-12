Troubleshooting
===============

This page deals with common issues in running the analysis. For issues with
installing or running the software please raise an issue on github.

.. contents::
   :local:

Error/warning messages
----------------------

Errors in graph.py
^^^^^^^^^^^^^^^^^^
If you get an ``AttributeError``::

    AttributeError: 'Graph' object has no attribute 'node'

Then your ``networkx`` package is out of date. Its version needs to be at >=v2.0.

Trying to create a very large network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When using ``--refine-model`` you may see the message::

    Warning: trying to create very large network

One or more times. This is triggered if :math:`5 \times 10^5` edges or greater than 40%
of the maximum possible number of edges have been added into the network. This suggests that
the boundary is too large including too many links as within sample. This isn't necessarily a
problem as it can occur at the edge of the optimisation range, so will not be the final optimised
result. However, if you have a large number of samples it may make this step run very slowly
and/or use a lot of memory. If that is the case, decrease ``--pos-shift``.

Row name mismatch
^^^^^^^^^^^^^^^^^
PopPUNK may throw::

    RuntimeError: Row name mismatch. Old: 6999_2#17.fa,6259_5#6.fa
    New: 6952_7#16.fa,6259_5#6.fa

This is an error where the mash output order does not match the order in stored
databases (``.pkl``). Most likely, the input files are from different runs, possibly
due to using ``--overwrite``. Run again, giving each step its own output directory.

Samples are missing from the final network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When running ``--assign-query`` an error such as::

    WARNING: Samples 7553_5#54.fa,6999_5#1.fa are missing from the final network

Means that samples present in ``--distances`` and or ``--ref-db`` are not present
in the loaded network. This should be considered an error as it will likely lead to other
errors and warnings. Make sure the provided network is the one created by applying
the ``--model-dir`` to ``--distances``, and that the same output directory has
not been used and overwriten by different steps or inputs.

Old cluster split across multiple new clusters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When running ``--assign-query``, after distances have been calculated and queries are being
assigned warnings such as::

    WARNING: Old cluster 1 split across multiple new clusters

Mean that a single cluster in the original clustering is now split into more than one
cluster. This means something has gone wrong, as the addition of new queries should only
be able to merge existing clusters, not cause them to split.

Most likely, the ``--previous-clustering`` directory is inconsistent with the ``--ref-db``
and/or ``--model-dir``. Make sure the clusters are those created from the network being
used to assign new queries.

If you want to change cluster names or assign queries to your own cluster definitions
you can use the ``--external-clustering`` argument instead.

.. _kmer-length:

Choosing the right k-mer lengths
--------------------------------
When using in the ``--create-db`` mode a straight line fit is required. Make
sure to run with the ``--plot-fit`` option, which will randomly choose a number
of sample pairs to plot the relation between k-mer distances and core and
accessory fits.

To get a sensitive estimate of accessory distance independent from core
distance, a small a k-mer size as possible needs to be included in the fit.
However, for longer genomes too small a k-mer size will result in biased
estimates of distances as small k-mers will match at random.

Here is an example of a fit with ``--k-step 2 --min-k 13``:

.. image:: fit_example_wrong.png
   :alt:  A bad fit to k-mer distances
   :align: center

The genome being fitted is 4.6Mb long, which at 13-mer matches gives a 6%
chance of random matches (this information is written to ``STDERR``), resulting
in the left-most point being over-estimated. Using exactly the same command,
but changing ``--min-k 15`` fixes the issue:

.. image:: fit_example_fixed.png
   :alt:  A fixed fit to k-mer distances
   :align: center

A ``--kmer-step`` of four is usually sufficient, but drop this to two or three
to give the best accuracy.

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

.. _cytoscape-view:

Viewing the network with cytoscape
----------------------------------
If you add the ``--cytoscape`` option when running ``--fit-model`` _cytoscape.graphml
and _cytoscape.csv files will be written to the output directory.

Open `cytoscape <http://www.cytoscape.org/>`_ and drag and drop the .graphml
file onto the window to import the network. Import -> table -> file to load the
CSV. Click 'Select None' then add the 'id' column as a key, and any required
metadata columns (at least the 'Cluster' column) as attributes. Make sure
'Node Table Columns' is selected as the data type.

Click on 'Style' and change the node fill colour to be by cluster, the mapping
type as discrete, then right click to autogenerate a colour scheme. You can
also modify the node size here. In the :doc:`tutorial` example, the components
are nicely separated and the network has high transitivity:

.. image:: cytoscape.png
   :alt:  Cytoscape plot of network
   :align: center

In some cases, edges which are between strain links may have been erroneously included
in the network. This could be due to poor model fit, or a poor quality
sequence. Use Tools -> NetworkAnalyzer -> Analyze Network to compute
information for each node and edge. It may help to analyze connected components separately.
They can be split under Tools -> NetworkAnalyzer -> Subnetwork Creation.

Here is an example where an errant node is connecting two clusters into one
large cluster, which should be split:

.. image:: cytoscape_component.png
   :alt:  Cytoscape plot of network
   :align: center

The incorrect node in question has a low CluteringCoefficient and high Stress.
The EdgeBetweeness of its connections are also high. Sorting the node and edge
tables by these columns can find individual problems such as this.

.. _perplexity:

Setting the perplexity parameter for t-SNE
------------------------------------------
In t-SNE an embedding of the accessory genome distances is found which
represents local structure of the data. Isolates with similar accessory content
will visually appear in clusters together.

The perplexity sets a guess about the number of close neighbours each point
has, and is a trade-off between local and global structure. t-SNE is reasonably
robust to changes in the perplexity parameter (set with ``--perplexity`` when
creating microreact output with ``--microreact`` in the``--fit-model`` mode),
however we would recommend trying a few values to get
a good embedding for the accessory distances.

There is a good discussion of the effect of perplexity `here <https://distill.pub/2016/misread-tsne/>`_
and the sklearn documentation shows some examples of the effect of `changing
perplexity <http://scikit-learn.org/stable/auto_examples/manifold/plot_t_sne_perplexity.html>`_.

In the :doc:`tutorial` example, a perplexity of 30 gives clear clustering of
the accessory genome content, condordant with the core genome structure (`data <https://microreact.org/project/Skg0j9sjz>`__):

.. image:: microreact.png
   :alt:  Microreact plot of results with perplexity = 30
   :align: center

With a lower perplexity of 5, the clustering is too loose, and the strain
structure cannot clearly be seen (`data <https://microreact.org/project/S1RwpK9if>`__):

.. image:: microreact_perplexity5.png
   :alt:  Microreact plot of results with perplexity = 5
   :align: center

30 is a good default, but you may wish to try other values, particularly with
larger or smaller datasets. You can re-run the t-SNE using the ``poppunk_tsne``
command, providing the distances from the previous run::

   poppunk_tsne --distances strain_db/strain_db.dists --output strain_db \
   --perplexity 20 --verbosity 1

.. _qc:

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

.. image:: contam_DPGMM_fit.png
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

In this case it is sufficent to increase the number of mixture components to four,
which no longer includes these inflated distances. This gives a score of 0.9401 and 28 components:

.. image:: contam_DPGMM_better_fit.png
   :alt:  A better fit to pairwise distances
   :align: center

The best thing to do is to remove the poor quality isolate, or if possible
remove the contaminated reads/contigs from the assembly.

With the network
^^^^^^^^^^^^^^^^
Alternatively, the network itself can be inspected with ``--cytoscape``. Using
the approach detailed in :ref:`cytoscape-view` gives the following view:

.. image:: cytoscape_contaminant.png
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

Memory/run-time issues
----------------------
For larger datasets resource use may be challenging. So far the largest dataset
we've analysed was around 12000 genomes, which used modest computational
resources. Here are some tips based on these experiences:

- Add ``--threads`` -- they are used fairly efficiently throughout.
- When running ``--create-db`` with many threads, add the ``--no-stream`` option.
  This will trade-off memory for disk usage, as it seems that many threaded
  ``mash dist`` output cannot be processed as fast as it is produced.
- In ``--refine-model`` set ``--pos-shift 0`` to avoid creating huge networks
  with close to :math:`N^2` edges. Mixture models normally need to be pruned.
- In ``--refine-model`` you may add the ``--no-local`` option to skip that step
  and decrease run-time, though gains are likely marginal.
- Use ``--rapid-nj``, if producing MicroReact output.

Another option for scaling is to run ``--create-db`` with a smaller initial set (not
using the ``--full-db`` command), then use ``--assign-query`` to add to this.

