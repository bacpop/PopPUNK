Creating visualisations
=======================
We have moved visualisation tools into their own program ``poppunk_visualise``, both
to reinforce our commitment to UK spellings, and so that you can rerun visualisations
with different outputs and settings without rerunning the other parts of the code.

Starting with either a full database where you have fitted a model (:doc:`model_fitting`), or
output where you have assigned queries using an existing database (:doc:`query_assignment`), you
can create outputs for external interactive visualisation tools:

- `Microreact <https://microreact.org/>`__ -- a genomic epidemiology visualisation tool, displaying clusters, phylogeny and accessory clustering.
- `GrapeTree <https://achtman-lab.github.io/GrapeTree/MSTree_holder.html>`__ -- a tool to visualise strains (designed for cgMLST).
- `Cytoscape <https://cytoscape.org/>`__ -- a network visualisation tool, which can be used to create, view and manipulate a layout of the graph.
- `Phandango <https://jameshadfield.github.io/phandango/#/>`__ -- visualisation linking genomes and phylogenies.

.. important::
   If you run a visualisation on output from query assignment (:doc:`query_assignment`)
   this will not contain all the necessary distances, and they will be calculated before
   the visualisation files are produced.
   You will see a message ``Note: Distance will be extended to full all-vs-all distances``.
   If you are running multiple visualisations this calculation will be completed every time. To avoid
   this re-run your assignment with ``--update-db``, which will add these distances in permanently.

**Contents**:

.. contents::
   :local:

Common options
--------------
Some typical commands for various input settings (with ``--microreact``, but this can
be altered to any output type) with a database ``example``.

Visualisation of a full database::

   poppunk_visualise --ref-db example_db --output example_viz --microreact

Visualisation after query assignment::

   poppunk_visualise --ref-db example_db --query-db example_query --output example_viz --microreact

Visualisation when sketches and models are in different folders::

   poppunk_visualise --ref-db example_db --previous-clustering example_lineages \
    --model-dir example_lineages --output example_viz --microreact

Visualisation with a lineage model, which has been queried (query-query distances must be provided)::

   poppunk_visualise --distances example_query/example_query.dists --ref-db example_db \
    --model-dir example_lineages --query-db example_lineage_query \
    --output example_viz --microreact

Notable modifiers include:

- ``--include-files`` -- give a file with a subset of names to be included in the visualisation.
- ``--external-clustering`` -- other cluster names to map to strains (such as MLST, serotype etc),
  as described in model fitting and query assignment.
- ``--info-csv`` -- similar to the above, but a CSV which is simply (inner-)joined to the output on sample name.
- ``--rapidnj`` -- the location of a `rapidnj <https://birc.au.dk/software/rapidnj/>`__ binary,
  used to build the core NJ tree. We highly recommend using this for any tree-building (and is included with
  the conda install). This defaults to ``rapidnj`` on the PATH. Set blank to use dendropy instead (slower, especially
  for large datasets).
- ``--core-only``/``--accessory-only`` -- use the core or accessory fit from an individually refined model (see :ref:`indiv-refine`).
- ``--threads``, ``--gpu-dist``, ``--deviceid``, ``--strand-preserved`` -- querying options used if extra distance calculations are needed.
  To avoid these, rerun your query with ``--update-db``.

Microreact
----------

.. _perplexity:

Setting the perplexity parameter for t-SNE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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

In the :doc:`model_fitting` example, a perplexity of 30 gives clear clustering of
the accessory genome content, condordant with the core genome structure (`data <https://microreact.org/project/Skg0j9sjz>`__):

.. image:: images/microreact.png
   :alt:  Microreact plot of results with perplexity = 30
   :align: center

With a lower perplexity of 5, the clustering is too loose, and the strain
structure cannot clearly be seen (`data <https://microreact.org/project/S1RwpK9if>`__):

.. image:: images/microreact_perplexity5.png
   :alt:  Microreact plot of results with perplexity = 5
   :align: center

30 is a good default, but you may wish to try other values, particularly with
larger or smaller datasets. You can re-run the t-SNE using the ``poppunk_tsne``
command, providing the distances from the previous run::

   poppunk_tsne --distances strain_db/strain_db.dists --output strain_db \
   --perplexity 20 --verbosity 1

GrapeTree
---------

.. _cytoscape-view:

Cytoscape
---------
If you add the ``--cytoscape`` option when running ``--fit-model`` _cytoscape.graphml
and _cytoscape.csv files will be written to the output directory.

Open `cytoscape <http://www.cytoscape.org/>`_ and drag and drop the .graphml
file onto the window to import the network. Import -> table -> file to load the
CSV. Click 'Select None' then add the 'id' column as a key, and any required
metadata columns (at least the 'Cluster' column) as attributes. Make sure
'Node Table Columns' is selected as the data type.

Click on 'Style' and change the node fill colour to be by cluster, the mapping
type as discrete, then right click to autogenerate a colour scheme. You can
also modify the node size here. In the :doc:`model_fitting` example, the components
are nicely separated and the network has high transitivity:

.. image:: images/cytoscape.png
   :alt:  Cytoscape plot of network
   :align: center

In some cases, edges which are between strain links may have been erroneously included
in the network. This could be due to poor model fit, or a poor quality
sequence. Use Tools -> NetworkAnalyzer -> Analyze Network to compute
information for each node and edge. It may help to analyze connected components separately.
They can be split under Tools -> NetworkAnalyzer -> Subnetwork Creation.

Here is an example where an errant node is connecting two clusters into one
large cluster, which should be split:

.. image:: images/cytoscape_component.png
   :alt:  Cytoscape plot of network
   :align: center

The incorrect node in question has a low CluteringCoefficient and high Stress.
The EdgeBetweeness of its connections are also high. Sorting the node and edge
tables by these columns can find individual problems such as this.

Phandango
---------