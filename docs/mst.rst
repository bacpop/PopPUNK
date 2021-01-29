Minimum spanning trees
=======================

**Contents**:

.. contents::
   :local:

Using the distances and a network, you can generate a minimum spanning tree. This
can be useful when a neigbour joining tree is difficult to produce, for example
if the dataset is very large, and in some cases has uses in tracing spread
(take care with this interpretation, direction is not usually obvious).

There are three different ways to make MSTs, depending on how much data you have.
Roughly:

- 'Small': Up to :math:`\sim 10^3` samples.
- 'Medium': Up to :math:`\sim 10^5` samples.
- 'Large': Over :math:`10^5` samples.

In each mode, you can get as output:

- A plot of the MST as a graph layout, optionally coloured by strain.
- A plot of the MST as a graph layout, highlighting edge betweenness and node degree.
- The graph as a graphml file, to view in :doc:`cytoscape`.
- The MST formatted as a newick file, to view in a tree viewer of your choice.

With small data
---------------
For a small dataset it's feasible to simply make an MST from your dense distance matrix

With medium data
----------------

With big data
-------------