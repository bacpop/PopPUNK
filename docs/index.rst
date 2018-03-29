.. PopPUNK documentation master file, created by
   sphinx-quickstart on Wed Mar 28 17:33:40 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PopPUNK documentation
==================================
``PopPUNK`` uses the fast k-mer distance estimation enabled by `mash <https://mash.readthedocs.io/en/latest/>`_
to calculate core and accessory distances between all pairs of isolates of bacteria in a collection. By clustering
these distances into 'within-strain' and 'between-strain' distances a network
of within-strain comparisons can be constructed. The use of a network has
a number of convenient properties, the first being that the connected
components represent a cluster of strains.

.. image:: poppunk_v2.png
   :alt:  PopPUNK (Population Partitioning Using Nucleotide K-mers)
   :align: center

As well as identifying strains, the pairwise distance distribution also helps
with assembly quality control (particularly in the case of contaminated
contigs) and may be informative of the level of recombination in the
population. The network representation also allows definition of representative isolates by
sampling one example from each clique, and calculation of various statistics
which can show how good the clustering is.

The advantages of this approach are broadly that:

- It is fast, and scalable to :math:`10^{4}` genomes in a single run.
- Assigning new query sequences to a cluster using an existing database is scalable even beyond this.
- Databases can be updated online (as sequences arrive).
- Online updating is equivalent to building databases from scratch.
- Databases can be kept small and managable by only keeping representative isolates.
- There is no bin cluster. Outlier isolates will be in their own cluster.
- Pre-processing, such as generation of an alignment, is not required.
- The definition of clusters is biologically relevant to how bacteria evolve.
- There is a lot of quantitative and graphical output to assist with
  clustering.
- A direct import into  `microreact <https://microreact.org/>`_ is
  available
- Everything is available within a single python executable.

.. note:: PopPUNK is only applicable where there are 'strains'
   This is when core and accessory distances are broadly correlated with each
   other in separate groups. This is often the case across a species, but would
   not be true within a lineage

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   self
   installation.rst
   options.rst
   tutorial.rst
   troubleshooting.rst
   api.rst
   miscellaneous.rst

Index:
------

* :ref:`genindex`
* :ref:`search`
