.. PopPUNK documentation master file, created by
   sphinx-quickstart on Wed Mar 28 17:33:40 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PopPUNK documentation
==================================
.. image:: poppunk_v2.png
   :alt:  PopPUNK (Population Partitioning Using Nucleotide K-mers)
   :align: center

In straightforward cases, usage can be as simple as::

   poppunk --easy-run --r-files references.txt --output poppunk_db

Where ``references.txt`` is a list of assembly fasta files, one per line. See
:doc:`quickstart` and the :doc:`tutorial` for full details.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   self
   installation.rst
   options.rst
   quickstart.rst
   tutorial.rst
   troubleshooting.rst
   scripts.rst
   miscellaneous.rst

Details
-------
A full description of the method can be found in the `paper <https://doi.org/10.1101/gr.241455.118>`_.

``PopPUNK`` uses the fast k-mer distance estimation enabled by `mash <https://mash.readthedocs.io/en/latest/>`_
to calculate core and accessory distances between all pairs of isolates of bacteria in a collection. By clustering
these distances into 'within-strain' and 'between-strain' distances a network
of within-strain comparisons can be constructed. The use of a network has
a number of convenient properties, the first being that the connected
components represent a cluster of strains.

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
  available, as well as `cytoscape <http://www.cytoscape.org/>`_,
  `grapetree <http://dx.doi.org/10.1101/gr.232397.117>`_ and
  `phandango <http://jameshadfield.github.io/phandango/>`_.
- Everything is available within a single python executable.

Citation:
---------
If you find PopPUNK useful, please cite as:

Lees JA, Harris SR, Tonkin-Hill G, Gladstone RA, Lo SW, Weiser JN, Corander J, Bentley SD, Croucher NJ. Fast and flexible
bacterial genomic epidemiology with PopPUNK. *Genome Research* **29**:1-13 (2019).
doi:`10.1101/gr.241455.118 <https://dx.doi.org/10.1101/gr.241455.118>`__

Index:
------

* :ref:`genindex`
* :ref:`search`
