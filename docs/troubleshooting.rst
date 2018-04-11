Troubleshooting
===============

This page deals with common issues in running the analysis. For issues with
installing or running the software please raise an issue on github.

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

Dealing with contamination
---------------------------
Isolate QC

.. _bgmm-fit:

Fine tuning model fit
---------------------
BGMM

Use of cytoscape


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
larger or smaller datasets.

