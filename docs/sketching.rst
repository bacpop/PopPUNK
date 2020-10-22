Sketching
=========

pp-sketchlib: how to use for Sketching
including jaccard distances & GPU use

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

.. image:: images/fit_example_wrong.png
   :alt:  A bad fit to k-mer distances
   :align: center

The genome being fitted is 4.6Mb long, which at 13-mer matches gives a 6%
chance of random matches, resulting in the left-most point being over-estimated.
Using exactly the same command, but changing ``--min-k 15`` fixes the issue:

.. image:: images/fit_example_fixed.png
   :alt:  A fixed fit to k-mer distances
   :align: center

A ``--kmer-step`` of four is usually sufficient, but drop this to two or three
to give the best accuracy at the expense of increased execution time.