Sketching
=========

The basis of all analysis is estimation of core and accessory genome distances between samples.
PopPUNK uses genome sketching to make analysis more efficient. In previous versions we used
mash, however the current version now requires `pp-sketchlib <https://github.com/johnlees/pp-sketchlib>`__.

This page details options related to sketching and distance calculation, and is relevant
to both :ref:`query_assignment` and :ref:`model_fitting`.

Overview
--------
Any input given to ``--r-files`` or ``--q-files`` will be sketched using the following
steps:

1. Run pp-sketchlib to sketch all input genomes.
2. (r-files only) Run :doc:`qc` on the sketches. Remove, ignore or stop, depending on ``--qc-filter``.
3. (r-files only) Calculate random match chances and add to the database.
4. Save sketches in a HDF5 datbase (the .h5 file).
5. (r-files only) Calculate core and accessory distances between every pair of sketches, save in .npy and .pkl.
6. (q-files only) Calculate core and accessory distances between query and reference sketches.

To run this before :ref:`model_fitting`, use ``--create-db``::

   poppunk --create-db --output database --r-files rlist.txt --threads 8

The rest of this page describes options to further control this process.

Using pp-sketchlib directly
^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can use pp-sketchlib directly to create sketches, though functionality is identical
to doing this through PopPUNK. You will need to run both sketch and query modes to generate
the sketch database and the distance files as in ``--create-db``::

   poppunk_sketch --sketch --rfile rfiles.txt --ref-db database --sketch-size 10000 --min-k 15 --k-step 2 --cpus 4
   poppunk_sketch --query --ref-db database --query-db database --cpus 4

You may want to do this if you anticipate trying different k-mer sizes, are using the
databases for other purposes, or running a very large analysis where it is useful to split
up the sketching and distance steps. Useful options include:

- ``--print`` -- to print distances in human-readable format to the terminal.
- ``--jaccard`` -- will output Jaccard distances at each k-mer length, rather than core and accessory distances.
- ``--subset`` -- to only calculate distances for a subset of the genomes in the reference database.

.. note::
   Some options have slightly different names. See the pp-sketchlib README for full details.

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

GPU acceleration
----------------


Sketching from read data
------------------------


Sketching RNA viruses
---------------------