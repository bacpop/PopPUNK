Troubleshooting
===============

This page deals with common issues in running the analysis. For issues with
installing or running the software please raise an issue on github.

.. contents::
   :local:

Known bugs
----------

When I look at my clusters on a tree, they make no sense
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This is a bug caused by alphabetic sorting of labels in ``PopPUNK >=v2.0.0``
with ``pp-sketchlib <v1.5.1``. There are three ways to fix this:

- Upgrade to ``PopPUNK >=v2.2`` and ``pp-sketchlib >=v1.5.1`` (preferred).
- Run ``scripts/poppunk_pickle_fix.py`` on your ``.dists.pkl`` file and re-run
  model fits.
- Create the database with ``poppunk_sketch --sketch`` and
  ``poppunk_sketch --query`` directly, rather than ``PopPUNK --create-db``.

I have updated PopPUNK, and my clusters still seemed scrambled
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This is possible using query assignment with ``--update-db``, or in some cases
with ``--gpu-dists``. Please update to ``PopPUNK >=v2.4.0``
with ``pp-sketchlib >=v1.7.0``

Calculating distances using 0 thread(s)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This will lead to an error later on in execution. This is due to a version
mismatch between PopPUNK and ``pp-sketchlib``. Installation of both packages
via conda should keep the versions compatible, but there are ways they can get out of sync.

The solution is as above: upgrade to ``PopPUNK >=v2.2`` and ``pp-sketchlib >=v1.5.1``.

Error/warning messages
----------------------

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

Memory/run-time issues
----------------------
Here are some tips based on experiences analysing larger datasets:

- Add ``--threads`` -- they are used fairly efficiently throughout.
- Consider the ``--gpu-sketch`` and ``--gpu-dists`` options is applicable,
  and a GPU is available.
- In ``--refine-model`` set ``--pos-shift 0`` to avoid creating huge networks
  with close to :math:`N^2` edges. Mixture models normally need to be pruned.
- In ``--refine-model`` you may add the ``--no-local`` option to skip that step
  and decrease run-time, though gains are likely marginal.
- Use ``--rapid-nj``, if producing microreact output.

Another option for scaling is to run ``--create-db`` with a smaller initial set (not
using the ``--full-db`` command), then use ``--assign-query`` to add to this.
