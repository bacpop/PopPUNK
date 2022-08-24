Troubleshooting
===============

This page deals with common issues in running the analysis. For issues with
installing or running the software please raise an issue on github.

Most/all of my samples merge when I run a query
-----------------------------------------------
If you see a gigantic merge when running ``poppunk_assign``, for example::

    Clusters 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 have merged into 1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16

This might be caused by:

* A single/a few 'bad' genome(s) with artificial zero/close-to-zero distances to many other samples, which links much of the network together.
* A boundary which is actually a bit lax, and when more isolates are added,
  they have distances just below the boundary between a few clusters,
  and there are enough of these to link more and more clusters together.

The first issue should be easy to fix with one of the following methods:

* Use ``--run-qc`` (see :doc:`qc`) with one or more of:

    * ``--max-pi-dist`` and/or ``--max-a-dist`` to remove outlier distances.
    * ``--max-zero-dist`` which sets the maximum proportion of zeros to other samples. Poor quality genomes often end up with zero distances to many samples, linking them together.
    * ``--max-merge`` which sets the maximum number of clusters a single query can cause to merge.
    * ``--betweenness`` which will print all queries in order which have high stress in the network, and are likely causing merges.
* Use ``--serial`` to run samples through one-by-one. This is a little less
  efficient than fully batched querying, but much faster than running independent
  jobs. Note, lineage models and updating the database are not supported with
  this mode.

The second issue above is potentially more insidiuous, and may require a refit to all the
data to obtain a tighter boundary. You can (mostly) keep old cluster IDs via
the use of ``--external-clustering`` if you do this.
Alternatively, you can add the ``--serial`` command to type samples one at a time
as above.

See `issue 194 <https://github.com/bacpop/PopPUNK/issues/194>`__ for more discussion.

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

Known bugs
----------
Calculating query-query distances when unnecessary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A bug in v2.4.0 only, (fixed in v2.5.0 and not in previous versions).

You will always see ``Found novel query clusters. Calculating distances between them.``
when running ``poppunk_assign`` with more than one input sample. This should only
happen when unassigned isolates/novel clusters are found. Our check on this condition
became invalid.

Additionally, this *may* have affected typing when query-query links were present,
this appeared as invalid merges in some tests. If you used this, you may wish to
re-run with v2.5.0 or higher.

Older HDBSCAN models fail to load
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tl;dr if you see an error ``ModuleNotFoundError: No module named 'sklearn.neighbors._dist_metrics'``
you probably need to downgrade ``sklearn`` to v0.24.

The change in scikit-learn's API in v1.0.0 and above mean that HDBSCAN models
fitted with ```sklearn <=v0.24``` will give an error when loaded. If you run into this,
the solution is one of:
- Downgrade sklearn to v0.24.
- Run model refinement to turn your model into a boundary model instead (this will
change clusters).
- Refit your model in an environment with ```sklearn >=v1.0``.

If this is a common problem let us know, as we could write a script to 'upgrade'
HDBSCAN models.
See issue [#213](https://github.com/bacpop/PopPUNK/issues/213) for more details.

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


