# vim: set fileencoding=<utf-8> :
# Copyright 2018-2023 John Lees and Nick Croucher

'''DBSCAN using hdbscan'''

# universal
import os
import sys
# hdbscan
import hdbscan

from .utils import check_and_set_gpu

def fitDbScan(X, min_samples, min_cluster_size, cache_out, use_gpu = False):
    """Function to fit DBSCAN model as an alternative to the Gaussian

    Fits the DBSCAN model to the distances using hdbscan

    Args:
        X (np.array)
            n x 2 array of core and accessory distances for n samples
        min_samples (int)
            Parameter for DBSCAN clustering 'conservativeness'
        min_cluster_size (int)
            Minimum number of points in a cluster for HDBSCAN
        cache_out (str)
            Prefix for DBSCAN cache used for refitting
        use_gpu (bool)
            Whether GPU algorithms should be used in DBSCAN fitting

    Returns:
        hdb (hdbscan.HDBSCAN or cuml.cluster.HDBSCAN)
            Fitted HDBSCAN to subsampled data
        labels (list)
            Cluster assignments of each sample
        n_clusters (int)
            Number of clusters used
    """
    # Check on initialisation of GPU libraries and memory
    try:
        import cudf
        from cuml import cluster
        import cupy as cp
        gpu_lib = True
    except ImportError as e:
        gpu_lib = False
    # check on GPU
    use_gpu = check_and_set_gpu(use_gpu,
                                gpu_lib,
                                quit_on_fail = True)
    # set DBSCAN clustering parameters
    if use_gpu:
      sys.stderr.write('Fitting HDBSCAN model using a GPU\n')
      hdb = cluster.hdbscan.HDBSCAN(min_samples = min_samples,
                                 output_type = 'cupy',
                                 prediction_data = True,
                                 min_cluster_size = min_cluster_size
                                 ).fit(X)
      # Number of clusters in labels, ignoring noise if present.
      labels = hdb.labels_
      n_clusters = len(cp.unique(labels[labels>-1]))
    else:
      sys.stderr.write('Fitting HDBSCAN model using a CPU\n')
      hdb = hdbscan.HDBSCAN(algorithm='boruvka_balltree',
                       min_samples = min_samples,
                       #core_dist_n_jobs = threads, # may cause error, see #19
                       memory = cache_out,
                       prediction_data = True,
                       min_cluster_size = min_cluster_size
                       ).fit(X)
      # Number of clusters in labels, ignoring noise if present.
      labels = hdb.labels_
      n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # return model parameters
    return hdb, labels, n_clusters


def evaluate_dbscan_clusters(model):
    """Evaluate whether fitted dbscan model contains non-overlapping clusters

    Args:
        model (DBSCANFit)
            Fitted model from :func:`~PopPUNK.models.DBSCANFit.fit`

    Returns:
        indistinct (bool)
            Boolean indicating whether putative within- and
            between-strain clusters of points overlap
    """
    indistinct = True

    # calculate ranges of minima and maxima
    core_minimum_of_between = model.cluster_mins[model.between_label,0]
    core_maximum_of_within = model.cluster_maxs[model.within_label,0]
    accessory_minimum_of_between = model.cluster_mins[model.between_label,1]
    accessory_maximum_of_within = model.cluster_maxs[model.within_label,1]

    # evaluate whether maxima of cluster nearest origin do not
    # overlap with minima of cluster furthest from origin
    if core_minimum_of_between > core_maximum_of_within or \
        accessory_minimum_of_between > accessory_maximum_of_within:
        indistinct = False

    # return distinctiveness
    return indistinct

def findBetweenLabel(assignments, within_cluster):
    """Identify between-strain links from a DBSCAN model

    Finds the component containing the largest number of between-strain
    links, excluding the cluster identified as containing within-strain
    links.

    Args:
        assignments (numpy.array)
            Sample cluster assignments
        within_cluster (int)
            Cluster ID assigned to within-strain assignments, from :func:`~PopPUNK.bgmm.findWithinLabel`

    Returns:
        between_cluster (int)
            The cluster label for the between-strain assignments
    """
    # remove noise and within-strain distance cluster
    assignment_list = assignments.tolist()
    assignment_list = list(filter((within_cluster).__ne__, assignment_list)) # remove within-cluster
    assignment_list = list(filter((-1).__ne__, assignment_list)) # remove noise

    # identify non-within cluster with most members
    between_cluster = max(set(assignment_list), key=assignment_list.count)

    return between_cluster
