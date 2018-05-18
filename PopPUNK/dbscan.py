'''DBSCAN using hdbscan'''

# universal
import os
import sys
import re
# additional
import numpy as np
import random
import operator
import pickle
# hdbscan
import hdbscan


def fitDbScan(X, outPrefix, threads = 1):
    """Function to fit DBSCAN model as an alternative to the Gaussian

    Fits the DBSCAN model to the distances using hdbscan

    Args:
        X (np.array)
            n x 2 array of core and accessory distances for n samples
        outPrefix (str)
            Prefix for output files to be saved under
        threads (int)
            Number of threads to use in parallelisation of dbscan model fitting

    Returns:
        hdb (hdbscan.HDBSCAN)
            Fitted HDBSCAN to subsampled data
        labels (list)
            Cluster assignments of each sample
        n_clusters (int)
            Number of clusters used
    """
    # set DBSCAN clustering parameters
    cache_out = "./" + outPrefix + "_cache"
    min_samples = max(int(0.0001 * X.shape[0]), 10)
    min_cluster_size = max(int(0.01 * X.shape[0]), 10)
    hdb = hdbscan.HDBSCAN(algorithm='boruvka_balltree',
                         min_samples = min_samples,
                         core_dist_n_jobs = threads,
                         memory = cache_out,
                         prediction_data = True,
                         min_cluster_size = min_cluster_size
                         ).fit(X)
    labels = hdb.labels_
    # not used
    #core_samples_mask = np.zeros_like(hdb.labels_, dtype=bool)

    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    return hdb, labels, n_clusters


def assign_samples_dbscan(X, hdb, scale):
    """Use a fitted dbscan model to assign new samples to a cluster

    Args:
        X (numpy.array)
            N x 2 array of core and accessory distances
        hdb (hdbscan.HDBSCAN)
            Fitted DBSCAN from hdbscan package
        scale (numpy.array)
            Scale factor of model object

    Returns:
        y (numpy.array)
            Cluster assignments by sample
    """
    y, strengths = hdbscan.approximate_predict(hdb, X/scale)
    return y


def findBetweenLabel(means, assignments, within_cluster):
    """Identify between-strain links from a DBSCAN model

    Finds the component containing the largest number of between-strain
    links, excluding the cluster identified as containing within-strain
    links.

    Args:
        means (numpy.array)
            K x 2 array of component means from :class:`~PopPUNK.model.BGMMFit` or
            :class:`~PopPUNK.model.DBSCANFit`
        assignments (numpy.array)
            Sample cluster assignments
        within_cluster (int)
            Cluster ID assigned to within-strain assignments, from :func:`~PopPUNK.bgmm.findWithinLabel`

    Returns:
        between_cluster (int)
            The cluster label for the between-strain assignments
    """
    # remove noise and within-strain distance cluster
    assignments = list(filter((within_cluster).__ne__, assignments)) # remove within-cluster
    assignments = list(filter((-1).__ne__, assignments)) # remove noise

    # identify non-within cluster with most members
    between_cluster = max(set(assignments), key=assignments.count)

    return between_cluster


