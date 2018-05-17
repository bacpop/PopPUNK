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
    """Function to fit DBSCAN model as an alternative to the Gaussian, called from :func:`~PopPUNK.__main__.main()`

    Fits the DBSCAN model to the distances, saves model parameters to a file,
    and assigns the samples to a component. Write fit summary stats to STDERR.

    By default, subsamples :math:`10^6` random distances to fit the model to.

    Args:
        X (np.array)
            n x 2 array of core and accessory distances for n samples
        outPrefix (str)
            Prefix for output files to be saved under
        threads (int)
            Number of threads to use in parallelisation of dbscan model fitting

    Returns:
        y (np.array)
            Cluster assignment for each sample
        db (hdbscan.HDBSCAN)
            Fitted HDBSCAN to subsampled data
        cluster_means (numpy.array)
            Mean positions (x, y) of each cluster
        cluster_mins (numpy.array)
            Minimum values (x, y) assigned to each cluster
        cluster_maxs
            Maximum values (x, y) assigned to each cluster
        scale (numpy.array)
            Scaling of core and accessory distances
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
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    return hdb, labels, n_clusters_


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
            between_cluster (int)
                The cluster label for the between-strain assignments
    """
    y, strengths = hdbscan.approximate_predict(hdb, X/scale)
    return y


def findBetweenLabel(means, assignments, within_cluster):
    """Identify between-strain links

    Finds the component containing the largest number of between-strain
    links, excluding the cluster identified as containing within-strain
    links.

        Args:
            means (numpy.array)
                K x 2 array of mixture component means from :func:`~PopPUNK.bgmm.fit2dMultiGaussian` or
                :func:`~PopPUNK.bgmm.assignQuery` or :func:`~fitDbScan`
            assignments (numpy.array)
                Sample cluster assignments from :func:`~PopPUNK.bgmm.assign_samples` or :func:`~fitDbScan`
            within_cluster (int)
                Cluster assigned to within-strain assignments

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


