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

from .bgmm import findWithinLabel

def fitDbScan(X, outPrefix, max_num_clusters, min_cluster_prop, threads = 1):
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
    min_samples = max(int(min_cluster_prop * X.shape[0]), 10)
    min_cluster_size = max(int(0.01 * X.shape[0]), 10)
    indistinct_clustering = True
    while indistinct_clustering and min_cluster_size >= min_samples:
        # Fit DBSCAN model
        hdb = hdbscan.HDBSCAN(algorithm='boruvka_balltree',
                         min_samples = min_samples,
                         core_dist_n_jobs = threads,
                         memory = cache_out,
                         prediction_data = True,
                         min_cluster_size = min_cluster_size
                         ).fit(X)
        # Number of clusters in labels, ignoring noise if present.
        labels = hdb.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        # Test whether model fit contains distinct clusters
        if n_clusters > 1 and n_clusters <= max_num_clusters:
            indistinct_clustering = evaluate_dbscan_clusters(X, hdb, n_clusters)
        # Alter minimum cluster size criterion
        min_cluster_size = int(min_cluster_size / 2)

    # Report failure where it happens
    if indistinct_clustering:
        print("Failed to find distinct clusters in this dataset", file = sys.stderr)
        exit(1)

    # tidy up cache
    rm_rf(cache_out)

    # return model parameters
    return hdb, labels, n_clusters

def rm_rf(d):
    """Function to recursively remove directory
        
    Removes the HDBSCAN directory and all subdirectories
    
    Args:
        d (string)
            Path of top-level directory to remove
        
    """
    for path in (os.path.join(d,f) for f in os.listdir(d)):
        if os.path.isdir(path):
            rm_rf(path)
        else:
            os.unlink(path)
    os.rmdir(d)

def evaluate_dbscan_clusters(X, hdb, n_clusters):
    """Evaluate whether fitted dbscan model contains non-overlapping clusters

    Args:
        X (numpy.array)
            N x 2 array of core and accessory distances
        hdb (hdbscan.HDBSCAN)
            Fitted DBSCAN from hdbscan package
        scale (numpy.array)
            Scale factor of model object
        n_clusters (integer)
            Number of clusters in the fitted model

    Returns:
        indistinct (bool)
            Boolean indicating whether putative within- and
            between-strain clusters of points overlap
    """
    indistinct = True

    # assign points to clusters
    y, strengths = hdbscan.approximate_predict(hdb, X)
    max_cluster_num = np.max(y)

    # calculate minima and maxima of clusters
    cluster_means = np.full((max_cluster_num + 1, 2), 0.0, dtype=float)
    cluster_mins = np.full((max_cluster_num + 1, 2), 0.0, dtype=float)
    cluster_maxs = np.full((max_cluster_num + 1, 2), 0.0, dtype=float)
    
    for i in range(max_cluster_num + 1):
        cluster_means[i,] = [np.mean(X[y==i,0]),np.mean(X[y==i,1])]
        cluster_mins[i,] = [np.min(X[y==i,0]),np.min(X[y==i,1])]
        cluster_maxs[i,] = [np.max(X[y==i,0]),np.max(X[y==i,1])]

    # identify within-strain and between-strain links
    within_cluster = findWithinLabel(cluster_means, y)
    between_cluster = findBetweenLabel(y, within_cluster)

    # calculate ranges of minima and maxima
    core_minimum_of_between = cluster_mins[between_cluster,0]
    core_maximum_of_within = cluster_maxs[within_cluster,0]
    accessory_minimum_of_between = cluster_mins[between_cluster,1]
    accessory_maximum_of_within = cluster_maxs[within_cluster,1]

    # evaluate whether maxima of cluster nearest origin do not
    # overlap with minima of cluster furthest from origin
    if core_minimum_of_between > core_maximum_of_within and \
        accessory_minimum_of_between > accessory_maximum_of_within:
        indistinct = False

    # return distinctiveness
    return indistinct

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
