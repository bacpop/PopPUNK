# vim: set fileencoding=<utf-8> :
'''DBSCAN using hdbscan'''

# universal
import os
import sys
# hdbscan
import hdbscan

def fitDbScan(X, min_samples, min_cluster_size, cache_out):
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

    Returns:
        hdb (hdbscan.HDBSCAN)
            Fitted HDBSCAN to subsampled data
        labels (list)
            Cluster assignments of each sample
        n_clusters (int)
            Number of clusters used
    """
    # set DBSCAN clustering parameters
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
    y = hdbscan.approximate_predict(hdb, X/scale)[0]
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
