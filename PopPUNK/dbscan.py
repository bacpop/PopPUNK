'''DBSCAN using hdbscan'''

# universal
import os
import sys
import argparse
import re
# additional
import numpy as np
import random
import operator
import pickle
# hdbscan
import hdbscan

from .plot import plot_dbscan_results

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
    # set output dir
    if not os.path.isdir(outPrefix):
        if not os.path.isfile(outPrefix):
            os.makedirs(outPrefix)
        else:
            sys.stderr.write(outPrefix + " already exists as a file! Use a different --output\n")
            sys.exit(1)

    # set the maximum sampling size
    max_samples = 1000000

    # preprocess scaling
    scale = np.amax(X, axis = 0)
    scaled_X = np.copy(X)
    scaled_X /= scale
    if X.shape[0] > max_samples:
        subsampled_X = utils.shuffle(scaled_X, random_state=random.randint(1,max_samples))[0:max_samples,]
    else:
        subsampled_X = np.copy(scaled_X)

    # set DBSCAN clustering parameters
    cache_out = "./" + outPrefix + "_cache"
    min_samples = int(0.0001*subsampled_X.shape[0])
    min_cluster_size = int(0.01*subsampled_X.shape[0])
    db = hdbscan.HDBSCAN(algorithm='boruvka_balltree',
                         min_samples = min_samples,
                         core_dist_n_jobs = threads,
                         memory = cache_out,
                         prediction_data = True,
                         min_cluster_size = min_cluster_size
                         ).fit(subsampled_X)
    labels = db.labels_
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    sys.stderr.write("Number of clusters: " + str(n_clusters_) + "\n")
    sys.stderr.write("Number of datapoints: " + str(subsampled_X.shape[0]) + "\n")
    sys.stderr.write("Number of assignments: " + str(len(labels)) + "\n")

    # Plot results
    plot_dbscan_results(subsampled_X, labels, n_clusters_, outPrefix + "/" + outPrefix + "_dbscan")

    # get within strain cluster
    max_cluster_num = db.labels_.max()
    cluster_means = np.full((n_clusters_,2),0.0,dtype=float)
    cluster_mins = np.full((n_clusters_,2),0.0,dtype=float)
    cluster_maxs = np.full((n_clusters_,2),0.0,dtype=float)

    for i in range(max_cluster_num+1):
        cluster_means[i,] = [np.mean(subsampled_X[db.labels_==i,0]),np.mean(subsampled_X[db.labels_==i,1])]
        cluster_mins[i,] = [np.min(subsampled_X[db.labels_==i,0]),np.min(subsampled_X[db.labels_==i,1])]
        cluster_maxs[i,] = [np.max(subsampled_X[db.labels_==i,0]),np.max(subsampled_X[db.labels_==i,1])]

    # assign all samples
    y, strengths = hdbscan.approximate_predict(db, scaled_X)

    # Save model fit
    np.savez(outPrefix + "/" + outPrefix + '_dbscan_fit.npz',
             means=cluster_means,
             mins=cluster_mins,
             maxs=cluster_maxs,
             scale=scale)
    pickle_file_name = outPrefix + "/" + outPrefix + '_dbscan_fit.pkl'
    with open(pickle_file_name, 'wb') as pickle_file:
        pickle.dump(db, pickle_file)

    # return output
    return y, db, cluster_means, cluster_mins, cluster_maxs, scale

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

def assign_samples_dbscan(X, db, scale):

    scaled_X = np.copy(X)
    scaled_X /= scale

    y, strengths = hdbscan.approximate_predict(db, scaled_X)

    return y
