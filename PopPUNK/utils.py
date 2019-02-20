# vim: set fileencoding=<utf-8> :
'''General utility functions for data read/writing/manipulation in PopPUNK'''

# universal
import os
import sys
# additional
import pickle
from collections import defaultdict
from tempfile import mkstemp
import numpy as np
import pandas as pd
import sharedmem

def storePickle(rlist, qlist, self, X, pklName):
    """Saves core and accessory distances in a .npy file, names in a .pkl

    Called during ``--create-db``

    Args:
        rlist (list)
            List of reference sequence names (for :func:`~iterDistRows`)
        qlist (list)
            List of query sequence names (for :func:`~iterDistRows`)
        self (bool)
            Whether an all-vs-all self DB (for :func:`~iterDistRows`)
        X (numpy.array)
            n x 2 array of core and accessory distances
        pklName (str)
            Prefix for output files
    """
    with open(pklName + ".pkl", 'wb') as pickle_file:
        pickle.dump([rlist, qlist, self], pickle_file)
    np.save(pklName + ".npy", X)


def readPickle(pklName):
    """Loads core and accessory distances saved by :func:`~storePickle`

    Called during ``--fit-model``

    Args:
        pklName (str)
            Prefix for saved files

    Returns:
        rlist (list)
            List of reference sequence names (for :func:`~iterDistRows`)
        qlist (list)
            List of query sequence names (for :func:`~iterDistRows`)
        self (bool)
            Whether an all-vs-all self DB (for :func:`~iterDistRows`)
        X (numpy.array)
            n x 2 array of core and accessory distances
    """
    with open(pklName + ".pkl", 'rb') as pickle_file:
        rlist, qlist, self = pickle.load(pickle_file)
    X = np.load(pklName + ".npy")
    return rlist, qlist, self, X


def iterDistRows(refSeqs, querySeqs, self=True):
    """Gets the ref and query ID for each row of the distance matrix

    Returns an iterable with ref and query ID pairs by row.

    Args:
        refSeqs (list)
            List of reference sequence names.
        querySeqs (list)
            List of query sequence names.
        self (bool)
            Whether a self-comparison, used when constructing a database.

            Requires refSeqs == querySeqs

            Default is True
    Returns:
        ref, query (str, str)
            Iterable of tuples with ref and query names for each distMat row.
    """
    if self:
        if refSeqs != querySeqs:
            raise RuntimeError('refSeqs must equal querySeqs for db building (self = true)')
        for i, ref in enumerate(refSeqs):
            for j in range(i + 1, len(refSeqs)):
                yield(refSeqs[j], ref)
    else:
        for query in querySeqs:
            for ref in refSeqs:
                yield(ref, query)


def writeTmpFile(fileList):
    """Writes a list to a temporary file. Used for turning variable into mash
    input.

    Args:
        fileList (list)
            List of files to write to file
    Returns:
        tmpName (str)
            Name of temp file list written to
    """
    tmpName = mkstemp(suffix=".tmp", dir=".")[1]
    with open(tmpName, 'w') as tmpFile:
        for fileName in fileList:
            tmpFile.write(fileName + "\n")

    return tmpName


def qcDistMat(distMat, refList, queryList, a_max):
    """Checks distance matrix for outliers. At the moment
    just a threshold for accessory distance

    Args:
        distMat (np.array)
            Core and accessory distances
        refList (list)
            Reference labels
        queryList (list)
            Query labels (or refList if self)
        a_max (float)
            Maximum accessory distance to allow

    Returns:
        passed (bool)
            False if any samples failed
    """
    passed = True

    # First check with numpy, which is quicker than iterating over everything
    if np.any(distMat[:,1] > a_max):
        passed = False
        names = iterDistRows(refList, queryList, refList == queryList)
        for i, (ref, query) in enumerate(names):
            if distMat[i,1] > a_max:
                sys.stderr.write("WARNING: Accessory outlier at a=" + str(distMat[i,1]) +
                                 " 1:" + ref + " 2:" + query + "\n")

    return passed


def readClusters(clustCSV):
    """Read a previous reference clustering from CSV

    Args:
        clustCSV (str)
            File name of CSV with previous cluster assignments

    Returns:
        clusters (dict)
            Dictionary of cluster assignments (keys are cluster names, values are
            sets containing samples in the cluster)
    """
    clusters = defaultdict(set)

    with open(clustCSV, 'r') as csv_file:
        header = csv_file.readline()
        for line in csv_file:
            (sample, clust_id) = line.rstrip().split(",")
            clusters[clust_id].add(sample)

    return clusters


def readExternalClusters(clustCSV):
    """Read a cluster definition from CSV (does not have to be PopPUNK
    generated clusters). Rows samples, columns clusters.

    Args:
        clustCSV (str)
            File name of CSV with previous cluster assingments

    Returns:
        extClusters (dict)
            Dictionary of dictionaries of cluster assignments
            (first key cluster assignment name, second key sample, value cluster assignment)
    """
    extClusters = defaultdict(lambda: defaultdict(str))

    extClustersFile = pd.read_csv(clustCSV, index_col = 0, quotechar='"')
    for row in extClustersFile.itertuples():
        for cls_idx, cluster in enumerate(extClustersFile.columns):
            extClusters[str(cluster)][row.Index] = str(row[cls_idx + 1])

    return(extClusters)


def translate_distMat(combined_list, core_distMat, acc_distMat):
    """Convert distances from a square form (2 NxN matrices) to a long form
    (1 matrix with n_comparisons rows and 2 columns).

    Args:
        combined_list
            Combined list of references followed by queries (list)
        core_distMat (numpy.array)
            NxN core distances
        acc_distMat (numpy.array)
            NxN accessory distances

    Returns:
        distMat (numpy.array)
            Distances in long form
    """

    # indices
    i = 0
    j = 1

    # create distmat
    number_pairs = int(0.5 * len(combined_list) * (len(combined_list) - 1))
    distMat = sharedmem.empty((number_pairs, 2))

    # extract distances
    for row in distMat:
        row[0] = core_distMat[i, j]
        row[1] = acc_distMat[i, j]

        if j == len(combined_list) - 1:
            i += 1
            j = i + 1
        else:
            j += 1

    return distMat


def update_distance_matrices(refList, distMat, queryList = None, query_ref_distMat = None,
                             query_query_distMat = None):
    """Convert distances from long form (1 matrix with n_comparisons rows and 2 columns)
    to a square form (2 NxN matrices), with merging of query distances if necessary.

    Args:
        refList (list)
            List of references
        distMat (numpy.array)
            Two column long form list of core and accessory distances
            for pairwise comparisons between reference db sequences
        queryList (list)
            List of queries
        query_ref_distMat (numpy.array)
            Two column long form list of core and accessory distances
            for pairwise comparisons between queries and reference db
            sequences
        query_query_distMat (numpy.array)
            Two column long form list of core and accessory distances
            for pairwise comparisons between query sequences

    Returns:
        seqLabels (list)
            Combined list of reference and query sequences
        coreMat (numpy.array)
            NxN array of core distances for N sequences
        accMat (numpy.array)
            NxN array of accessory distances for N sequences
    """
    seqLabels = refList
    if queryList is not None:
        seqLabels = seqLabels + queryList

    coreMat = np.zeros((len(seqLabels), len(seqLabels)))
    accMat = np.zeros((len(seqLabels), len(seqLabels)))

    # Fill in symmetric matrices for core and accessory distances
    i = 0
    j = 1

    # ref v ref (used for --create-db)
    for row in distMat:
        coreMat[i, j] = row[0]
        coreMat[j, i] = coreMat[i, j]
        accMat[i, j] = row[1]
        accMat[j, i] = accMat[i, j]

        if j == len(refList) - 1:
            i += 1
            j = i + 1
        else:
            j += 1

    # if query vs refdb (--assign-query), also include these comparisons
    if queryList is not None:

        # query v query - symmetric
        i = len(refList)
        j = len(refList)+1
        for row in query_query_distMat:
            coreMat[i, j] = row[0]
            coreMat[j, i] = coreMat[i, j]
            accMat[i, j] = row[1]
            accMat[j, i] = accMat[i, j]
            if j == (len(refList) + len(queryList) - 1):
                i += 1
                j = i + 1
            else:
                j += 1

        # ref v query - asymmetric
        i = len(refList)
        j = 0
        for row in query_ref_distMat:
            coreMat[i, j] = row[0]
            coreMat[j, i] = coreMat[i, j]
            accMat[i, j] = row[1]
            accMat[j, i] = accMat[i, j]
            if j == (len(refList) - 1):
                i += 1
                j = 0
            else:
                j += 1

    # return outputs
    return seqLabels, coreMat, accMat

