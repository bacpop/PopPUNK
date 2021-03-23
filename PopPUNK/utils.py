# vim: set fileencoding=<utf-8> :
# Copyright 2018-2020 John Lees and Nick Croucher

'''General utility functions for data read/writing/manipulation in PopPUNK'''

# universal
import os
import sys
# additional
import pickle
import subprocess
from collections import defaultdict
from itertools import chain
from tempfile import mkstemp
from functools import partial
import contextlib

import numpy as np
import pandas as pd
import h5py

import pp_sketchlib

def setGtThreads(threads):
    import graph_tool.all as gt
    # Check on parallelisation of graph-tools
    if gt.openmp_enabled():
        gt.openmp_set_num_threads(threads)
        sys.stderr.write('\nGraph-tools OpenMP parallelisation enabled:')
        sys.stderr.write(' with ' + str(gt.openmp_get_num_threads()) + ' threads\n')

# thanks to Laurent LAPORTE on SO
@contextlib.contextmanager
def set_env(**environ):
    """
    Temporarily set the process environment variables.
    >>> with set_env(PLUGINS_DIR=u'test/plugins'):
    ...   "PLUGINS_DIR" in os.environ
    True
    >>> "PLUGINS_DIR" in os.environ
    False
    """
    old_environ = dict(os.environ)
    os.environ.update(environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)

# Use partials to set up slightly different function calls between
# both possible backends
def setupDBFuncs(args, min_count, qc_dict):
    """Wraps common database access functions from sketchlib and mash,
    to try and make their API more similar

    Args:
        args (argparse.opts)
            Parsed command lines options
        min_count (int)
            Minimum k-mer count for reads
        qc_dict (dict)
            Table of parameters for QC function

    Returns:
        dbFuncs (dict)
            Functions with consistent arguments to use as the database API
    """
    from .sketchlib import checkSketchlibVersion
    from .sketchlib import createDatabaseDir
    from .sketchlib import joinDBs
    from .sketchlib import constructDatabase as constructDatabaseSketchlib
    from .sketchlib import queryDatabase as queryDatabaseSketchlib
    from .sketchlib import readDBParams
    from .sketchlib import getSeqsInDb

    backend = "sketchlib"
    version = checkSketchlibVersion()

    constructDatabase = partial(constructDatabaseSketchlib,
                                strand_preserved = args.strand_preserved,
                                min_count = args.min_kmer_count,
                                use_exact = args.exact_count,
                                qc_dict = qc_dict,
                                use_gpu = args.gpu_sketch,
                                deviceid = args.deviceid)
    queryDatabase = partial(queryDatabaseSketchlib,
                            use_gpu = args.gpu_dist,
                            deviceid = args.deviceid)

    # Dict of DB access functions for assign_query (which is out of scope)
    dbFuncs = {'createDatabaseDir': createDatabaseDir,
               'joinDBs': joinDBs,
               'constructDatabase': constructDatabase,
               'queryDatabase': queryDatabase,
               'readDBParams': readDBParams,
               'getSeqsInDb': getSeqsInDb,
               'backend': backend,
               'backend_version': version
               }

    return dbFuncs

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


def readPickle(pklName, enforce_self=False, distances=True):
    """Loads core and accessory distances saved by :func:`~storePickle`

    Called during ``--fit-model``

    Args:
        pklName (str)
            Prefix for saved files
        enforce_self (bool)
            Error if self == False

            [default = True]
        distances (bool)
            Read the distance matrix

            [default = True]

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
        if enforce_self and not self:
            sys.stderr.write("Old distances " + pklName + ".npy not complete\n")
            sys.stderr.exit(1)
    if distances:
        X = np.load(pklName + ".npy")
    else:
        X = None
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

def listDistInts(refSeqs, querySeqs, self=True):
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
    num_ref = len(refSeqs)
    num_query = len(querySeqs)
    if self:
        if refSeqs != querySeqs:
            raise RuntimeError('refSeqs must equal querySeqs for db building (self = true)')
        for i in range(num_ref):
            for j in range(i + 1, num_ref):
                yield(j, i)
    else:
        comparisons = [(0,0)] * (len(refSeqs) * len(querySeqs))
        for i in range(num_query):
            for j in range(num_ref):
                yield(j, i)

        return comparisons


def qcDistMat(distMat, refList, queryList, ref_db, prefix, qc_dict):
    """Checks distance matrix for outliers.

    Args:
        distMat (np.array)
            Core and accessory distances
        refList (list)
            Reference labels
        queryList (list)
            Query labels (or refList if self)
        ref_db (str)
            Prefix of reference database
        prefix (str)
            Prefix of output files
        qc_dict (dict)
            Dict of QC options

    Returns:
        seq_names_passing (list)
            List of isolates passing QC distance filters
        distMat ([n,2] numpy ndarray)
            Filtered long form distance matrix
    """

    # avoid circular import
    from .prune_db import prune_distance_matrix
    from .sketchlib import removeFromDB
    from .sketchlib import pickTypeIsolate

    # Create overall list of sequences
    if refList == refList:
        seq_names_passing = refList
    else:
        seq_names_passing = refList + queryList

    # Sequences to remove
    to_prune = []

    # Create output directory if it does not exist already
    if not os.path.isdir(prefix):
        try:
            os.makedirs(prefix)
        except OSError:
            sys.stderr.write("Cannot create output directory " + prefix + "\n")
            sys.exit(1)

    # Pick type isolate if not supplied
    if qc_dict['type_isolate'] is None:
        qc_dict['type_isolate'] = pickTypeIsolate(ref_db, seq_names_passing)
        sys.stderr.write('Selected type isolate for distance QC is ' + qc_dict['type_isolate'] + '\n')

    # First check with numpy, which is quicker than iterating over everything
    long_distance_rows = np.where([(distMat[:, 0] > qc_dict['max_pi_dist']) | (distMat[:, 1] > qc_dict['max_a_dist'])])[1].tolist()
    if len(long_distance_rows) > 0:
        names = list(iterDistRows(refList, queryList, refList == queryList))
        # Prune sequences based on reference sequence
        for i in long_distance_rows:
            if names[i][0] == qc_dict['type_isolate']:
                to_prune.append(names[i][1])
            elif names[i][1] == qc_dict['type_isolate']:
                to_prune.append(names[i][0])

    # prune based on distance from reference if provided
    if qc_dict['qc_filter'] == 'stop' and len(to_prune) > 0:
        sys.stderr.write('Outlier distances exceed QC thresholds; prune sequences or raise thresholds\n')
        sys.stderr.write('Problem distances involved sequences ' + ';'.join(to_prune) + '\n')
        sys.exit(1)
    elif qc_dict['qc_filter'] == 'prune' and len(to_prune) > 0:
        if qc_dict['type_isolate'] is None:
            sys.stderr.write('Distances exceeded QC thresholds but no reference isolate supplied\n')
            sys.stderr.write('Problem distances involved sequences ' + ';'.join(to_prune) + '\n')
            sys.exit(1)
        else:
            # Remove sketches
            db_name = ref_db + '/' + os.path.basename(ref_db) + '.h5'
            filtered_db_name = prefix + '/' + 'filtered.' + os.path.basename(prefix) + '.h5'
            removeFromDB(db_name,
                         filtered_db_name,
                         to_prune,
                         full_names = True)
            os.rename(filtered_db_name, db_name)
            # Remove from distance matrix
            seq_names_passing, distMat = prune_distance_matrix(seq_names_passing,
                                                                to_prune,
                                                                distMat,
                                                                prefix + "/" + os.path.basename(prefix) + ".dists")
            # Remove from reflist
            sys.stderr.write('Pruned from the database after failing distance QC: ' + ';'.join(to_prune) + '\n')
    else:
        storePickle(seq_names_passing, seq_names_passing, True, distMat, prefix + "/" + os.path.basename(prefix) + ".dists")

    return seq_names_passing, distMat


def readIsolateTypeFromCsv(clustCSV, mode = 'clusters', return_dict = False):
    """Read cluster definitions from CSV file.

    Args:
        clustCSV (str)
            File name of CSV with isolate assignments
        return_type (str)
            If True, return a dict with sample->cluster instead
            of sets

    Returns:
        clusters (dict)
            Dictionary of cluster assignments (keys are cluster names, values are
            sets containing samples in the cluster). Or if return_dict is set keys
            are sample names, values are cluster assignments.
    """
    # data structures
    if return_dict:
        clusters = defaultdict(dict)
    else:
        clusters = {}

    # read CSV
    clustersCsv = pd.read_csv(clustCSV, index_col = 0, quotechar='"')

    # select relevant columns according to mode
    if mode == 'clusters':
        type_columns = [n for n,col in enumerate(clustersCsv.columns) if ('Cluster' in col)]
    elif mode == 'lineages':
        type_columns = [n for n,col in enumerate(clustersCsv.columns) if ('Rank_' in col or 'overall' in col)]
    elif mode == 'external':
        if len(clustersCsv.columns) == 1:
            type_columns = [0]
        elif len(clustersCsv.columns) > 1:
            type_columns = range((len(clustersCsv.columns)-1))
    else:
        sys.stderr.write('Unknown CSV reading mode: ' + mode + '\n')
        sys.exit(1)

    # read file
    for row in clustersCsv.itertuples():
        for cls_idx in type_columns:
            cluster_name = clustersCsv.columns[cls_idx]
            cluster_name = cluster_name.replace('__autocolour','')
            if return_dict:
                clusters[cluster_name][str(row.Index)] = str(row[cls_idx + 1])
            else:
                if cluster_name not in clusters.keys():
                    clusters[cluster_name] = defaultdict(set)
                clusters[cluster_name][str(row[cls_idx + 1])].add(row.Index)

    # return data structure
    return clusters


def joinClusterDicts(d1, d2):
    """Join two dictionaries returned by :func:`~readIsolateTypeFromCsv` with
    return_dict = True. Useful for concatenating ref and query assignments

    Args:
        d1 (dict of dicts)
            First dictionary to concat
        d2 (dict of dicts)
            Second dictionary to concat

    Returns:
        d1 (dict of dicts)
            d1 with d2 appended
    """
    if d1.keys() != d2.keys():
        sys.stderr.write("Cluster columns not compatible\n")
        sys.exit(1)

    for column in d1.keys():
        # Combine dicts: https://stackoverflow.com/a/15936211
        d1[column] = \
            dict(chain.from_iterable(d.items() for d in (d1[column], d2[column])))

    return d1


def update_distance_matrices(refList, distMat, queryList = None, query_ref_distMat = None,
                             query_query_distMat = None, threads = 1):
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
        threads (int)
            Number of threads to use

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

    if queryList == None:
        coreMat = pp_sketchlib.longToSquare(distMat[:, [0]], threads)
        accMat = pp_sketchlib.longToSquare(distMat[:, [1]], threads)
    else:
        coreMat = pp_sketchlib.longToSquareMulti(distMat[:, [0]],
                                                 query_ref_distMat[:, [0]],
                                                 query_query_distMat[:, [0]],
                                                 threads)
        accMat = pp_sketchlib.longToSquareMulti(distMat[:, [1]],
                                                 query_ref_distMat[:, [1]],
                                                 query_query_distMat[:, [1]],
                                                 threads)

    # return outputs
    return seqLabels, coreMat, accMat

def readRfile(rFile, oneSeq=False):
    """Reads in files for sketching. Names and sequence, tab separated

    Args:
        rFile (str)
            File with locations of assembly files to be sketched
        oneSeq (bool)
            Return only the first sequence listed, rather than a list
            (used with mash)

    Returns:
        names (list)
            Array of sequence names
        sequences (list of lists)
            Array of sequence files
    """
    names = []
    sequences = []
    with open(rFile, 'rU') as refFile:
        for refLine in refFile:
            rFields = refLine.rstrip().split("\t")
            if len(rFields) < 2:
                sys.stderr.write("Input reference list is misformatted\n"
                                 "Must contain sample name and file, tab separated\n")
                sys.exit(1)

            if "/" in rFields[0]:
                sys.stderr.write("Sample names may not contain slashes\n")
                sys.exit(1)
            names.append(rFields[0])
            sample_files = []
            for sequence in rFields[1:]:
                sample_files.append(sequence)

            # Take first of sequence list if using mash
            if oneSeq:
                if len(sample_files) > 1:
                    sys.stderr.write("Multiple sequence found for " + rFields[0] +
                                     ". Only using first\n")
                sequences.append(sample_files[0])
            else:
                sequences.append(sample_files)

    if len(set(names)) != len(names):
        seen = set()
        dupes = set(x for x in names if x in seen or seen.add(x))
        sys.stderr.write("Input contains duplicate names! All names must be unique\n")
        sys.stderr.write("Non-unique names are " + ",".join(dupes) + "\n")
        sys.exit(1)

    # Names are sorted on return
    # We have had issues (though they should be fixed) with unordered input
    # not matching the database. This should help simplify things
    list_iterable = zip(names, sequences)
    sorted_names = sorted(list_iterable)
    tuples = zip(*sorted_names)
    names, sequences = [list(r_tuple) for r_tuple in tuples]

    return (names, sequences)

def isolateNameToLabel(names):
    """Function to process isolate names to labels
    appropriate for visualisation.

    Args:
        names (list)
            List of isolate names.
    Returns:
        labels (list)
            List of isolate labels.
    """
    # useful to have as a function in case we
    # want to remove certain characters
    labels = [name.split('/')[-1].split('.')[0] for name in names]
    return labels


def createOverallLineage(rank_list, lineage_clusters):
    # process multirank lineages
    overall_lineages = {'Rank_' + str(rank):{} for rank in rank_list}
    overall_lineages['overall'] = {}
    isolate_list = lineage_clusters[rank_list[0]].keys()
    for isolate in isolate_list:
        overall_lineage = None
        for rank in rank_list:
            overall_lineages['Rank_' + str(rank)][isolate] = lineage_clusters[rank][isolate]
            if overall_lineage is None:
                overall_lineage = str(lineage_clusters[rank][isolate])
            else:
                overall_lineage = overall_lineage + '-' + str(lineage_clusters[rank][isolate])
        overall_lineages['overall'][isolate] = overall_lineage

    return overall_lineages


def transformLine(s, mean0, mean1):
    """Return x and y co-ordinates for traversing along a line between mean0 and mean1, parameterised by
    a single scalar distance s from the start point mean0.

    Args:
        s (float)
            Distance along line from mean0
        mean0 (numpy.array)
            Start position of line (x0, y0)
        mean1 (numpy.array)
            End position of line (x1, y1)
    Returns:
        x (float)
            The Cartesian x-coordinate
        y (float)
            The Cartesian y-coordinate
    """
    tan_theta = (mean1[1] - mean0[1]) / (mean1[0] - mean0[0])
    x = mean0[0] + s * (1/np.sqrt(1+tan_theta))
    y = mean0[1] + s * (tan_theta/np.sqrt(1+tan_theta))

    return np.array([x, y])


def decisionBoundary(intercept, gradient):
    """Returns the co-ordinates where the triangle the decision boundary forms
    meets the x- and y-axes.

    Args:
        intercept (numpy.array)
            Cartesian co-ordinates of point along line (:func:`~transformLine`)
            which intercepts the boundary
        gradient (float)
            Gradient of the line
    Returns:
        x (float)
            The x-axis intercept
        y (float)
            The y-axis intercept
    """
    x = intercept[0] + intercept[1] * gradient
    y = intercept[1] + intercept[0] / gradient
    return(x, y)
