# vim: set fileencoding=<utf-8> :
# Copyright 2018-2023 John Lees and Nick Croucher

'''General utility functions for data read/writing/manipulation in PopPUNK'''

# universal
import os
import sys
# additional
import pickle
import multiprocessing
from collections import defaultdict
from itertools import chain
from tempfile import mkstemp
from functools import partial
import contextlib

import numpy as np
import pandas as pd

try:
    import cudf
    import rmm
    import cupy
    from numba import cuda
except ImportError:
    pass

import poppunk_refine
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

# https://stackoverflow.com/a/17954769
from contextlib import contextmanager
@contextmanager
def stderr_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stderr.fileno()

    def _redirect_stderr(to):
        sys.stderr.close()
        os.dup2(to.fileno(), fd)
        sys.stderr = os.fdopen(fd, 'w')

    with os.fdopen(os.dup(fd), 'w') as old_stderr:
        with open(to, 'w') as file:
            _redirect_stderr(to=file)
        try:
            yield
        finally:
            _redirect_stderr(to=old_stderr)

# Use partials to set up slightly different function calls between
# both possible backends
def setupDBFuncs(args):
    """Wraps common database access functions from sketchlib and mash,
    to try and make their API more similar

    Args:
        args (argparse.opts)
            Parsed command lines options
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
        if enforce_self and (not self or rlist != qlist):
            sys.stderr.write("Old distances " + pklName + ".npy not complete\n")
            sys.exit(1)
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


def readIsolateTypeFromCsv(clustCSV, mode = 'clusters', return_dict = False):
    """Read cluster definitions from CSV file.

    Args:
        clustCSV (str)
            File name of CSV with isolate assignments
        mode (str)
            Type of file to read 'clusters', 'lineages', or 'external'
        return_type (str)
            If True, return a dict with sample->cluster instead
            of sets
            [default = False]

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
    matching_cols = set(d1.keys()).intersection(d2.keys())
    if len(matching_cols) == 0:
        sys.stderr.write("Cluster columns do not match between sets being combined\n")
        sys.stderr.write(f"{d1.keys()} {d2.keys()}\n")
        sys.exit(1)

    missing_cols = []
    for column in d1.keys():
        if column in matching_cols:
            # Combine dicts: https://stackoverflow.com/a/15936211
            d1[column] = \
                dict(chain.from_iterable(d.items() for d in (d1[column], d2[column])))
        else:
            missing_cols.append(column)

    for missing in missing_cols:
        del d1[missing]

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
        coreMat = pp_sketchlib.longToSquare(distVec=distMat[:, [0]],
                                            num_threads=threads)
        accMat = pp_sketchlib.longToSquare(distVec=distMat[:, [1]],
                                           num_threads=threads)
    else:
        coreMat = pp_sketchlib.longToSquareMulti(distVec=distMat[:, [0]],
                                                 query_ref_distVec=query_ref_distMat[:, [0]],
                                                 query_query_distVec=query_query_distMat[:, [0]],
                                                 num_threads=threads)
        accMat = pp_sketchlib.longToSquareMulti(distVec=distMat[:, [1]],
                                                query_ref_distVec=query_ref_distMat[:, [1]],
                                                query_query_distVec=query_query_distMat[:, [1]],
                                                num_threads=threads)

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

            # Take first of sequence list
            if oneSeq:
                if len(sample_files) > 1:
                    sys.stderr.write("Multiple sequence found for " + rFields[0] +
                                     ". Only using first\n")
                sequences.append(sample_files[0])
            else:
                sequences.append(sample_files)

    # Process names to ensure compatibility with downstream software
    names = isolateNameToLabel(names)

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
    labels = [name.split('/')[-1].replace('.','_').replace(':','').replace('(','_').replace(')','_') \
                        for name in names]
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
    dx = mean1[0] - mean0[0]
    dy = mean1[1] - mean0[1]
    ds = np.sqrt(dx**2 + dy**2)
    x = mean0[0] + s * (dx / ds)
    y = mean0[1] + s * (dy / ds)

    return np.array([x, y])


def decisionBoundary(intercept, gradient, adj = 0.0):
    """Returns the co-ordinates where the triangle the decision boundary forms
    meets the x- and y-axes.

    Args:
        intercept (numpy.array)
            Cartesian co-ordinates of point along line (:func:`~transformLine`)
            which intercepts the boundary
        gradient (float)
            Gradient of the line
        adj (float)
            Distance by which to shift the interception point
    Returns:
        x (float)
            The x-axis intercept
        y (float)
            The y-axis intercept
    """
    if adj != 0.0:
        original_hypotenuse = (intercept[0]**2 + intercept[1]**2)**0.5
        length_ratio = (original_hypotenuse + adj)/original_hypotenuse
        intercept[0] = intercept[0] * length_ratio
        intercept[1] = intercept[1] * length_ratio
    x = intercept[0] + intercept[1] * gradient
    y = intercept[1] + intercept[0] / gradient
    return(x, y)

def check_and_set_gpu(use_gpu, gpu_lib, quit_on_fail = False):
    """Check GPU libraries can be loaded and set managed memory.

    Args:
        use_gpu (bool)
            Whether GPU packages have been requested
        gpu_lib (bool)
            Whether GPU packages are available
    Returns:
        use_gpu (bool)
            Whether GPU packages can be used
    """
    # load CUDA libraries
    if use_gpu and not gpu_lib:
        if quit_on_fail:
            sys.stderr.write('Unable to load GPU libraries; exiting\n')
            sys.exit(1)
        else:
            sys.stderr.write('Unable to load GPU libraries; using CPU libraries '
            'instead\n')
            use_gpu = False

    # Set memory management for large networks
    if use_gpu:
        multiprocessing.set_start_method('spawn', force=True)
        rmm.reinitialize(managed_memory=True)
        if "cupy" in sys.modules:
            cupy.cuda.set_allocator(rmm.allocators.cupy.rmm_cupy_allocator)
        if "cuda" in sys.modules:
            cuda.set_memory_manager(rmm.allocators.numba.RMMNumbaManager)
        assert(rmm.is_initialized())

    return use_gpu

def read_rlist_from_distance_pickle(fn, allow_non_self = True):
    """Return the list of reference sequences from a distance pickle.

    Args:
        fn (str)
            Name of distance pickle
        allow_non_self (bool)
            Whether non-self distance datasets are permissible
    Returns:
        rlist (list)
            List of reference sequence names
    """
    with open(fn, 'rb') as pickle_file:
        rlist, qlist, self = pickle.load(pickle_file)
        if not allow_non_self and not self:
            sys.stderr.write("Thi analysis requires an all-v-all"
                             " distance dataset\n")
            sys.exit(1)
    return rlist

def get_match_search_depth(rlist,rank_list):
    """Return a default search depth for lineage model fitting.

    Args:
        rlist (list)
            List of sequences in database
        rank_list (list)
            List of ranks to be used to fit lineage models
    Returns:
        max_search_depth (int)
            Maximum kNN used for lineage model fitting
    """
    # Defaults to maximum of 10% of database size, unless this is smaller than the maximum search rank
    max_search_depth = max([int(0.1*len(rlist)),int(1.1*max(rank_list)),int(1+max(rank_list))])
    # Cannot be higher than the number of comparisons
    if max_search_depth > len(rlist) - 1:
        max_search_depth = len(rlist) - 1
    return max_search_depth
