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
from tempfile import mkstemp
from functools import partial
import collections
try:
    from multiprocessing import Pool, shared_memory
    from multiprocessing.managers import SharedMemoryManager
    NumpyShared = collections.namedtuple('NumpyShared', ('name', 'shape', 'dtype'))
except ImportError as e:
    sys.stderr.write("This version of PopPUNK requires python v3.8 or higher\n")
    sys.exit(0)
import numpy as np
import pandas as pd

# Use partials to set up slightly different function calls between
# both possible backends
def setupDBFuncs(args, kmers, min_count):
    """Wraps common database access functions from sketchlib and mash,
    to try and make their API more similar

    Args:
        args (argparse.opts)
            Parsed command lines options
        kmers (list)
            List of k-mer sizes
        min_count (int)
            Minimum k-mer count for reads

    Returns:
        dbFuncs (dict)
            Functions with consistent arguments to use as the database API
    """
    if args.use_mash:
        from .mash import checkMashVersion
        from .mash import createDatabaseDir
        from .mash import getKmersFromReferenceDatabase
        from .mash import joinDBs as joinDBsMash
        from .mash import constructDatabase as constructDatabaseMash
        from .mash import queryDatabase as queryDBMash
        from .mash import readMashDBParams
        from .mash import getSeqsInDb
    
        # check mash is installed
        backend = "mash"
        version = checkMashVersion(args.mash)

        constructDatabase = partial(constructDatabaseMash, mash_exec = args.mash)
        readDBParams = partial(readMashDBParams, mash_exec = args.mash)
        queryDatabase = partial(queryDBMash, no_stream = args.no_stream, mash_exec = args.mash)
        joinDBs = partial(joinDBsMash, klist = getKmersFromReferenceDatabase(args.output), mash_exec = args.mash)


    else:
        from .sketchlib import checkSketchlibVersion
        from .sketchlib import createDatabaseDir
        from .sketchlib import joinDBs
        from .sketchlib import constructDatabase as constructDatabaseSketchlib
        from .sketchlib import queryDatabase as queryDatabaseSketchlib
        from .sketchlib import readDBParams
        from .sketchlib import getSeqsInDb

        backend = "sketchlib"
        version = checkSketchlibVersion()

        constructDatabase = partial(constructDatabaseSketchlib, strand_preserved = args.strand_preserved, 
                                    min_count = args.min_kmer_count, use_exact = args.exact_count)
        queryDatabase = partial(queryDatabaseSketchlib, use_gpu = args.use_gpu, deviceid = args.deviceid)

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


def readIsolateTypeFromCsv(clustCSV, mode = 'clusters', return_dict = False):
    """Read isolate types from CSV file.

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
        exit(1)
    
    # read file
    for row in clustersCsv.itertuples():
        for cls_idx in type_columns:
            cluster_name = clustersCsv.columns[cls_idx]
            cluster_name = cluster_name.replace('__autocolour','')
            if return_dict:
                clusters[cluster_name][row.Index] = str(row[cls_idx + 1])
            else:
                if cluster_name not in clusters.keys():
                    clusters[cluster_name] = defaultdict(set)
                clusters[cluster_name][str(row[cls_idx + 1])].add(row.Index)

    # return data structure
    return clusters

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
    distMat = np.zeros((number_pairs, 2), dtype=core_distMat.dtype)

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


def get_shared_memory_version(d, m):
    d_raw = m.SharedMemory(size = d.nbytes)
    d_array = np.ndarray(d.shape, dtype = d.dtype, buffer = d_raw.buf)
    d_array[:] = d[:]
    d_array = NumpyShared(name = d_raw.name, shape = d.shape, dtype = d.dtype)
    return d_array, d_raw

def update_distance_matrices(refList, distMat, queryList = None, query_ref_distMat = None,
                             query_query_distMat = None, num_processes = 4):
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

    # share existing distmat
    with SharedMemoryManager() as smm:
        
        # share existing distance matrix
        distMat_array, distMat_raw = get_shared_memory_version(distMat, smm)

        # create shared memory objects - core distances
        coreMat = np.zeros((len(seqLabels), len(seqLabels)), dtype=distMat.dtype)
        coreMat_array, coreMat_raw = get_shared_memory_version(coreMat, smm)
                
        # create shared memory objects - core distances
        accMat = np.zeros((len(seqLabels), len(seqLabels)), dtype=distMat.dtype)
        accMat_array, accMat_raw = get_shared_memory_version(accMat, smm)
        
        # Fill in symmetric matrices for core and accessory distances
        i = 0
        j = 1
        max_j = len(refList) - 1
        coords = defaultdict(tuple)
        for n in range(distMat.shape[0]):
            coords[n] = (i,j)
            if j == max_j:
                i += 1
                j = i + 1
            else:
                j += 1
        
        # fill in matrices
        coord_ranges = get_chunk_ranges(max(coords.keys()), num_processes)
        with Pool(processes = num_processes) as pool:
            pool.map(partial(fill_distance_matrix,
                            coords = coords,
                            dist = distMat_array,
                            core = coreMat_array,
                            acc = accMat_array),
                            coord_ranges)

        # if query vs refdb (--assign-query), also include these comparisons
        if queryList is not None:
            # query v query - symmetric
            qq_coords = defaultdict(tuple)
            i = len(refList)
            j = len(refList)+1
            max_j = (len(refList) + len(queryList) - 1)
            for n in range(query_query_distMat.shape[0]):
                qq_coords[n] = (i,j)
                if j == max_j:
                    i += 1
                    j = i + 1
                else:
                    j += 1
                    
            # fill in matrices
            qq_coord_ranges = get_chunk_ranges(query_query_distMat.shape[0], num_processes)
            # share existing distance matrix
            qq_distMat_array, qq_distMat_raw = get_shared_memory_version(query_query_distMat, smm)
            with Pool(processes = num_processes) as pool:
                pool.map(partial(fill_distance_matrix,
                                coords = qq_coords,
                                dist = qq_distMat_array,
                                core = coreMat_array,
                                acc = accMat_array),
                                qq_coord_ranges)

            # ref v query - asymmetric
            qr_coords = defaultdict(tuple)
            i = len(refList)
            j = 0
            max_j = (len(refList) - 1)
            for n in range(query_ref_distMat.shape[0]):
                qr_coords[n] = (i,j)
                if j == max_j:
                    i += 1
                    j = 0
                else:
                    j += 1
                    
            # fill in matrices
            qr_coord_ranges = get_chunk_ranges(query_ref_distMat.shape[0], num_processes)
            qr_distMat_array, qq_distMat_raw = get_shared_memory_version(query_ref_distMat, smm)
            with Pool(processes = num_processes) as pool:
                pool.map(partial(fill_distance_matrix,
                                coords = qr_coords,
                                dist = qr_distMat_array,
                                core = coreMat_array,
                                acc = accMat_array),
                                qr_coord_ranges)

        # copy data out of shared memory
        coreMat[:] = np.ndarray(coreMat_array.shape, dtype=coreMat_array.dtype, buffer=coreMat_raw.buf)[:]
        accMat[:] = np.ndarray(accMat_array.shape, dtype=accMat_array.dtype, buffer=accMat_raw.buf)[:]
        
    # return outputs
    return seqLabels, coreMat, accMat

def assembly_qc(assemblyList, klist, ignoreLengthOutliers, estimated_length):
    """Calculates random match probability based on means of genomes
    in assemblyList, and looks for length outliers.

    Calls a hard sys.exit(1) if failing!

    Args:
        assemblyList (str)
            File with locations of assembly files to be sketched
        klist (list)
            List of k-mer sizes to sketch
        ignoreLengthOutliers (bool)
            Whether to check for outlying genome lengths (and error
            if found)
        estimated_length (int)
            Estimated length of genome, if not calculated from data

    Returns:
        genome_length (int)
            Average length of assemblies
        max_prob (float)
            Random match probability at minimum k-mer length
    """
    # Genome length needed to calculate prob of random matches
    genome_length = estimated_length # assume 2 Mb in the absence of other information

    try:
        input_lengths = []
        input_names = []
        for sampleAssembly in assemblyList:
            if type(sampleAssembly) != list:
                sampleAssembly = [sampleAssembly]
            for assemblyFile in sampleAssembly:
                input_genome_length = 0
                with open(assemblyFile, 'r') as exampleAssembly:
                    for line in exampleAssembly:
                        if line[0] != ">":
                            input_genome_length += len(line.rstrip())
            input_lengths.append(input_genome_length)
            input_names.append(sampleAssembly)

        # Check for outliers
        outliers = []
        sigma = 5
        if not ignoreLengthOutliers:
            genome_length = np.mean(np.array(input_lengths))
            outlier_low = genome_length - sigma*np.std(input_lengths)
            outlier_high = genome_length + sigma*np.std(input_lengths)
            for length, name in zip(input_lengths, input_names):
                if length < outlier_low or length > outlier_high:
                    outliers.append(name)
            if outliers:
                sys.stderr.write("ERROR: Genomes with outlying lengths detected\n")
                for outlier in outliers:
                    sys.stderr.write('\n'.join(outlier) + '\n')
                sys.exit(1)

    except FileNotFoundError as e:
        sys.stderr.write("Could not find sequence assembly " + e.filename + "\n"
                         "Assuming length of " + str(estimated_length) + " for random match probs.\n")

    except UnicodeDecodeError as e:
        sys.stderr.write("Could not read input file. Is it zipped?\n"
                         "Assuming length of " + str(estimated_length) + " for random match probs.\n")

    # check minimum k-mer is above random probability threshold
    if genome_length <= 0:
        genome_length = estimated_length
        sys.stderr.write("WARNING: Could not detect genome length. Assuming " + str(estimated_length) + "\n")
    if genome_length > 10000000:
        sys.stderr.write("WARNING: Average length over 10Mb - are these assemblies?\n")

    k_min = min(klist)
    max_prob = 1/(pow(4, k_min)/float(genome_length) + 1)
    if max_prob > 0.05:
        sys.stderr.write("Minimum k-mer length " + str(k_min) + " is too small for genome length " + str(genome_length) +"; results will be adjusted for random match probabilities\n")
    if k_min < 6:
        sys.stderr.write("Minimum k-mer length is too low; please increase to at least 6\n")
        exit(1)

    return (int(genome_length), max_prob)

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
        sys.stderr.write("Input contains duplicate names! All names must be unique\n")
        sys.exit(1)

    return (names, sequences)

def get_chunk_ranges(N, nb):
    """ Calculates boundaries for dividing distances array
    into chunks for parallelisation.

    Args:
        N (int)
            Number of rows in array
        nb (int)
            Number of blocks into which to divide array.

    Returns:
        range_sizes (list of tuples)
            Limits of blocks for dividing array.
    """
    step = N / nb
    range_sizes = [(round(step*i), round(step*(i+1))) for i in range(nb)]
    # extend to end of distMat
    range_sizes[len(range_sizes) - 1] = (range_sizes[len(range_sizes) - 1][0],N)
    # return ranges
    return range_sizes

def fill_distance_matrix(num_range, coords = None, dist = None, core = None, acc = None):
    # load shared memory objects
    dist_shm = shared_memory.SharedMemory(name = dist.name)
    dist = np.ndarray(dist.shape, dtype = dist.dtype, buffer = dist_shm.buf)
    
    core_shm = shared_memory.SharedMemory(name = core.name)
    core = np.ndarray(core.shape, dtype = core.dtype, buffer = core_shm.buf)

    acc_shm = shared_memory.SharedMemory(name = acc.name)
    acc = np.ndarray(acc.shape, dtype = acc.dtype, buffer = acc_shm.buf)
    # iterate and fill
    n = num_range[0]
    for n in range(num_range[0],num_range[1]):
        upper = coords[n]
        lower = (coords[n][1],coords[n][0])
        core[upper] = dist[n,0]
        core[lower] = dist[n,0]
        acc[upper] = dist[n,1]
        acc[lower] = dist[n,1]
