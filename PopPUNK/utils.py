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

import numpy as np
import pandas as pd
import h5py

import pp_sketchlib

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
            tmpFile.write(fileName + '\t' + fileName + "\n")

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


def sketchlib_assembly_qc(prefix, klist, qc_dict, strand_preserved, threads):
    """Calculates random match probability based on means of genomes
    in assemblyList, and looks for length outliers.

    Args:
        prefix (str)
            Prefix of output files
        klist (list)
            List of k-mer sizes to sketch
        qc_dict (dict)
            Dictionary of QC parameters
        strand_preserved (bool)
            Ignore reverse complement k-mers (default = False)
        threads (int)
            Number of threads to use in parallelisation

    Returns:
        retained (list)
            List of sequences passing QC filters
    """

    # open databases
    db_name = prefix + '/' + os.path.basename(prefix) + '.h5'
    hdf_in = h5py.File(db_name, 'r')
    # new database file if pruning
    if qc_dict['qc_filter'] == 'prune':
        filtered_db_name = prefix + '/' + 'filtered.' + os.path.basename(prefix) + '.h5'
        hdf_out = h5py.File(filtered_db_name, 'w')
    # retain sketches of failed samples
    if qc_dict['retain_failures']:
        failed_db_name = prefix + '/' + 'failed.' + os.path.basename(prefix) + '.h5'
        hdf_fail = h5py.File(failed_db_name, 'w')
    
    # try/except structure to prevent h5 corruption
    try:
        # process data structures
        read_grp = hdf_in['sketches']
        if qc_dict['qc_filter'] == 'prune':
            out_grp = hdf_out.create_group('sketches')
        if qc_dict['retain_failures']:
            fail_grp = hdf_fail.create_group('sketches')
        seq_length = {}
        seq_ambiguous = {}
        seq_excluded = {}
        removed = []
        retained = []
        
        # iterate through sketches
        for dataset in read_grp:
            # test thresholds
            remove = False
            seq_length[dataset] = hdf_in['sketches'][dataset].attrs['length']
            seq_ambiguous[dataset] = hdf_in['sketches'][dataset].attrs['missing_bases']
            # if no filtering to be undertaken, retain all sequences
            if qc_dict['qc_filter'] == 'continue':
                retained.append(dataset)

        # calculate thresholds
        # get mean length
        genome_lengths = np.fromiter(seq_length.values(), dtype = int)
        mean_genome_length = np.mean(genome_lengths)
        
        # calculate length threshold unless user-supplied
        if qc_dict['length_range'][0] is None:
            lower_length = mean_genome_length - qc_dict['length_sigma']*np.std(genome_lengths)
            upper_length = mean_genome_length + qc_dict['length_sigma']*np.std(genome_lengths)
        else:
            lower_length, upper_length = qc_dict['length_range']

        # open file to report QC failures
        with open(prefix + '/' + os.path.basename(prefix) + '_qcreport.txt', 'a+') as qc_file:
        
            # iterate through and filter
            failed_sample = False
            for dataset in seq_length.keys():
                        
                # determine if sequence passes filters
                remove = False
                if seq_length[dataset] < lower_length:
                    remove = True
                    qc_file.write(dataset + '\tBelow lower length threshold\n')
                elif seq_length[dataset] > upper_length:
                    qc_file.write(dataset + '\tAbove upper length threshold\n')
                if qc_dict['upper_n'] is not None and seq_ambiguous[dataset] > qc_dict['upper_n']:
                    remove = True
                    qc_file.write(dataset + '\tAmbiguous sequence too high\n')
                elif seq_ambiguous[dataset] > qc_dict['prop_n'] * seq_length[dataset]:
                    remove = True
                    qc_file.write(dataset + '\tAmbiguous sequence too high\n')

                # write to files
                if remove:
                    sys.stderr.write(dataset + ' failed QC\n')
                    failed_sample = True
                    if qc_dict['retain_failures']:
                        fail_grp.copy(read_grp[dataset], dataset)
                else:
                    if qc_dict['qc_filter'] == 'prune':
                        out_grp.copy(read_grp[dataset], dataset)
                        retained.append(dataset)
        
            # get kmers from original database
            db_kmers = hdf_in['sketches'][retained[0]].attrs['kmers']
            
            # close files
            hdf_in.close()
            if qc_dict['qc_filter'] == 'prune':
                hdf_out.close()
            if qc_dict['retain_failures']:
                hdf_fail.close()

        # replace original database with pruned version
        if qc_dict['qc_filter'] == 'prune':
            os.rename(filtered_db_name, db_name)
    
    # if failure still close files to avoid corruption
    except:
        hdf_in.close()
        if qc_dict['qc_filter'] == 'prune':
           hdf_out.close()
        if qc_dict['retain_failures']:
           hdf_fail.close()
        sys.stderr.write('Problem processing h5 databases during QC - aborting\n')
        sys.exit(1)
    
    
    # stop if at least one sample fails QC and option is not continue/prune
    if failed_sample and qc_dict['qc_filter'] == 'stop':
        sys.stderr.write('Sequences failed QC filters - details in ' + prefix + '/' + os.path.basename(prefix) + '_qcreport.txt\n')
        sys.exit(1)
    
    # calculate random matches if any sequences pass QC filters
    if len(retained) == 0:
        sys.stderr.write('No sequences passed QC filters - please adjust your settings\n')
        sys.exit(1)
    use_rc = not strand_preserved
    db_name_prefix = prefix + '/' + os.path.basename(prefix)
    # remove random matches if already present
    hdf_in = h5py.File(db_name, 'r+')
    if 'random' in hdf_in:
        del hdf_in['random']
    hdf_in.close()
    pp_sketchlib.addRandom(db_name_prefix, retained, db_kmers.tolist(), use_rc, threads)
    
    return retained
