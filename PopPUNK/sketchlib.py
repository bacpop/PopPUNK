# vim: set fileencoding=<utf-8> :
# Copyright 2018-2020 John Lees and Nick Croucher

'''Sketchlib functions for database construction'''

# universal
import os
import sys
import subprocess
# additional
import collections
import pickle
import time
from tempfile import mkstemp
from multiprocessing import Pool, Lock
from functools import partial
from itertools import product
from glob import glob
from random import sample
import numpy as np
from scipy import optimize

# Try to import sketchlib
try:
    import pp_sketchlib
    import h5py
except ImportError as e:
    sys.stderr.write("Sketchlib backend not available")
    sys.exit(1)

from .__init__ import SKETCHLIB_MAJOR, SKETCHLIB_MINOR, SKETCHLIB_PATCH
from .utils import iterDistRows
from .utils import readRfile
from .plot import plot_fit

sketchlib_exe = "poppunk_sketch"

def checkSketchlibVersion():
    """Checks that sketchlib can be run, and returns version

    Returns:
        version (str)
            Version string
    """
    p = subprocess.Popen([sketchlib_exe + ' --version'], shell=True, stdout=subprocess.PIPE)
    version = 0
    for line in iter(p.stdout.readline, ''):
        if line != '':
            version = line.rstrip().decode().split(" ")[1]
            break

    sketchlib_version = [int(v) for v in version.split(".")]
    if sketchlib_version[0] < SKETCHLIB_MAJOR or \
        sketchlib_version[0] == SKETCHLIB_MAJOR and sketchlib_version[1] < SKETCHLIB_MINOR or \
        sketchlib_version[0] == SKETCHLIB_MAJOR and sketchlib_version[1] == SKETCHLIB_MINOR and sketchlib_version[2] < SKETCHLIB_PATCH:
        sys.stderr.write("This version of PopPUNK requires sketchlib "
                            "v" + str(SKETCHLIB_MAJOR) + \
                            "." + str(SKETCHLIB_MINOR) + \
                            "." + str(SKETCHLIB_PATCH) + " or higher\n")
        sys.exit(1)

    return version

def checkSketchlibLibrary():
    """Gets the location of the sketchlib library

    Returns:
        lib (str)
            Location of sketchlib .so/.dyld
    """
    sketchlib_loc = pp_sketchlib.__file__
    return(sketchlib_loc)

def createDatabaseDir(outPrefix, kmers):
    """Creates the directory to write sketches to, removing old files if unnecessary

    Args:
        outPrefix (str)
            output db prefix
        kmers (list)
            k-mer sizes in db
    """
    # check for writing
    if os.path.isdir(outPrefix):
        # remove old database files if not needed
        db_file = outPrefix + "/" + os.path.basename(outPrefix) + ".h5"
        if os.path.isfile(db_file):
            ref_db = h5py.File(db_file, 'r')
            for sample_name in list(ref_db['sketches'].keys()):
                knum = ref_db['sketches/' + sample_name].attrs['kmers']
                remove_prev_db = False
                for kmer_length in knum:
                    if not (kmer_length in knum):
                        sys.stderr.write("Previously-calculated k-mer size " + str(kmer_length) +
                                        " not in requested range (" + str(knum) + ")\n")
                        remove_prev_db = True
                        break
                if remove_prev_db:
                    sys.stderr.write("Removing old database " + db_file + "\n")
                    os.remove(db_file)
                    break

    else:
        try:
            os.makedirs(outPrefix)
        except OSError:
            sys.stderr.write("Cannot create output directory\n")
            sys.exit(1)


def getSketchSize(dbPrefix):
    """Determine sketch size, and ensures consistent in whole database

    ``sys.exit(1)`` is called if DBs have different sketch sizes

    Args:
        dbprefix (str)
            Prefix for mash databases

    Returns:
        sketchSize (int)
            sketch size (64x C++ definition)
        codonPhased (bool)
            whether the DB used codon phased seeds
    """
    db_file = dbPrefix + "/" + os.path.basename(dbPrefix) + ".h5"
    ref_db = h5py.File(db_file, 'r')
    try:
        codon_phased = ref_db['sketches'].attrs['codon_phased']
    except KeyError:
        codon_phased = False

    prev_sketch = 0
    for sample_name in list(ref_db['sketches'].keys()):
        sketch_size = ref_db['sketches/' + sample_name].attrs['sketchsize64']
        if prev_sketch == 0:
            prev_sketch = sketch_size
        elif sketch_size != prev_sketch:
            sys.stderr.write("Problem with database; sketch sizes for sample " +
                             sample_name + " is " + str(prev_sketch) +
                             ", but smaller kmers have sketch sizes of " + str(sketch_size) + "\n")
            sys.exit(1)

    return int(sketch_size), codon_phased

def getKmersFromReferenceDatabase(dbPrefix):
    """Get kmers lengths from existing database

    Args:
        dbPrefix (str)
            Prefix for sketch DB files
    Returns:
        kmers (list)
            List of k-mer lengths used in database
    """
    db_file = dbPrefix + "/" + os.path.basename(dbPrefix) + ".h5"
    ref_db = h5py.File(db_file, 'r')
    prev_kmer_sizes = []
    for sample_name in list(ref_db['sketches'].keys()):
        kmer_size = ref_db['sketches/' + sample_name].attrs['kmers']
        if len(prev_kmer_sizes) == 0:
            prev_kmer_sizes = kmer_size
        elif np.any(kmer_size != prev_kmer_sizes):
            sys.stderr.write("Problem with database; kmer lengths inconsistent: " +
                             str(kmer_size) + " vs " + str(prev_kmer_sizes) + "\n")
            sys.exit(1)

    prev_kmer_sizes.sort()
    kmers = np.asarray(prev_kmer_sizes)
    return kmers

def readDBParams(dbPrefix):
    """Get kmers lengths and sketch sizes from existing database

    Calls :func:`~getKmersFromReferenceDatabase` and :func:`~getSketchSize`
    Uses passed values if db missing

    Args:
        dbPrefix (str)
            Prefix for sketch DB files

    Returns:
        kmers (list)
            List of k-mer lengths used in database
        sketch_sizes (list)
            List of sketch sizes used in database
        codonPhased (bool)
            whether the DB used codon phased seeds
    """
    db_kmers = getKmersFromReferenceDatabase(dbPrefix)
    if len(db_kmers) == 0:
        sys.stderr.write("Couldn't find sketches in " + dbPrefix + "\n")
        sys.exit(1)
    else:
        sketch_sizes, codon_phased = getSketchSize(dbPrefix)

    return db_kmers, sketch_sizes, codon_phased


def getSeqsInDb(dbname):
    """Return an array with the sequences in the passed database

    Args:
        dbname (str)
            Sketches database filename

    Returns:
        seqs (list)
            List of sequence names in sketch DB
    """
    seqs = []
    ref = h5py.File(dbname, 'r')
    for sample_name in list(ref['sketches'].keys()):
        seqs.append(sample_name)

    return seqs

def joinDBs(db1, db2, output):
    """Join two sketch databases with the low-level HDF5 copy interface

    Args:
        db1 (str)
            Prefix for db1
        db2 (str)
            Prefix for db2
        output (str)
            Prefix for joined output
    """
    join_name = output + "/" + os.path.basename(output) + ".h5"
    db1_name = db1 + "/" + os.path.basename(db1) + ".h5"
    db2_name = db2 + "/" + os.path.basename(db2) + ".h5"

    hdf1 = h5py.File(db1_name, 'r')
    hdf2 = h5py.File(db2_name, 'r')
    hdf_join = h5py.File(join_name + ".tmp", 'w') # add .tmp in case join_name exists

    # Can only copy into new group, so for second file these are appended one at a time
    try:
        hdf1.copy('sketches', hdf_join)
        if 'random' in hdf1:
            hdf1.copy('random', hdf_join)
        join_grp = hdf_join['sketches']
        read_grp = hdf2['sketches']
        for dataset in read_grp:
            join_grp.copy(read_grp[dataset], dataset)

    except RuntimeError as e:
        sys.stderr.write("ERROR: " + str(e) + "\n")
        sys.stderr.write("Joining sketches failed, try running without --update-db\n")
        sys.exit(1)

    # Clean up
    hdf1.close()
    hdf2.close()
    hdf_join.close()
    os.rename(join_name + ".tmp", join_name)


def removeFromDB(db_name, out_name, removeSeqs, full_names = False):
    """Remove sketches from the DB the low-level HDF5 copy interface

    Args:
        db_name (str)
            Prefix for hdf database
        out_name (str)
            Prefix for output (pruned) database
        removeSeqs (list)
            Names of sequences to remove from database
        full_names (bool)
            If True, db_name and out_name are the full paths to h5 files
    """
    removeSeqs = set(removeSeqs)
    if not full_names:
        db_file = db_name + "/" + os.path.basename(db_name) + ".h5"
        out_file = out_name + "/" + os.path.basename(out_name) + ".tmp.h5"
    else:
        db_file = db_name
        out_file = out_name

    hdf_in = h5py.File(db_file, 'r')
    hdf_out = h5py.File(out_file, 'w')

    try:
        if 'random' in hdf_in.keys():
            hdf_in.copy('random', hdf_out)
        out_grp = hdf_out.create_group('sketches')
        read_grp = hdf_in['sketches']
        for attr_name, attr_val in read_grp.attrs.items():
            out_grp.attrs.create(attr_name, attr_val)

        removed = []
        for dataset in read_grp:
            if dataset not in removeSeqs:
                out_grp.copy(read_grp[dataset], dataset)
            else:
                removed.append(dataset)
    except RuntimeError as e:
        sys.stderr.write("ERROR: " + str(e) + "\n")
        sys.stderr.write("Error while deleting sequence " + dataset + "\n")
        sys.exit(1)

    missed = removeSeqs.difference(set(removed))
    if len(missed) > 0:
        sys.stderr.write("WARNING: Did not find samples to remove:\n")
        sys.stderr.write("\t".join(missed) + "\n")

    # Clean up
    hdf_in.close()
    hdf_out.close()

def constructDatabase(assemblyList, klist, sketch_size, oPrefix,
                        threads, overwrite,
                        strand_preserved, min_count,
                        use_exact, qc_dict, calc_random = True,
                        codon_phased = False,
                        use_gpu = False, deviceid = 0):
    """Sketch the input assemblies at the requested k-mer lengths

    A multithread wrapper around :func:`~runSketch`. Threads are used to either run multiple sketch
    processes for each klist value, or increase the threads used by each ``mash sketch`` process
    if len(klist) > threads.

    Also calculates random match probability based on length of first genome
    in assemblyList.

    Args:
        assemblyList (str)
            File with locations of assembly files to be sketched
        klist (list)
            List of k-mer sizes to sketch
        sketch_size (int)
            Size of sketch (``-s`` option)
        oPrefix (str)
            Output prefix for resulting sketch files
        threads (int)
            Number of threads to use (default = 1)
        overwrite (bool)
            Whether to overwrite sketch DBs, if they already exist.
            (default = False)
        strand_preserved (bool)
            Ignore reverse complement k-mers (default = False)
        min_count (int)
            Minimum count of k-mer in reads to include
            (default = 0)
        use_exact (bool)
            Use exact count of k-mer appearance in reads
            (default = False)
        qc_dict (dict)
            Dict containg QC settings
        calc_random (bool)
            Add random match chances to DB (turn off for queries)
        codon_phased (bool)
            Use codon phased seeds
            (default = False)
        use_gpu (bool)
            Use GPU for read sketching
            (default = False)
        deviceid (int)
            GPU device id
            (default = 0)
    """
    # read file names
    names, sequences = readRfile(assemblyList)

    # create directory
    dbname = oPrefix + "/" + os.path.basename(oPrefix)
    dbfilename = dbname + ".h5"
    if os.path.isfile(dbfilename) and overwrite == True:
        sys.stderr.write("Overwriting db: " + dbfilename + "\n")
        os.remove(dbfilename)

    # generate sketches
    pp_sketchlib.constructDatabase(dbname,
                                   names,
                                   sequences,
                                   klist,
                                   sketch_size,
                                   codon_phased,
                                   False,
                                   not strand_preserved,
                                   min_count,
                                   use_exact,
                                   threads,
                                   use_gpu,
                                   deviceid)

    # QC sequences
    if qc_dict['run_qc']:
        filtered_names = sketchlibAssemblyQC(oPrefix,
                                             klist,
                                             qc_dict,
                                             strand_preserved,
                                             threads)
    else:
        filtered_names = names

    # Add random matches if required
    # (typically on for reference, off for query)
    if (calc_random):
        addRandom(oPrefix,
                  filtered_names,
                  klist,
                  strand_preserved,
                  overwrite = True,
                  threads = threads)

    # return filtered file names
    return filtered_names


def addRandom(oPrefix, sequence_names, klist,
              strand_preserved = False, overwrite = False, threads = 1):
    """Add chance of random match to a HDF5 sketch DB

    Args:
        oPrefix (str)
            Sketch database prefix
        sequence_names (list)
            Names of sequences to include in calculation
        klist (list)
            List of k-mer sizes to sketch
        strand_preserved (bool)
            Set true to ignore rc k-mers
        overwrite (str)
            Set true to overwrite existing random match chances
        threads (int)
            Number of threads to use (default = 1)
    """
    if len(sequence_names) <= 2:
        sys.stderr.write("Cannot add random match chances with this few genomes\n")
    else:
        dbname = oPrefix + "/" + os.path.basename(oPrefix)
        hdf_in = h5py.File(dbname + ".h5", 'r+')

        if 'random' in hdf_in:
            if overwrite:
                del hdf_in['random']
            else:
                sys.stderr.write("Using existing random match chances in DB\n")
                return

        hdf_in.close()
        pp_sketchlib.addRandom(dbname,
                            sequence_names,
                            klist,
                            not strand_preserved,
                            threads)

def queryDatabase(rNames, qNames, dbPrefix, queryPrefix, klist, self = True, number_plot_fits = 0,
                  threads = 1, use_gpu = False, deviceid = 0):
    """Calculate core and accessory distances between query sequences and a sketched database

    For a reference database, runs the query against itself to find all pairwise
    core and accessory distances.

    Uses the relation :math:`pr(a, b) = (1-a)(1-c)^k`

    To get the ref and query name for each row of the returned distances, call to the iterator
    :func:`~PopPUNK.utils.iterDistRows` with the returned refList and queryList

    Args:
        rNames (list)
            Names of references to query
        qNames (list)
            Names of queries
        dbPrefix (str)
            Prefix for reference mash sketch database created by :func:`~constructDatabase`
        queryPrefix (str)
            Prefix for query mash sketch database created by :func:`~constructDatabase`
        klist (list)
            K-mer sizes to use in the calculation
        self (bool)
            Set true if query = ref
            (default = True)
        number_plot_fits (int)
            If > 0, the number of k-mer length fits to plot (saved as pdfs).
            Takes random pairs of comparisons and calls :func:`~PopPUNK.plot.plot_fit`
            (default = 0)
        threads (int)
            Number of threads to use in the mash process
            (default = 1)
        use_gpu (bool)
            Use a GPU for querying
            (default = False)
        deviceid (int)
            Index of the CUDA GPU device to use
            (default = 0)

    Returns:
         refList (list)
            Names of reference sequences
         queryList (list)
            Names of query sequences
         distMat (numpy.array)
            Core distances (column 0) and accessory distances (column 1) between
            refList and queryList
    """
    ref_db = dbPrefix + "/" + os.path.basename(dbPrefix)

    if self:
        if dbPrefix != queryPrefix:
            raise RuntimeError("Must use same db for self query")
        qNames = rNames

        # Calls to library
        distMat = pp_sketchlib.queryDatabase(ref_db, ref_db, rNames, rNames, klist,
                                             True, False, threads, use_gpu, deviceid)

        # option to plot core/accessory fits. Choose a random number from cmd line option
        if number_plot_fits > 0:
            jacobian = -np.hstack((np.ones((klist.shape[0], 1)), klist.reshape(-1, 1)))
            for plot_idx in range(number_plot_fits):
                example = sample(rNames, k=2)
                raw = np.zeros(len(klist))
                corrected = np.zeros(len(klist))
                for kidx, kmer in enumerate(klist):
                    raw[kidx] = pp_sketchlib.jaccardDist(ref_db, example[0], example[1], kmer, False)
                    corrected[kidx] = pp_sketchlib.jaccardDist(ref_db, example[0], example[1], kmer, True)
                raw_fit = fitKmerCurve(raw, klist, jacobian)
                corrected_fit = fitKmerCurve(corrected, klist, jacobian)
                plot_fit(klist,
                         raw,
                         raw_fit,
                         corrected,
                         corrected_fit,
                         dbPrefix + "/" + dbPrefix + "_fit_example_" + str(plot_idx + 1),
                         "Example fit " + str(plot_idx + 1) + " - " +  example[0] + " vs. " + example[1])
    else:
        duplicated = set(rNames).intersection(set(qNames))
        if len(duplicated) > 0:
            sys.stderr.write("Sample names in query are contained in reference database:\n")
            sys.stderr.write("\n".join(duplicated))
            sys.stderr.write("Unique names are required!\n")
            sys.exit(1)

        # Calls to library
        query_db = queryPrefix + "/" + os.path.basename(queryPrefix)
        distMat = pp_sketchlib.queryDatabase(ref_db, query_db, rNames, qNames, klist,
                                             True, False, threads, use_gpu, deviceid)

    return(rNames, qNames, distMat)

def calculateQueryQueryDistances(dbFuncs, qlist, kmers,
                                 queryDB, threads = 1):
    """Calculates distances between queries.

    Args:
        dbFuncs (list)
            List of backend functions from :func:`~PopPUNK.utils.setupDBFuncs`
        rlist (list)
            List of reference names
        qlist (list)
            List of query names
        kmers (list)
            List of k-mer sizes
        queryDB (str)
            Query database location
        threads (int)
            Number of threads to use if new db created
            (default = 1)

    Returns:
        qlist1 (list)
            Ordered list of queries
        distMat (numpy.array)
            Query-query distances
    """

    queryDatabase = dbFuncs['queryDatabase']

    qlist1, qlist2, distMat = queryDatabase(rNames = qlist,
                                            qNames = qlist,
                                            dbPrefix = queryDB,
                                            queryPrefix = queryDB,
                                            klist = kmers,
                                            self = True,
                                            number_plot_fits = 0,
                                            threads = threads)

    return qlist1, distMat

def sketchlibAssemblyQC(prefix, klist, qc_dict, strand_preserved, threads):
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
    hdf_in = h5py.File(db_name, 'r+')

    # try/except structure to prevent h5 corruption
    try:
        # process data structures
        read_grp = hdf_in['sketches']

        seq_length = {}
        seq_ambiguous = {}
        retained = []
        failed = []

        # iterate through sketches
        for dataset in read_grp:
            # test thresholds
            remove = False
            seq_length[dataset] = hdf_in['sketches'][dataset].attrs['length']
            seq_ambiguous[dataset] = hdf_in['sketches'][dataset].attrs['missing_bases']

        # calculate thresholds
        # get mean length
        genome_lengths = np.fromiter(seq_length.values(), dtype = int)
        mean_genome_length = np.mean(genome_lengths)

        # calculate length threshold unless user-supplied
        if qc_dict['length_range'][0] is None:
            lower_length = mean_genome_length - \
                qc_dict['length_sigma'] * np.std(genome_lengths)
            upper_length = mean_genome_length + \
                qc_dict['length_sigma'] * np.std(genome_lengths)
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
                    remove = True
                    qc_file.write(dataset + '\tAbove upper length threshold\n')
                if qc_dict['upper_n'] is not None and seq_ambiguous[dataset] > qc_dict['upper_n']:
                    remove = True
                    qc_file.write(dataset + '\tAmbiguous sequence too high\n')
                elif seq_ambiguous[dataset] > qc_dict['prop_n'] * seq_length[dataset]:
                    remove = True
                    qc_file.write(dataset + '\tAmbiguous sequence too high\n')

                if remove:
                    sys.stderr.write(dataset + ' failed QC\n')
                    failed.append(dataset)
                else:
                    retained.append(dataset)

            # retain sketches of failed samples
            if qc_dict['retain_failures']:
                removeFromDB(db_name,
                             prefix + '/' + 'failed.' + os.path.basename(prefix) + '.h5',
                             retained,
                             full_names = True)
            # new database file if pruning
            if qc_dict['qc_filter'] == 'prune':
                filtered_db_name = prefix + '/' + 'filtered.' + os.path.basename(prefix) + '.h5'
                removeFromDB(db_name,
                             prefix + '/' + 'filtered.' + os.path.basename(prefix) + '.h5',
                             failed,
                             full_names = True)
                os.rename(filtered_db_name, db_name)

    # if failure still close files to avoid corruption
    except:
        hdf_in.close()
        sys.stderr.write('Problem processing h5 databases during QC - aborting\n')

        print("Unexpected error:", sys.exc_info()[0], file = sys.stderr)
        raise

    # stop if at least one sample fails QC and option is not continue/prune
    if failed_sample and qc_dict['qc_filter'] == 'stop':
        sys.stderr.write('Sequences failed QC filters - details in ' + \
                         prefix + '/' + os.path.basename(prefix) + \
                         '_qcreport.txt\n')
        sys.exit(1)
    elif qc_dict['qc_filter'] == 'continue':
        retained = retained + failed

    # stop if no sequences pass QC
    if len(retained) == 0:
        sys.stderr.write('No sequences passed QC filters - please adjust your settings\n')
        sys.exit(1)

    # remove random matches if already present
    if 'random' in hdf_in:
        del hdf_in['random']
    hdf_in.close()

    return retained

def fitKmerCurve(pairwise, klist, jacobian):
    """Fit the function :math:`pr = (1-a)(1-c)^k`

    Supply ``jacobian = -np.hstack((np.ones((klist.shape[0], 1)), klist.reshape(-1, 1)))``

    Args:
        pairwise (numpy.array)
            Proportion of shared k-mers at k-mer values in klist
        klist (list)
            k-mer sizes used
        jacobian (numpy.array)
            Should be set as above (set once to try and save memory)

    Returns:
        transformed_params (numpy.array)
            Column with core and accessory distance
    """
    # curve fit pr = (1-a)(1-c)^k
    # log pr = log(1-a) + k*log(1-c)
    # a = p[0]; c = p[1] (will flip on return)
    try:
        distFit = optimize.least_squares(fun=lambda p, x, y: y - (p[0] + p[1] * x),
                                     x0=[0.0, -0.01],
                                     jac=lambda p, x, y: jacobian,
                                     args=(klist, np.log(pairwise)),
                                     bounds=([-np.inf, -np.inf], [0, 0]))
        transformed_params = 1 - np.exp(distFit.x)
    except ValueError as e:
        sys.stderr.write("Fitting k-mer curve failed: " + format(e) +
                         "\nWith mash input " +
                         np.array2string(pairwise, precision=4, separator=',',suppress_small=True) +
                         "\nCheck for low quality input genomes\n")
        exit(0)

    # Return core, accessory
    return(np.flipud(transformed_params))
