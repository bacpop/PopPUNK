# vim: set fileencoding=<utf-8> :
'''Mash functions for database construction'''

# universal
import os
import sys
import subprocess
# additional
import collections
import pickle
from tempfile import mkstemp
from multiprocessing import Pool, Lock
from functools import partial
from itertools import product
from glob import glob
from random import sample
import numpy as np
import sharedmem
import networkx as nx
from scipy import optimize

from .utils import iterDistRows

from .plot import plot_fit

def checkMashVersion(mash_exec):
    """Checks that mash can be run, and is version 2 or higher.
    Exits if version < 2.

    Args:
        mash_exec (str)
            Location of mash executable
    """
    p = subprocess.Popen([mash_exec + ' --version'], shell=True, stdout=subprocess.PIPE)
    version = 0
    for line in iter(p.stdout.readline, ''):
        if line != '':
            version = line.rstrip().decode().split(".")[0]
            break
    if not version.isdigit() or int(version) < 2:
        sys.stderr.write("Need mash v2 or higher\n")
        sys.exit(1)

def getDatabaseName(prefix, k):
    """Gets the name for the mash database for a given k size

    Args:
        prefix (str)
            db prefix
        k (str)
            k-mer size
    Returns:
        db_name (str)
            Name of mash db
    """
    return prefix + "/" + os.path.basename(prefix) + "." + k + ".msh"


def createDatabaseDir(outPrefix, kmers):
    """Creates the directory to write mash sketches to, removing old files if unnecessary

    Args:
        outPrefix (str)
            output db prefix
        kmers (list)
            k-mer sizes in db
    """
    # check for writing
    if os.path.isdir(outPrefix):
        # remove old database files if not needed
        for msh_file in glob(outPrefix + "/" + os.path.basename(outPrefix) + "*.msh"):
            knum = int(msh_file.split('.')[-2])
            if not (kmers == knum).any():
                sys.stderr.write("Removing old database " + msh_file + "\n")
                os.remove(msh_file)
    else:
        try:
            os.makedirs(outPrefix)
        except OSError:
            sys.stderr.write("Cannot create output directory\n")
            sys.exit(1)


def getSketchSize(dbPrefix, klist, mash_exec = 'mash'):
    """Call to ``mash info`` to determine sketch size

    ``sys.exit(1)`` is called if DBs have different sketch sizes

    Args:
        dbprefix (str)
            Prefix for mash databases
        klist (list)
            List of k-mer lengths which databases were constructed at
        mash_exec (str)
            Location of mash executable

    Returns:
        sketchdb (dict)
            Dict of sketch sizes indexed by k-mer size
    """
    sketchdb = {}
    sketch = 0
    oldSketch = 0

    # iterate over kmer lengths
    for k in klist:
        dbname = dbPrefix + "/" + os.path.basename(dbPrefix) + "." + str(k) + ".msh"
        try:
            mash_cmd = mash_exec + " info -t " + dbname
            mash_info = subprocess.Popen(mash_cmd, universal_newlines=True, shell=True, stdout=subprocess.PIPE)
            for line in mash_info.stdout:
                if (line.startswith("#") is False):
                    sketchValues = line.split("\t")
                    if len(sketchValues[0]) > 0:
                        if oldSketch == 0:
                            oldSketch = int(sketchValues[0])
                        else:
                            oldSketch = sketch
                        sketch = int(sketchValues[0])
                        if (sketch == oldSketch):
                            sketchdb[k] = sketch
                        else:
                            sys.stderr.write("Problem with database; sketch size for kmer length " +
                                    str(k) + " is " + str(oldSketch) +
                                    ", but smaller kmers have sketch sizes of " + str(sketch) + "\n")
                            sys.exit(1)

                        break

            mash_info.kill()
            if sketch == 0:
                raise RuntimeError("Could not find sketch size for " + str(k) + "\n")

        except subprocess.CalledProcessError as e:
            sys.stderr.write("Could not get info about " + dbname + "; command " + mash_exec +
                    " info -t " + dbname + " returned " + str(mash_info.returncode) +
                    ": " + e.output + "\n")
            sys.exit(1)

    return sketchdb

def getSeqsInDb(mashSketch, mash_exec = 'mash'):
    """Return an array with the sequences in the passed mash database

    Calls ``mash info -t``

    Args:
        mashSketch (str)
            Mash sketches/database
        mash_exec (str)
            Location of mash executable

    Returns:
        seqs (list)
            List of sequence names in sketch DB
    """
    seqs = []
    mash_cmd = str(mash_exec) + " info -t " + str(mashSketch)
    try:
        mash_info = subprocess.Popen(mash_cmd, universal_newlines=True, shell=True, stdout=subprocess.PIPE)
        for line in mash_info.stdout:
            line = line.rstrip()
            if line != '':
                if line.startswith("#") is False:
                    seqs.append(line.split("\t")[2])

        # Make sure process executed correctly
        if mash_info.poll() != 0:
            raise RuntimeError('mash command "' + mash_cmd + '" failed')
    except subprocess.CalledProcessError as e:
        sys.stderr.write("Could not get info about " + str(mashSketch) + "; command " +
                mash_cmd + " returned " + str(mash_info.returncode) + ": " + e.output + "\n")
        sys.exit(1)

    return seqs

def joinDBs(db1, db2, output, klist, mash_exec = 'mash'):
    """Join two mash sketch databases with ``mash paste``

    Args:
        db1 (str)
            Prefix for db1
        db2 (str)
            Prefix for db2
        output (str)
            Prefix for joined output
        klist (list)
            List of k-mer sizes to sketch
        mash_exec (str)
            Location of mash executable

            (default = 'mash')
    """
    for kmer in klist:
        try:
            join_name = output + "/" + os.path.basename(output) + "." + str(kmer) + ".joined"
            db1_name = db1 + "/" + os.path.basename(db1) + "." + str(kmer) + ".msh"
            db2_name = db2 + "/" + os.path.basename(db2) + "." + str(kmer) + ".msh"

            mash_cmd = mash_exec + " paste " + join_name + " " + db1_name + " " + db2_name
            subprocess.run(mash_cmd, shell=True, check=True)
            os.rename(join_name + ".msh", output + "/" + os.path.basename(output) + "." + str(kmer) + ".msh")
        except subprocess.CalledProcessError as e:
            sys.stderr.write("Could not run command " + mash_cmd + "; returned: " + e.output + "\n")
            sys.exit(1)


def constructDatabase(assemblyList, klist, sketch, oPrefix, ignoreLengthOutliers = False,
                      threads = 1, mash_exec = 'mash', overwrite = False):
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
        sketch (int)
            Size of sketch (``-s`` option)
        oPrefix (str)
            Output prefix for resulting sketch files
        ignoreLengthOutliers (bool)
            Whether to check for outlying genome lengths (and error
            if found)

            (default = True)
        threads (int)
            Number of threads to use

            (default = 1)
        mash_exec (str)
            Location of mash executable

            (default = 'mash')
        overwrite (bool)
            Whether to overwrite sketch DBs, if they already exist.

            (default = False)

    """
    # Genome length needed to calculate prob of random matches
    try:
        input_lengths = []
        input_names = []
        with open(assemblyList, 'r') as assemblyFiles:
            for assembly in assemblyFiles:
                with open(assembly.rstrip(), 'r') as exampleAssembly:
                    genome_length = 0
                    for line in exampleAssembly:
                        if line[0] != ">":
                            genome_length += len(line.rstrip())
                    input_lengths.append(genome_length)
                    input_names.append(assembly)

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
                sys.stderr.write("ERROR: Genomes with outlying lengths detected\n" +
                                 "\n".join(outliers))
                sys.exit(1)

    except FileNotFoundError as e:
        sys.stderr.write("Could not find sequence assembly " + e.filename + "\n"
                         "Assuming length of 2Mb for random match probs.\n")
        genome_length = 2000000

    # check minimum k-mer is above random probability threshold
    assert(genome_length > 0)
    k_min = min(klist)
    if 1/(pow(4, k_min)/float(genome_length) + 1) > 0.05:
        sys.stderr.write("Minimum k-mer length " + str(k_min) + " is too small; please increase to avoid nonsense results\n")
        exit(1)

    # create kmer databases
    if threads > len(klist):
        num_processes = 1
        num_threads = threads
    else:
        num_processes = threads
        num_threads = 1

    # run database construction using multiprocessing
    l = Lock()
    with Pool(processes=num_processes, initializer=init_lock, initargs=(l,)) as pool:
        pool.map(partial(runSketch, assemblyList=assemblyList, sketch=sketch,
                         genome_length=genome_length,oPrefix=oPrefix, mash_exec=mash_exec,
                         overwrite=overwrite, threads=num_threads), klist)

def init_lock(l):
    """Sets a global lock to use when writing to STDERR in :func:`~runSketch`"""
    global lock
    lock = l

def runSketch(k, assemblyList, sketch, genome_length, oPrefix, mash_exec = 'mash', overwrite = False, threads = 1):
    """Actually run the mash sketch command

    Called by :func:`~constructDatabase`

    Args:
        k (int)
            k-mer size to sketch
        assemblyList (list)
            Locations of assembly files to be sketched
        sketch (int)
            Size of sketch (``-s`` option)
        genome_length (int)
            Length of genomes being sketch, for random match probability calculation
        oPrefix (str)
            Output prefix for resulting sketch files
        mash_exec (str)
            Location of mash executable

            (default = 'mash')
        overwrite (bool)
            Whether to overwrite sketch DB, if it already exists.

            (default = False)
        threads (int)
            Number of threads to use in the mash process

            (default = 1)
    """
    # define database name
    dbname = oPrefix + "/" + os.path.basename(oPrefix) + "." + str(k)
    dbfilename = dbname + ".msh"
    # Causes mash sketch to fail at end -- createDatabaseDir should stop this
    if not os.path.isdir(oPrefix):
        sys.stderr.write("Directory " + oPrefix + " does not exist\n")
        sys.exit(1)

    # calculate false positive rate
    random_prob = 1/(pow(4, k)/float(genome_length) + 1)

    # print info. Lock is released once all stderr printing is done to keep
    # all messages from each k-mer length together
    lock.acquire()
    sys.stderr.write("Creating mash database for k = " + str(k) + "\n")
    sys.stderr.write("Random " + str(k) + "-mer probability: " + "{:.2f}".format(random_prob) + "\n")

    # overwrite existing file if instructed
    if os.path.isfile(dbfilename) and overwrite == True:
        sys.stderr.write("Overwriting db: " + dbfilename + "\n")
        os.remove(dbfilename)

    # create new file or leave original intact
    if not os.path.isfile(dbfilename):

        # Release lock before running sketch
        lock.release()

        # Run sketch
        mash_cmd = mash_exec \
                   + " sketch -w 1 -p " + str(threads) \
                   + " -s " + str(sketch[k]) \
                   + " -o " + dbname \
                   + " -k " + str(k) \
                   + " -l " + assemblyList \
                   + " 2> /dev/null"

        subprocess.run(mash_cmd, shell=True, check=True)
    else:
        sys.stderr.write("Found existing mash database " + dbname + ".msh for k = " + str(k) + "\n")
        lock.release()

def queryDatabase(qFile, klist, dbPrefix, queryPrefix, self = True, number_plot_fits = 0,
        no_stream = False, mash_exec = 'mash', threads = 1):
    """Calculate core and accessory distances between query sequences and a sketched database

    For a reference database, runs the query against itself to find all pairwise
    core and accessory distances.

    Uses the relation :math:`pr(a, b) = (1-a)(1-c)^k`

    To get the ref and query name for each row of the returned distances, call to the iterator
    :func:`~PopPUNK.utils.iterDistRows` with the returned refList and queryList

    Args:
        qFile (str)
            File with location of query sequences
        klist (list)
            K-mer sizes to use in the calculation
        dbPrefix (str)
            Prefix for reference mash sketch database created by :func:`~constructDatabase`
        queryPrefix (str)
            Prefix for query mash sketch database created by :func:`~constructDatabase`
        self (bool)
            Set true if query = ref

            (default = True)
        number_plot_fits (int)
            If > 0, the number of k-mer length fits to plot (saved as pdfs).
            Takes random pairs of comparisons and calls :func:`~PopPUNK.plot.plot_fit`

            (default = 0)
        no_stream (bool)
            Rather than streaming mash dist input directly into parser, will write
            through an intermediate temporary file

            (default = False)
        mash_exec (str)
            Location of mash executable

            (default = 'mash')
        threads (int)
            Number of threads to use in the mash process

            (default = 1)

    Returns:
         refList (list)
            Names of reference sequences
         queryList (list)
            Names of query sequences
         distMat (numpy.array)
            Core distances (column 0) and accessory distances (column 1) between
            refList and queryList
    """
    queryList = []
    with open(qFile, 'r') as queryFile:
        for line in queryFile:
            queryList.append(line.rstrip())
    refList = getSeqsInDb(dbPrefix + "/" + os.path.basename(dbPrefix) + "." + str(klist[0]) + ".msh", mash_exec)

    if self:
        if dbPrefix != queryPrefix:
            raise RuntimeError("Must use same db for self query")
        number_pairs = int(0.5 * len(refList) * (len(refList) - 1))
    else:
        number_pairs = int(len(refList) * len(queryList))

    # Pre-assign array for storage. float32 sufficient accuracy for 10**4 sketch size, halves memory use
    raw = sharedmem.empty((number_pairs, len(klist)), dtype=np.float32)

    # iterate through kmer lengths
    for k_idx, k in enumerate(klist):
        row = 0

        # run mash distance query based on current file
        ref_dbname = dbPrefix + "/" + os.path.basename(dbPrefix) + "." + str(k) + ".msh"
        query_dbname = queryPrefix + "/" + os.path.basename(queryPrefix) + "." + str(k) + ".msh"
        # construct mash command
        mash_cmd = mash_exec + " dist -p " + str(threads) + " " + ref_dbname + " " + query_dbname

        if no_stream:
            tmpHandle, tmpName = mkstemp(prefix=os.path.basename(dbPrefix),
                                         suffix=".tmp", dir="./" + os.path.basename(dbPrefix))
            mash_cmd += " > " + tmpName
        mash_cmd += " 2> " + os.path.basename(dbPrefix) + ".err.log"
        sys.stderr.write(mash_cmd + "\n")

        try:
            if no_stream:
                subprocess.run(mash_cmd, shell=True, check=True)
                mashOut = open(tmpName, 'r')
            else:
                rawOutput = subprocess.Popen(mash_cmd, shell=True, stdout=subprocess.PIPE, universal_newlines=True)
                mashOut = rawOutput.stdout

            # Check mash output is consistent with expected order
            # This is ok in all tests, but best to check and exit in case something changes between mash versions
            expected_names = iterDistRows(refList, queryList, self)

            prev_ref = ""
            skip = 0
            skipped = 0
            for line in mashOut:
                # Skip the first row with self and symmetric elements
                if skipped < skip:
                    skipped += 1
                    continue

                mashVals = line.rstrip().split("\t")
                if (len(mashVals) > 2):
                    if self and mashVals[1] != prev_ref:
                        prev_ref = mashVals[1]
                        skip += 1
                        skipped = 1
                    else:
                        mashMatch = mashVals[-1].split('/')
                        (e_ref, e_query) = next(expected_names)
                        if mashVals[0] == e_ref and mashVals[1] == e_query:
                            raw[row, k_idx] = float(mashMatch[0])/int(mashMatch[1])
                            row += 1
                        else:
                            sys.stderr.write("mash dist output order:" + e_query + "," + e_ref + "\n" +
                                             "not as expected: " + mashVals[0] + "," + mashVals[1] + "\n")
                            sys.exit(1)


            if no_stream:
                os.remove(tmpName)
            else:
                rawOutput.wait(timeout=1)
                if rawOutput.poll() != 0:
                    raise RuntimeError('mash dist command "' + mash_cmd + '" failed with raw output ' + str(rawOutput.poll()))

            # Remove the stderr file
            if os.path.isfile(dbPrefix + ".err.log"):
                os.remove(dbPrefix + ".err.log")

        except subprocess.CalledProcessError as e:
            sys.stderr.write("mash dist command " + mash_cmd + " failed with error " + e.message + "\n")
            sys.exit(1)

    # Pre-assign return (to higher precision)
    sys.stderr.write("Calculating core and accessory distances\n")

    # Hessian = 0, so Jacobian for regression is a constant
    jacobian = -np.hstack((np.ones((klist.shape[0], 1)), klist.reshape(-1, 1)))

    # option to plot core/accessory fits. Choose a random number from cmd line option
    if number_plot_fits > 0:
        examples = sample(range(number_pairs), k=number_plot_fits)
        for plot_idx, plot_example in enumerate(sorted(examples)):
            fit = fitKmerCurve(raw[plot_example, :], klist, jacobian)
            plot_fit(klist, raw[plot_example, :], fit,
                    dbPrefix + "/fit_example_" + str(plot_idx + 1),
                    "Example fit " + str(plot_idx + 1) + " (row " + str(plot_example) + ")")

    # run pairwise analyses across kmer lengths, mutating distMat
    # Create range of rows that each thread will work with
    rows_per_thread = int(number_pairs / threads)
    big_threads = number_pairs % threads
    start = 0
    mat_chunks = []
    for thread in range(threads):
        end = start + rows_per_thread
        if thread < big_threads:
            end += 1
        mat_chunks.append((start, end))
        start = end

    distMat = sharedmem.empty((number_pairs, 2))
    with sharedmem.MapReduce(np = threads) as pool:
        pool.map(partial(fitKmerBlock, distMat=distMat, raw = raw, klist=klist, jacobian=jacobian), mat_chunks)

    return(refList, queryList, distMat)


def fitKmerBlock(idxRanges, distMat, raw, klist, jacobian):
    """Multirow wrapper around :func:`~fitKmerCurve` to the specified rows in idxRanges

    Args:
        idxRanges (int, int)
            Tuple of first and last row of slice to calculate
        distMat (numpy.array)
            sharedmem object to store core and accessory distances in (altered in place)
        raw (numpy.array)
            sharedmem object with proportion of k-mer matches for each query-ref pair
            by row, columns are at k-mer lengths in klist
        klist (list)
            List of k-mer lengths to use
        jacobian (numpy.array)
            The Jacobian for the fit, sent to :func:`~fitKmerCurve`

    """
    (start, end) = idxRanges
    distMat[start:end, :] = np.apply_along_axis(fitKmerCurve, 1, raw[start:end, :], klist, jacobian)


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
        sys.stderr.write("Fitting k-mer curve failed: " + e.message +
                         "\nWith mash input " +
                         np.array2string(pairwise, precision=4, separator=',',suppress_small=True) +
                         "\nCheck for low quality input genomes\n")
        exit(0)

    # Return core, accessory
    return(np.flipud(transformed_params))


def readMashDBParams(dbPrefix, kmers, sketch_sizes, mash_exec = 'mash'):
    """Get kmers lengths and sketch sizes from existing database

    Calls :func:`~getKmersFromReferenceDatabase` and :func:`~getSketchSize`
    Uses passed values if db missing

    Args:
        dbPrefix (str)
            Prefix for sketch DB files
        kmers (list)
            Kmers to use if db not found
        sketch_sizes (list)
            Sketch size to use if db not found
        mash_exec (str)
            Location of mash executable

            Default = 'mash'
    Returns:
        kmers (list)
            List of k-mer lengths used in database
        sketch_sizes (list)
            List of sketch sizes used in database
    """

    db_kmers = getKmersFromReferenceDatabase(dbPrefix)
    if len(db_kmers) == 0:
        sys.stderr.write("Couldn't find mash sketches in " + dbPrefix + "\n"
                         "Using command line input parameters for k-mer and sketch sizes\n")
    else:
        kmers = db_kmers
        sketch_sizes = getSketchSize(dbPrefix, kmers, mash_exec)

    return kmers, sketch_sizes


def getKmersFromReferenceDatabase(dbPrefix):
    """Get kmers lengths from existing database

    Parses the database name to determine klist

    Args:
        dbPrefix (str)
            Prefix for sketch DB files
    Returns:
        kmers (list)
            List of k-mer lengths used in database
    """
    # prepare
    knum = []
    fullDbPrefix = dbPrefix + "/" + os.path.basename(dbPrefix) + "."

    # iterate through files
    for msh_file in glob(fullDbPrefix + "*.msh"):
        knum.append(int(msh_file.split('.')[-2]))

    # process kmer list
    knum.sort()
    kmers = np.asarray(knum)
    return kmers



