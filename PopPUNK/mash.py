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

from .plot import plot_fit

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
    return prefix + "/" + prefix + "." + k + ".msh"


def createDatabaseDir(outPrefix, kmers):
    """Creates the directory to write mash sketches to, removing old files if unnecessary

    Args:
        outPrefix (str)
            output db prefix
        kmers (list)
            k-mer sizes in db
    """
    outputDir = os.getcwd() + "/" + outPrefix
    # check for writing
    if os.path.isdir(outputDir):
        # remove old database files if not needed
        for msh_file in glob(outputDir + "/" + outPrefix + "*.msh"):
            knum = int(msh_file.split('.')[-2])
            if not (kmers == knum).any():
                sys.stderr.write("Removing old database " + msh_file + "\n")
                os.remove(msh_file)
    else:
        try:
            os.makedirs(outputDir)
        except:
            sys.stderr.write("Cannot create output directory\n")
            sys.exit(1)

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
        dbname = "./" + dbPrefix + "/" + dbPrefix + "." + str(k) + ".msh"
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

            mash_info.wait(timeout=15)
            if mash_info.poll() != 0:
                raise RuntimeError('mash info command "'+mash_cmd+'" failed with raw output '+str(mash_info.poll()))

        except subprocess.CalledProcessError as e:
            sys.stderr.write("Could not get info about " + dbname + "; command " + mash_exec +
                    " info -t " + dbname + " returned " + str(mash_info.returncode) +
                    ": " + e.message + "\n")
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
        sys.stderr.write("Could not get info about " + dbname + "; command " +
                mash_cmd + " returned " + str(mash_info.returncode) + ": "+e.message+"\n")
        sys.exit(1)

    return seqs

def constructDatabase(assemblyList, klist, sketch, oPrefix, threads = 1, mash_exec = 'mash', overwrite = False):
    """Sketch the input assemblies at the requested k-mer lengths

    A multithread wrapper around :func:`~runSketch`. Threads are used to either run multiple sketch
    processes for each klist value, or increase the threads used by each ``mash sketch`` process
    if len(klist) > threads.

    Also calculates random match probability based on length of first genome
    in assemblyList.

    Args:
        assemblyList (list)
            Locations of assembly files to be sketched
        klist (list)
            List of k-mer sizes to sketch
        sketch (int)
            Size of sketch (``-s`` option)
        oPrefix (str)
            Output prefix for resulting sketch files
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
    genome_length = 1 # min of 1 to avoid div/0 errors
    with open(assemblyList, 'r') as assemblyFiles:
       exampleFile = assemblyFiles.readline()
       with open(exampleFile.rstrip(), 'r') as exampleAssembly:
           for line in exampleAssembly:
               if line[0] != ">":
                   genome_length += len(line.rstrip())

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
    dbname = "./" + oPrefix + "/" + oPrefix + "." + str(k)
    dbfilename = dbname + ".msh"

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
    :func:`~iterDistRows` with the returned refList and queryList

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
    refList = getSeqsInDb("./" + dbPrefix + "/" + dbPrefix + "." + str(klist[0]) + ".msh", mash_exec)

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
        ref_dbname = "./" + dbPrefix + "/" + dbPrefix + "." + str(k) + ".msh"
        query_dbname = "./" + queryPrefix + "/" + queryPrefix + "." + str(k) + ".msh"
        # construct mash command
        mash_cmd = mash_exec + " dist -p " + str(threads) + " " + ref_dbname + " " + query_dbname

        if no_stream:
            tmpHandle, tmpName = mkstemp(prefix=dbPrefix, suffix=".tmp", dir="./" + dbPrefix)
            mash_cmd += " > " + tmpName
        mash_cmd += " 2> " + dbPrefix + ".err.log"
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
                        print("test_match: "+str(mashVals[0])+"|"+str(e_ref)+" - "+str(mashVals[0] == e_ref)+"\t"+str(mashVals[1])+"|"+str(e_query)+" - "+str(mashVals[1] == e_query))
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
                if rawOutput.poll() != 0:
                    raise RuntimeError('mash dist command "'+mash_cmd+'" failed with raw output '+str(rawOutput.poll()))
                else:
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
            sharedmem object to store core and accessory distances in(altered in place)
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
    distFit = optimize.least_squares(fun=lambda p, x, y: y - (p[0] + p[1] * x),
                                     x0=[0.0, -0.01],
                                     jac=lambda p, x, y: jacobian,
                                     args=(klist, np.log(pairwise)),
                                     bounds=([-np.inf, -np.inf], [0, 0]))
    transformed_params = 1 - np.exp(distFit.x)

    # Return core, accessory
    return(np.flipud(transformed_params))

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
    fullDbPrefix = "./" + dbPrefix + "/" + dbPrefix + "."

    # iterate through files
    for msh_file in glob(fullDbPrefix + "*.msh"):
        knum.append(int(msh_file.split('.')[-2]))

    # process kmer list
    knum.sort()
    kmers = np.asarray(knum)
    return kmers

def printQueryOutput(rlist, qlist, X, outPrefix, self):
    """Write calculated distances between query and ref to a text file

    First three arguments are the return values from :func:`~queryDatabase`

    Args:
        rlist (list)
            Names of reference sequences
        qlist (list)
            Names of query sequences
        X (numpy.array)
            Core and accessory distances
        outPrefix (str)
            Prefix for output file
        self (bool)
            Whether :func:`~queryDatabase` was run with self (rlist = qlist)
    """
    # check if output directory exists and generate if not
    if not os.path.isdir(outPrefix):
        os.makedirs(outPrefix)

    # get names order
    names = iterDistRows(rlist, qlist, self)

    # open output file
    outFileName = outPrefix + "/" + outPrefix + ".search.out"
    with open(outFileName, 'w') as oFile:
        oFile.write("\t".join(['Query', 'Reference', 'Core', 'Accessory']) + "\n")
        for i, (ref, query) in enumerate(names):
            oFile.write("\t".join([query, ref, str(X[i,0]), str(X[i,1])]) + "\n")

def readFilteringList(fn):
    """Read a list of assemblies that should be removed from the dataset
        
        Passes argument of filename, returns a list
        
        Args:
            fn (string)
                Name of newline-delimited list of sequences to be removed
                
        Returns:
            flist (list)
                List of strings of assemblies to be removed
    """

    # open and parse
    flist = []
    with open(fn, 'r') as filteringFile:
        for line in filteringFile:
            if len(line) > 1:
                flist.append(line.rstrip())

    return flist

def filterData(rlist, qlist, self, X):

    return rlist, qlist, self, X
