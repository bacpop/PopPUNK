# vim: set fileencoding=<utf-8> :
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
import sharedmem
import networkx as nx
from scipy import optimize
import h5py

import pp_sketchlib

from .mash import fitKmerCurve
from .utils import iterDistRows
from .utils import assembly_qc
from .utils import readRfile
from .plot import plot_fit


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
        db_file = outPrefix + "/" + outPrefix + ".h5"
        if os.path.isfile(db_file):
            ref_db = h5py.File(db_file, 'r')
            for sample_name in list(ref_db['sketches'].keys()):
                knum = ref_db['sketches/' + sample_name].attrs['kmers']
                if not (kmers == knum).any():
                    sys.stderr.write("Removing old database " + db_file + "\n")
                    sys.stderr.write("(k-mer size " + str(knum) +
                                    " not in requested range " + str(knum) + ")\n")
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
    """
    db_file = dbPrefix + "/" + dbPrefix + ".h5"
    ref_db = h5py.File(db_file, 'r')
    prev_sketch = 0
    for sample_name in list(ref_db['sketches'].keys()):
        sketch_size = ref_db['sketches/' + sample_name].attrs()['sketchsize64'] * 64
        if prev_sketch == 0:
            prev_sketch = sketch_size
        elif sketch_size != prev_sketch:
            sys.stderr.write("Problem with database; sketch size for kmer length " +
                             str(k) + " is " + str(oldSketch) +
                             ", but smaller kmers have sketch sizes of " + str(sketch) + "\n")
            sys.exit(1)

    return sketch_size

def getKmersFromReferenceDatabase(dbPrefix):
    """Get kmers lengths from existing database

    Args:
        dbPrefix (str)
            Prefix for sketch DB files
    Returns:
        kmers (list)
            List of k-mer lengths used in database
    """
    db_file = dbPrefix + "/" + dbPrefix + ".h5"
    ref_db = h5py.File(db_file, 'r')
    prev_kmer_sizes = []
    for sample_name in list(ref_db['sketches'].keys()):
        kmer_size = ref_db['sketches/' + sample_name].attrs()['kmers']
        if len(prev_kmer_sizes) == 0:
            prev_kmer_sizes = kmer_size
        elif np.any(kmer_size != prev_kmer_sizes):
            sys.stderr.write("Problem with database; kmer lengths inconsistent: " +
                             str(kmer_size) + " vs " + str(prev_kmer_sizes) + "\n")
            sys.exit(1)

    prev_kmer_sizes.sort()
    kmers = np.asarray(prev_kmer_sizes)
    return kmers

def readDBParams(dbPrefix, kmers, sketch_sizes):
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
        sketch_sizes = getSketchSize(dbPrefix)

    return kmers, sketch_sizes


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
    """Join two mash sketch databases with ``mash paste``

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
    hdf_join = h5py.File(join_name, 'w')

    h5py.h5o.copy(hdf1.id, 'sketches', hdf_join.id, 'sketches')
    h5py.h5o.copy(hdf2.id, 'sketches', hdf_join.id, 'sketches')


def constructDatabase(assemblyList, klist, sketch_size, oPrefix, ignoreLengthOutliers = False,
                      threads = 1, overwrite = False, reads = False):
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
        ignoreLengthOutliers (bool)
            Whether to check for outlying genome lengths (and error
            if found)

            (default = False)
        threads (int)
            Number of threads to use

            (default = 1)
        overwrite (bool)
            Whether to overwrite sketch DBs, if they already exist.

            (default = False)
        reads (bool)
            If any reads are being used as input, do not run QC

            (default = False)
    """
    names, sequences = readRfile(assemblyList)
    if not reads:
        genome_length, max_prob = assembly_qc(sequences, klist, ignoreLengthOutliers)
        sys.stderr.write("Worst random match probability at " + str(min(klist)) + 
                            "-mers: " + "{:.2f}".format(max_prob) + "\n")
    
    dbname = oPrefix + "/" + oPrefix
    dbfilename = dbname + ".h5"
    if os.path.isfile(dbfilename) and overwrite == True:
        sys.stderr.write("Overwriting db: " + dbfilename + "\n")
        os.remove(dbfilename)

    pp_sketchlib.constructDatabase(dbname, names, sequences, klist, sketch_size, threads)
    
def queryDatabase(rFile, qFile, dbPrefix, queryPrefix, klist, self = True, number_plot_fits = 0,
                  threads = 1):
    """Calculate core and accessory distances between query sequences and a sketched database

    For a reference database, runs the query against itself to find all pairwise
    core and accessory distances.

    Uses the relation :math:`pr(a, b) = (1-a)(1-c)^k`

    To get the ref and query name for each row of the returned distances, call to the iterator
    :func:`~PopPUNK.utils.iterDistRows` with the returned refList and queryList

    Args:
        rFile (str)
            File with location of ref sequences
        qFile (str)
            File with location of query sequences
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

                    klist (list)
            List of k-mer sizes to sketch

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
        rNames, rSequences = readRfile(rFile)
        qNames = rNames
        
        # Calls to library
        distMat = pp_sketchlib.queryDatabase(ref_db, ref_db, rNames, rNames, klist, threads)

        # option to plot core/accessory fits. Choose a random number from cmd line option
        if number_plot_fits > 0:
            jacobian = -np.hstack((np.ones((klist.shape[0], 1)), klist.reshape(-1, 1)))
            for plot_idx in range(number_plot_fits):
                example = sample(rNames, k=2)
                raw = np.zeros(len(klist))
                for kidx, kmer in enumerate(klist):
                    raw[kidx] = pp_sketchlib.jaccardDist(ref_db, example[0], example[1], kmer)

                fit = fitKmerCurve(raw, klist, jacobian)
                plot_fit(klist, raw, fit,
                        dbPrefix + "/fit_example_" + str(plot_idx + 1),
                        "Example fit " + str(plot_idx + 1) + " - " +  example[0] + " vs. " + example[1])
    else:
        qNames, qSequences = readRfile(qFile)
        rNames = getSeqsInDb(ref_db)
        query_db = queryPrefix + "/" + os.path.basename(queryPrefix)

        if set(rNames).intersection(set(qNames)) > 0:
            sys.stderr.write("Sample names in query are contained in reference database\n")
            sys.stderr.write("Unique names are required!\n")
            exit(0)

        # Calls to library
        distMat = pp_sketchlib.queryDatabase(ref_db, query_db, rNames, qNames, klist, threads)

    return(rNames, qNames, distMat)
