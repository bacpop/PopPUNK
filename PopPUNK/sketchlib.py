# vim: set fileencoding=<utf-8> :
# Copyright 2018-2023 John Lees and Nick Croucher

'''Sketchlib functions for database construction'''

# universal
import os
import sys
import subprocess
# additional
import re
from random import sample
import numpy as np
from scipy import optimize

import pp_sketchlib
import h5py

from .__init__ import SKETCHLIB_MAJOR, SKETCHLIB_MINOR, SKETCHLIB_PATCH
from .utils import readRfile, stderr_redirected
from .plot import plot_fit

sketchlib_exe = "sketchlib"

def checkSketchlibVersion():
    """Checks that sketchlib can be run, and returns version

    Returns:
        version (str)
            Version string
    """
    sketchlib_version = [0, 0, 0]
    try:
        version = pp_sketchlib.version

    # Older versions didn't export attributes
    except AttributeError:
        try:
            p = subprocess.Popen([sketchlib_exe + ' --version'], shell=True, stdout=subprocess.PIPE)
            version = 0
            for line in iter(p.stdout.readline, ''):
                if line != '':
                    version = line.rstrip().decode().split(" ")[1]
                    break

        except IndexError:
            sys.stderr.write("WARNING: Sketchlib version could not be found\n")

    version = re.sub(r'^v', '', version) # Remove leading v
    sketchlib_version = [int(v) for v in version.split(".")]
    if sketchlib_version[0] < SKETCHLIB_MAJOR or \
        sketchlib_version[0] == SKETCHLIB_MAJOR and sketchlib_version[1] < SKETCHLIB_MINOR or \
        sketchlib_version[0] == SKETCHLIB_MAJOR and sketchlib_version[1] == SKETCHLIB_MINOR and sketchlib_version[2] < SKETCHLIB_PATCH:
        sys.stderr.write("This version of PopPUNK requires sketchlib "
                            "v" + str(SKETCHLIB_MAJOR) + \
                            "." + str(SKETCHLIB_MINOR) + \
                            "." + str(SKETCHLIB_PATCH) + " or higher\n")
        sys.stderr.write("Continuing... but safety not guaranteed\n")

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
            Prefix for databases

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

def joinDBs(db1, db2, output, update_random = None, full_names = False):
    """Join two sketch databases with the low-level HDF5 copy interface

    Args:
        db1 (str)
            Prefix for db1
        db2 (str)
            Prefix for db2
        output (str)
            Prefix for joined output
        update_random (dict)
            Whether to re-calculate the random object. May contain
            control arguments strand_preserved and threads (see :func:`addRandom`)
        full_names (bool)
            If True, db_name and out_name are the full paths to h5 files

    """
    
    if not full_names:
        join_prefix = output + "/" + os.path.basename(output)
        db1_name = db1 + "/" + os.path.basename(db1) + ".h5"
        db2_name = db2 + "/" + os.path.basename(db2) + ".h5"
    else:
        db1_name = db1
        db2_name = db2
        join_prefix = output

    hdf1 = h5py.File(db1_name, 'r')
    hdf2 = h5py.File(db2_name, 'r')
    hdf_join = h5py.File(join_prefix + ".tmp.h5", 'w') # add .tmp in case join_name exists

    # Can only copy into new group, so for second file these are appended one at a time
    try:
        hdf1.copy('sketches', hdf_join)

        join_grp = hdf_join['sketches']
        read_grp = hdf2['sketches']
        for dataset in read_grp:
            join_grp.copy(read_grp[dataset], dataset)

        # Copy or update random matches
        if update_random is not None:
            threads = 1
            strand_preserved = False
            if isinstance(update_random, dict):
                if "threads" in update_random:
                    threads = update_random["threads"]
                if "strand_preserved" in update_random:
                    strand_preserved = update_random["strand_preserved"]

            sequence_names = list(hdf_join['sketches'].keys())
            kmer_size = hdf_join['sketches/' + sequence_names[0]].attrs['kmers']

            # Need to close before adding random
            hdf_join.close()
            if len(sequence_names) > 2:
                sys.stderr.write("Updating random match chances\n")
                pp_sketchlib.addRandom(db_name=join_prefix + ".tmp",
                                       samples=sequence_names,
                                       klist=kmer_size,
                                       use_rc=(not strand_preserved),
                                       num_threads=threads)
        elif 'random' in hdf1:
            hdf1.copy('random', hdf_join)

        # Clean up
        hdf1.close()
        hdf2.close()
        if update_random is None:
            hdf_join.close()

    except RuntimeError as e:
        sys.stderr.write("ERROR: " + str(e) + "\n")
        sys.stderr.write("Joining sketches failed, try running without --update-db\n")
        sys.exit(1)

    # Rename results to correct location
    os.rename(join_prefix + ".tmp.h5", join_prefix + ".h5")


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
                        use_exact, calc_random = True,
                        codon_phased = False,
                        use_gpu = False, deviceid = 0):
    """Sketch the input assemblies at the requested k-mer lengths

    A multithread wrapper around :func:`~runSketch`. Threads are used to either run multiple sketch
    processes for each klist value.

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
    Returns:
        names (list)
            List of names included in the database (from rfile)
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
    pp_sketchlib.constructDatabase(db_name=dbname,
                                   samples=names,
                                   files=sequences,
                                   klist=klist,
                                   sketch_size=sketch_size,
                                   codon_phased=codon_phased,
                                   calc_random=False,
                                   use_rc=not strand_preserved,
                                   min_count=min_count,
                                   exact=use_exact,
                                   num_threads=threads,
                                   use_gpu=use_gpu,
                                   device_id=deviceid)

    # Add random matches if required
    # (typically on for reference, off for query)
    if (calc_random):
        addRandom(oPrefix,
                  names,
                  klist,
                  strand_preserved,
                  overwrite = True,
                  threads = threads)

    return names


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
        pp_sketchlib.addRandom(db_name=dbname,
                               samples=sequence_names,
                               klist=klist,
                               use_rc=(not strand_preserved),
                               num_threads=threads)

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
            Prefix for reference sketch database created by :func:`~constructDatabase`
        queryPrefix (str)
            Prefix for query sketch database created by :func:`~constructDatabase`
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
            Number of threads to use in the process
            (default = 1)
        use_gpu (bool)
            Use a GPU for querying
            (default = False)
        deviceid (int)
            Index of the CUDA GPU device to use
            (default = 0)

    Returns:
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
        distMat = pp_sketchlib.queryDatabase(ref_db_name=ref_db,
                                             query_db_name=ref_db,
                                             rList=rNames,
                                             qList=rNames,
                                             klist=klist,
                                             random_correct=True,
                                             jaccard=False,
                                             num_threads=threads,
                                             use_gpu=use_gpu,
                                             device_id=deviceid)

        # option to plot core/accessory fits. Choose a random number from cmd line option
        if number_plot_fits > 0:
            jacobian = -np.hstack((np.ones((klist.shape[0], 1)), klist.reshape(-1, 1)))
            for plot_idx in range(number_plot_fits):
                example = sample(rNames, k=2)
                raw = np.zeros(len(klist))
                corrected = np.zeros(len(klist))
                with stderr_redirected(): # Hide the many progress bars
                    raw = pp_sketchlib.queryDatabase(ref_db_name=ref_db,
                                                    query_db_name=ref_db,
                                                    rList=[example[0]],
                                                    qList=[example[1]],
                                                    klist=klist,
                                                    random_correct=False,
                                                    jaccard=True,
                                                    num_threads=threads,
                                                    use_gpu = False)
                    corrected = pp_sketchlib.queryDatabase(ref_db_name=ref_db,
                                                        query_db_name=ref_db,
                                                        rList=[example[0]],
                                                        qList=[example[1]],
                                                        klist=klist,
                                                        random_correct=True,
                                                        jaccard=True,
                                                        num_threads=threads,
                                                        use_gpu = False)
                raw_fit = fitKmerCurve(raw[0], klist, jacobian)
                corrected_fit = fitKmerCurve(corrected[0], klist, jacobian)
                plot_fit(klist,
                         raw[0],
                         raw_fit,
                         corrected[0],
                         corrected_fit,
                         ref_db + "_fit_example_" + str(plot_idx + 1),
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
        distMat = pp_sketchlib.queryDatabase(ref_db_name=ref_db,
                                             query_db_name=query_db,
                                             rList=rNames,
                                             qList=qNames,
                                             klist=klist,
                                             random_correct=True,
                                             jaccard=False,
                                             num_threads=threads,
                                             use_gpu=use_gpu,
                                             device_id=deviceid)

        # option to plot core/accessory fits. Choose a random number from cmd line option
        if number_plot_fits > 0:
            jacobian = -np.hstack((np.ones((klist.shape[0], 1)), klist.reshape(-1, 1)))
            ref_examples = sample(rNames, k = number_plot_fits)
            query_examples = sample(qNames, k = number_plot_fits)
            with stderr_redirected(): # Hide the many progress bars
                raw = pp_sketchlib.queryDatabase(ref_db_name=ref_db,
                                                query_db_name=query_db,
                                                rList=ref_examples,
                                                qList=query_examples,
                                                klist=klist,
                                                random_correct=False,
                                                jaccard=True,
                                                num_threads=threads,
                                                use_gpu = False)
                corrected = pp_sketchlib.queryDatabase(ref_db_name=ref_db,
                                                    query_db_name=query_db,
                                                    rList=ref_examples,
                                                    qList=query_examples,
                                                    klist=klist,
                                                    random_correct=True,
                                                    jaccard=True,
                                                    num_threads=threads,
                                                    use_gpu = False)
            for plot_idx in range(number_plot_fits):
                raw_fit = fitKmerCurve(raw[plot_idx], klist, jacobian)
                corrected_fit = fitKmerCurve(corrected[plot_idx], klist, jacobian)
                plot_fit(klist,
                          raw[plot_idx],
                          raw_fit,
                          corrected[plot_idx],
                          corrected_fit,
                          os.path.join(os.path.dirname(queryPrefix),
                                       os.path.basename(queryPrefix) + "_fit_example_" + str(plot_idx + 1)),
                          "Example fit " + str(plot_idx + 1) + " - " +  ref_examples[plot_idx] + \
                          " vs. " + query_examples[plot_idx])

    return distMat


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
                         "\nWith k-mer match values " +
                         np.array2string(pairwise, precision=4, separator=',',suppress_small=True) +
                         "\nCheck for low quality input genomes\n")
        transformed_params = [0, 0]

    # Return core, accessory
    return(np.flipud(transformed_params))

def get_database_statistics(prefix):
    """Extract statistics for evaluating databases.

    Args:
        prefix (str)
            Prefix of database
    """
    db_file = prefix + "/" + os.path.basename(prefix) + ".h5"
    ref_db = h5py.File(db_file, 'r')

    genome_lengths = []
    ambiguous_bases = []
    for sample_name in list(ref_db['sketches'].keys()):
        genome_lengths.append(ref_db['sketches/' + sample_name].attrs['length'])
        ambiguous_bases.append(ref_db['sketches/' + sample_name].attrs['missing_bases'])
        
    return genome_lengths, ambiguous_bases
