#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018-2022 John Lees and Nick Croucher

# universal
import os
import sys
# additional
import numpy as np

import poppunk_refine

from .utils import storePickle
from .utils import iterDistRows

def prune_distance_matrix(refList, remove_seqs_in, distMat, output):
    """Rebuild distance matrix following selection of panel of references

    Args:
        refList (list)
            List of sequences used to generate distance matrix
        remove_seqs_in (list)
            List of sequences to be omitted
        distMat (numpy.array)
            nx2 matrix of core distances (column 0) and accessory
            distances (column 1)
        output (string)
            Prefix for new distance output files
    Returns:
        newRefList (list)
            List of sequences retained in distance matrix
        newDistMat (numpy.array)
            Updated version of distMat
    """
    # Find list items to remove
    remove_seqs_list = []
    removal_indices = []
    for to_remove in remove_seqs_in:
        found = False
        for idx, item in enumerate(refList):
            if item == to_remove:
                removal_indices.append(idx)
                remove_seqs_list.append(item)
                found = True
                break
        if not found:
            sys.stderr.write("Couldn't find " + to_remove + " in database\n")
    remove_seqs = frozenset(remove_seqs_list)

    if len(remove_seqs) > 0:
        sys.stderr.write("Removing " + str(len(remove_seqs)) + " sequences\n")

        numNew = len(refList) - len(remove_seqs)
        newDistMat = np.zeros((int(0.5 * numNew * (numNew - 1)), 2), dtype=distMat.dtype)

        # Create new reference list iterator
        removal_indices.sort()
        removal_indices.reverse()
        next_remove = removal_indices.pop()
        newRefList = []
        for idx, seq in enumerate(refList):
            if idx == next_remove:
                if len(removal_indices) > 0:
                    next_remove = removal_indices.pop()
            else:
                newRefList.append(seq)

        newRowNames = iter(iterDistRows(newRefList, newRefList, self=True))

        # Copy over rows which don't have an excluded sequence
        newIdx = 0
        for distRow, (ref1, ref2) in zip(distMat, iterDistRows(refList, refList, self=True)):
            if ref1 not in remove_seqs and ref2 not in remove_seqs:
                (newRef1, newRef2) = next(newRowNames)
                if newRef1 == ref1 and newRef2 == ref2:
                    newDistMat[newIdx, :] = distRow
                    newIdx += 1
                else:
                    raise RuntimeError("Row name mismatch. Old: " + ref1 + "," + ref2 + "\n"
                                       "New: " + newRef1 + "," + newRef2 + "\n")

        storePickle(newRefList, newRefList, True, newDistMat, output)
    else:
        newRefList = refList
        newDistMat = distMat

    # return new distance matrix and sequence lists
    return newRefList, newDistMat


def sketchlibAssemblyQC(prefix, names, qc_dict):
    """Calculates random match probability based on means of genomes
    in assemblyList, and looks for length outliers.

    Args:
        prefix (str)
            Prefix of output files
        names (list)
            Names of samples to QC
        qc_dict (dict)
            Dictionary of QC parameters

    Returns:
        retained (list)
            List of sequences passing QC filters
        failed (dict)
            List of sequences failing, and reasons
    """
    import h5py
    from .sketchlib import removeFromDB

    sys.stderr.write("Running QC on sketches\n")

    # open databases
    db_name = prefix + '/' + os.path.basename(prefix) + '.h5'
    hdf_in = h5py.File(db_name, 'r')

    # try/except structure to prevent h5 corruption
    failed_samples = {}
    try:
        # process data structures
        read_grp = hdf_in['sketches']

        seq_length = {}
        seq_ambiguous = {}
        # iterate through sketches
        for dataset in read_grp:
            if dataset in names:
                # test thresholds
                seq_length[dataset] = hdf_in['sketches'][dataset].attrs['length']
                # If reads, do not QC based on Ns (simpler this way)
                # Older versions of DB do not save reads, so attr may not be present
                if 'reads' in hdf_in['sketches'][dataset].attrs and \
                    hdf_in['sketches'][dataset].attrs['reads']:
                    seq_ambiguous[dataset] = 0
                else:
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
        for dataset in seq_length.keys():
            # determine if sequence passes filters
            if seq_length[dataset] < lower_length:
                failed_samples[dataset] = ['Below lower length threshold']
            elif seq_length[dataset] > upper_length:
                failed_samples[dataset] = ['Above upper length threshold']
            if (qc_dict['upper_n'] is not None and seq_ambiguous[dataset] > qc_dict['upper_n']) or \
                (seq_ambiguous[dataset] > qc_dict['prop_n'] * seq_length[dataset]):
                message = "Ambiguous sequence too high"
                if dataset in failed_samples:
                    failed_samples[dataset].append(message)
                else:
                    failed_samples[dataset] = [message]

        hdf_in.close()
    # if failure still close files to avoid corruption
    except:
        hdf_in.close()
        sys.stderr.write('Problem processing h5 databases during QC - aborting\n')

        print("Unexpected error:", sys.exc_info()[0], file = sys.stderr)
        raise

    # This gives back retained in the same order as names
    retained_samples = [x for x in names if x not in frozenset(failed_samples.keys())]

    return retained_samples, failed_samples


def qcDistMat(distMat, refList, queryList, ref_db, qc_dict):
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
        qc_dict (dict)
            Dict of QC options

    Returns:
        retained (list)
            List of sequences passing QC filters
        failed (dict)
            List of sequences failing, and reasons
    """
    # Create overall list of sequences
    if refList == queryList:
        names = refList
    else:
        names = refList + queryList

    # Pick type isolate if not supplied
    if qc_dict['type_isolate'] is None:
        qc_dict['type_isolate'] = pickTypeIsolate(ref_db)
        sys.stderr.write('Selected type isolate for distance QC is ' + qc_dict['type_isolate'] + '\n')

    # First check with numpy, which is quicker than iterating over everything
    #long_distance_rows = np.where([(distMat[:, 0] > qc_dict['max_pi_dist']) | (distMat[:, 1] > qc_dict['max_a_dist'])])[1].tolist()
    long_distance_rows = np.where([(distMat[:, 0] > qc_dict['max_pi_dist']) | (distMat[:, 1] > qc_dict['max_a_dist'])],0,1)[0].tolist()
    long_edges = poppunk_refine.generateTuples(long_distance_rows,
                                                0,
                                                self = (refList == queryList),
                                                num_ref = len(refList),
                                                int_offset = 0)
    failed_samples = {}
    message = ["Failed distance QC"]
    if len(long_edges) > 0:
        # Prune sequences based on reference sequence
        for (s,t) in long_edges:
            if names[s] == qc_dict['type_isolate']:
                if names[t] not in failed_samples:
                    failed_samples[names[t]] = message
            elif names[s] not in failed_samples:
                failed_samples[names[s]] = message

    retained_samples = [x for x in names if x not in frozenset(failed_samples.keys())]

    return retained_samples, failed_samples

def remove_qc_fail(qc_dict, names, passed, fail_dicts, ref_db, distMat, prefix,
                   strand_preserved=False, threads=1):
    """Removes samples failing QC from the database and distances. Also
    recalculates random match chances.

    Args:
        qc_dict (dict)
            Dictionary of QC options
        names (list)
            The names of samples in the distMat
        passed (list)
            The names of passing samples
        fail_dicts (list)
            A list of dictionaries, which each have keys of samples,
            and values with lists of reasons
        ref_db (str)
            Prefix for the database
        distMat (numpy.array)
            The distance matrix to prune
        prefix (str)
            The prefix to save the new db and distMat to
        strand_preserved (bool)
            Whether to use a preserved strand when recalculating random match
            chances [default = False].
        threads (int)
            Number of CPU threads to use when recalculating random match chances
            [default = 1].
    """
    from .sketchlib import removeFromDB, addRandom, readDBParams

    # Create output directory if it does not exist already
    if not os.path.isdir(prefix):
        try:
            os.makedirs(prefix)
        except OSError:
            sys.stderr.write("Cannot create output directory " + prefix + "\n")
            sys.exit(1)

    failed = set(names) - set(passed)
    if qc_dict['retain_failures']:
        removeFromDB(ref_db,
                    f"{prefix}/failed.{os.path.basename(prefix)}.h5",
                    passed,
                    full_names = True)
    # new database file if pruning
    if failed and not qc_dict['no_remove']:
        # sys.stderr.write(f"Removing {len(failed)} samples from database and distances\n")
        tmp_filtered_db_name = f"{prefix}/filtered.{os.path.basename(prefix)}.h5"
        output_db_name = f"{prefix}/{os.path.basename(prefix)}.h5"
        input_db_name = f"{ref_db}/{os.path.basename(ref_db)}.h5"
        removeFromDB(input_db_name,
                     tmp_filtered_db_name,
                     failed,
                     full_names = True)
        os.rename(tmp_filtered_db_name, output_db_name)

        # Remove from the distMat too
        prune_distance_matrix(names,
                              failed,
                              distMat,
                              f"{prefix}/{os.path.basename(prefix)}.dists")

        #if any removed, recalculate random
        sys.stderr.write(f"Recalculating random matches with strand_preserved = {strand_preserved}\n")
        db_kmers = readDBParams(ref_db)[0]
        addRandom(prefix, passed, db_kmers, strand_preserved,
                    overwrite=True, threads=threads)

    # write failing & reasons
    with open(f"{prefix}/{os.path.basename(prefix)}_qcreport.txt", 'w') as qc_file:
        for sample in failed:
            reasons = []
            for fail_test in fail_dicts:
                if sample in fail_test:
                    reasons += (fail_test[sample])
            qc_file.write(f"{sample}\t{','.join(reasons)}\n")

def pickTypeIsolate(prefix):
    """Selects a type isolate as that with a minimal proportion
    of missing data.

    Args:
        prefix (str)
            Prefix of output files

    Returns:
        type_isolate (str)
            Name of isolate selected as reference
    """
    # open databases
    import h5py
    db_name = prefix + '/' + os.path.basename(prefix) + '.h5'
    hdf_in = h5py.File(db_name, 'r')

    min_prop_n = 1.0
    type_isolate = None

    try:
        # process data structures
        read_grp = hdf_in['sketches']
        # iterate through sketches
        for dataset in read_grp:
            if 'reads' in hdf_in['sketches'][dataset].attrs and \
                hdf_in['sketches'][dataset].attrs['reads']:
                sample_prop_n = 1.0
            else:
                sample_prop_n = hdf_in['sketches'][dataset].attrs['missing_bases']/hdf_in['sketches'][dataset].attrs['length']
            if sample_prop_n < min_prop_n:
                min_prop_n = sample_prop_n
                type_isolate = dataset
            if min_prop_n == 0.0:
                break
    # if failure still close files to avoid corruption
    except:
        hdf_in.close()
        sys.stderr.write('Problem processing h5 databases during QC - aborting\n')
        print("Unexpected error:", sys.exc_info()[0], file = sys.stderr)
        raise

    return type_isolate

