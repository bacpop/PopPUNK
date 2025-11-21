#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018-2023 John Lees and Nick Croucher

# universal
import os
import sys
# additional
import numpy as np
from collections import Counter

import poppunk_refine

from .network import prune_graph
from .utils import storePickle, iterDistRows, readIsolateTypeFromCsv

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
        # TODO this seems like it would be slow for a big dist matrix
        # TODO didn't we used to have this as a slice?
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
    
    else:
        newRefList = refList
        newDistMat = distMat
    
    storePickle(newRefList, newRefList, True, newDistMat, output)

    # return new distance matrix and sequence lists
    return newRefList, newDistMat

def prune_query_distance_matrix(refList, queryList, remove_seqs, qrDistMat,
                                queryAssign=None):
    """Remove chunks from the distance matrix which correspond to bad queries

    Args:
        refList (list)
            List of ref sequences used to generate distance matrix
        queryList (list)
            List of query sequences used to generate distance matrix
        remove_seqs_in (list)
            List of sequences to be omitted (must be from queries)
        qrDistMat (numpy.array)
            (r*q)x2 matrix of core distances (column 0) and accessory
            distances (column 1)
        queryAssign (numpy.array)
            Query assign results, which will also be pruned if passed
            [default = None]
    Returns:
        passing_queries (list)
            List of query sequences retained in distance matrix
        newqrDistMat (numpy.array)
            Updated version of qrDistMat
        queryAssign (numpy.array)
            Updated version of queryAssign
    """
    if remove_seqs.intersection(refList):
        raise RuntimeError("Trying to remove references")

    passing_queries = []
    pass_rows = []
    for name in queryList:
        if name not in remove_seqs:
            passing_queries.append(name)
            pass_rows += [True] * len(refList)
        else:
            pass_rows += [False] * len(refList)

    qrDistMat = qrDistMat[pass_rows, :]
    if queryAssign is not None:
        queryAssign = queryAssign[pass_rows]

    return passing_queries, qrDistMat, queryAssign

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

    # Make user aware of all filters being used (including defaults)
    sys.stderr.write("Running QC on sketches\n")
    if qc_dict['upper_n'] is not None:
        sys.stderr.write("Using count cutoff for ambiguous bases: " + str(qc_dict['upper_n']) + "\n")
    else:
        sys.stderr.write("Using proportion cutoff for ambiguous bases: " + str(qc_dict['prop_n']) + "\n")
    if qc_dict['length_range'][0] is None:
        sys.stderr.write("Using standard deviation for length cutoff: " + str(qc_dict['length_sigma']) + "\n")
    else:
        sys.stderr.write("Using range for length cutoffs: " + str(qc_dict['length_range'][0]) + " - " + \
                          str(qc_dict['length_range'][1]) + "\n")

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
    # Make user aware of all filters being used (including defaults)
    sys.stderr.write("Running QC on distances\n")
    sys.stderr.write("Using cutoff for core distances: " + str(qc_dict['max_pi_dist']) + "\n")
    sys.stderr.write("Using cutoff for accessory distances: " + str(qc_dict['max_a_dist']) + "\n")
    sys.stderr.write("Using cutoff for proportion of zero distances: " + str(qc_dict['prop_zero']) + "\n")

    # Create overall list of sequences
    if refList == queryList:
        names = refList
        self = True
    else:
        names = refList + queryList
        self = False

    # First check with numpy, which is quicker than iterating over everything
    #long_distance_rows = np.where([(distMat[:, 0] > qc_dict['max_pi_dist']) | (distMat[:, 1] > qc_dict['max_a_dist'])])[1].tolist()
    long_distance_rows = np.where([(distMat[:, 0] > qc_dict['max_pi_dist']) | (distMat[:, 1] > qc_dict['max_a_dist'])],0,1)[0].tolist()
    long_edges = poppunk_refine.generateTuples(long_distance_rows,
                                                0,
                                                self = self,
                                                num_ref = len(refList),
                                                int_offset = 0)

    failed = prune_edges(long_edges,
                                 query_start=len(refList),
                                 allow_ref_ref=self)
    # Convert the edge IDs back to sample names
    failed_samples = {names[x]: ["Failed distance QC (too high)"] for x in failed}

    # Check if too many zeros, basically the same way but update the existing
    # dicts/sets, and set a minimum count of zero lengths
    if qc_dict["prop_zero"] < 1:
        zero_count = round(qc_dict["prop_zero"] * len(names))
        zero_distance_rows = np.where([(distMat[:, 0] == 0) | (distMat[:, 1] == 0)],0,1)[0].tolist()
        zero_edges = poppunk_refine.generateTuples(zero_distance_rows,
                                                    0,
                                                    self = self,
                                                    num_ref = len(refList),
                                                    int_offset = 0)
        failed = prune_edges(zero_edges,
                            query_start=len(refList),
                            failed=failed,
                            min_count=zero_count,
                            allow_ref_ref=self)
        message = ["Failed distance QC (too many zeros)"]
        for sample in failed:
            name = names[sample]
            if name in failed_samples:
                failed_samples[name] += message
            else:
                failed_samples[name] = message

    retained_samples = [x for x in names if x not in frozenset(failed_samples.keys())]
    return retained_samples, failed_samples


def qcQueryAssignments(rList, qList, query_assignments, max_clusters,
                       original_cluster_file):
    """Checks assignments for too many links between clusters.

    Args:
        refList (list)
            Reference labels
        queryList (list)
            Query labels
        query_assignments (list or np.array)
            List of assignments for qrMat, where -1 is a link
        max_clusters (int)
            Maximum number of clusters which can be connected
        original_cluster_file (str)
            File to load original cluster definitions from

    Returns:
        retained (list)
            List of sequences passing QC filters
        failed (dict)
            List of sequences failing, and reasons
    """
    message = ["Failed graph QC (too many links)"]
    retained_samples = []
    failed_samples = {}

    # Read the rList cluster assignments, and turn into a dict which
    # is idx: cluster
    clusters = readIsolateTypeFromCsv(original_cluster_file, return_dict=True)
    clusters_idx = {idx: clusters['Cluster'][name] for idx, name in enumerate(rList)}

    # Find the edges for each query, and count how many unique clusters they
    # appear in
    for idx, query in enumerate(qList):
        row_start = idx * len(rList)
        row_end = (idx + 1) * len(rList)
        edges = np.argwhere(query_assignments[row_start:row_end] == -1).reshape(-1)
        cluster_links = set()
        for edge in edges:
            cluster_links.add(clusters_idx[edge])

        if len(cluster_links) > max_clusters:
            failed_samples[query] = message
        else:
            retained_samples.append(query)
    return retained_samples, failed_samples

def prune_edges(long_edges, query_start,
                failed=None, min_count=1, allow_ref_ref=True):
    """Gives a list of failed vertices from a list of failing
    edges. Tries to prune by those nodes with highest degree of
    bad nodes, preferentially removes queries.

    Args:
        long_edges (list of tuples)
            List of bad edges as node IDs
        query_start (int)
            The first node ID which corresponds to queries
        failed (set or None)
            If set, an existing prune list to add to
        min_count (int)
            Must be at least this many failures to prune
        allow_ref_ref (bool)
            Whether r-r edges can be pruned (set False if querying)
    Returns:
        failed (set)
            Failed sample IDs
    """
    if failed == None:
        failed = set()
    if len(long_edges) > 0:
        # Find nodes with the most bad edges
        counts = Counter()
        for (r, q) in long_edges:
            counts.update([r, q])

        # Sorts by edges which appear most often
        long_edges.sort(key=lambda x: max(counts[x[0]], counts[x[1]]), reverse=True)
        for (r, q) in long_edges:
            if q not in failed and r not in failed and (counts[r] >= min_count or counts[q] >= min_count):
                # Do not add any refs if querying
                if r < query_start and q < query_start:
                    if allow_ref_ref:
                        if (counts[r] > counts[q] and counts[r] >= min_count):
                            failed.add(r)
                        elif counts[q] >= min_count:
                            failed.add(q)
                # NB q > r
                elif r < query_start and q >= query_start:
                    failed.add(q)
                else:
                    if counts[r] > counts[q] and counts[r] >= min_count:
                        failed.add(r)
                    elif counts[q] >= min_count:
                        failed.add(q)

    return failed

def remove_qc_fail(qc_dict, names, passed, fail_dicts, ref_db, distMat, prefix,
                   strand_preserved=False, threads=1, use_gpu=False):
    """Removes samples failing QC from the database and distances. Also
    recalculates random match chances. Return a new distance matrix.

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
        use_gpu (bool)
            Whether GPU libraries were used to generate the original network.
    Return:
        newDistMat (numpy.array)
            Updated version of distMat
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
    if not qc_dict['no_remove']:
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
        newDistMat = prune_distance_matrix(names,
                              failed,
                              distMat,
                              f"{prefix}/{os.path.basename(prefix)}.dists")[1]
        
        # Update the graph
        prune_graph(ref_db,
                    names,
                    passed,
                    prefix,
                    threads,
                    use_gpu)
        
        #if any removed, recalculate random
        sys.stderr.write(f"Recalculating random matches with strand_preserved = {strand_preserved}\n")
        db_kmers = readDBParams(ref_db)[0]
        addRandom(prefix, passed, db_kmers, strand_preserved,
                    overwrite=True, threads=threads)

    # write failing & reasons
    write_qc_failure_report(failed, fail_dicts, prefix)

    return newDistMat

def write_qc_failure_report(failed_samples, fail_dicts, output_prefix):
    """
    Writes a report of failed samples and their reasons to a file.

    Parameters:
    - failed_samples: A list of samples that have failed.
    - fail_dicts: A list of dictionaries, each mapping samples to their failure reasons.
    - output_prefix: The prefix for the output file path.
    """
    # Accumulate output lines for each failed sample
    failed_output_lines = [
        f"{sample}\t{','.join(get_failure_reasons(sample, fail_dicts))}\n"
        for sample in failed_samples
    ]
    with open(f"{output_prefix}/{os.path.basename(output_prefix)}_qcreport.txt", 'w') as qc_file:
        qc_file.writelines(failed_output_lines)

def get_failure_reasons(sample, fail_dicts):
    """
    Retrieves all failure reasons for a given sample across multiple dictionaries.

    Parameters:
    - sample: The sample to retrieve failure reasons for.
    - fail_dicts: A list of dictionaries, each mapping samples to their failure reasons.

    Returns:
    A list of failure reasons for the given sample.
    """
    return [
        reason
        for fail_dict in fail_dicts
        if sample in fail_dict
        for reason in fail_dict[sample]
    ]
    
def pickTypeIsolate(prefix, refList):
    """Selects a type isolate as that with a minimal proportion
    of missing data.

    Args:
        prefix (str)
            Prefix for database
        refList (list)
            References to pick from

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
        for sample in refList:
            if sample in read_grp:
                if 'reads' in hdf_in['sketches'][sample].attrs and \
                    hdf_in['sketches'][sample].attrs['reads']:
                    sample_prop_n = 1.0
                else:
                    sample_prop_n = hdf_in['sketches'][sample].attrs['missing_bases']/hdf_in['sketches'][sample].attrs['length']
                if sample_prop_n < min_prop_n:
                    min_prop_n = sample_prop_n
                    type_isolate = sample
                if min_prop_n == 0.0:
                    break
    # if failure still close files to avoid corruption
    except:
        hdf_in.close()
        sys.stderr.write('Problem processing h5 databases during QC - aborting\n')
        print("Unexpected error:", sys.exc_info()[0], file = sys.stderr)
        raise

    return type_isolate

