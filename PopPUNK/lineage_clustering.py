#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018-2020 John Lees and Nick Croucher

# universal
import os
import sys
import re
# additional
import numpy as np
from scipy.stats import rankdata
from collections import defaultdict
import pickle
import collections
from multiprocessing import Pool, Lock, Manager, RawArray, shared_memory, managers
try:
    from multiprocessing import Pool, shared_memory
    from multiprocessing.managers import SharedMemoryManager
    NumpyShared = collections.namedtuple('NumpyShared', ('name', 'shape', 'dtype'))
except ImportError as e:
    sys.stderr.write("This version of PopPUNK requires python v3.8 or higher\n")
    sys.exit(0)
from functools import partial

# import poppunk package
from .plot import writeClusterCsv

from .utils import iterDistRows
from .utils import listDistInts
from .utils import readRfile

def run_lineage_clustering(lineage_clustering, neighbours, rank, lineage_index, seed_isolate, previous_lineage_clustering):
    """ Identifies isolates corresponding to a particular
    lineage given a cluster seed.
    
    Args:
        lineage_clustering (dict)
           Clustering of existing dataset.
        neighbours (nested dict)
           Pre-calculated neighbour relationships.
        rank (int)
           Maximum rank of neighbours used for clustering.
        lineage_index (int)
           Label of current lineage.
        seed_isolate (str)
           Isolate to used to initiate next lineage.
        previous_lineage_clustering (dict)
            Clustering of existing dataset in previous iteration.
        
    Returns:
        lineage_clustering (dict)
            Assignment of isolates to lineages.
    
    """
    # generate sets of neighbours based on rank
    neighbour_set = defaultdict(frozenset)
    for isolate in neighbours.keys():
        neighbour_set[isolate] = frozenset(neighbours[isolate].keys())
    # initiate lineage as the seed isolate and immediate unclustered neighbours
    in_lineage = {seed_isolate}
    for seed_neighbours in neighbour_set[seed_isolate]:
        if lineage_clustering[seed_neighbours] is None or lineage_clustering[seed_neighbours] > lineage_index:
            in_lineage.add(seed_neighbours)
            lineage_clustering[seed_neighbours] = lineage_index
    # iterate through other isolates until converged on a stable clustering
    alterations = len(neighbours.keys())
    while alterations > 0:
        alterations = 0
        for isolate in neighbour_set.keys():
            if lineage_clustering[isolate] is None or lineage_clustering[isolate] > lineage_index:
                intersection_size = in_lineage.intersection(neighbour_set[isolate])
                if intersection_size is not None and len(intersection_size) > 0:
                    in_lineage.add(isolate)
                    lineage_clustering[isolate] = lineage_index
                    alterations = alterations + 1
    # return final clustering
    return lineage_clustering

def get_seed_isolate(lineage_clustering, row_labels, distances, lineage_index, lineage_seed):
    """ Identifies the isolate used to initiate a cluster.

    Args:
        lineage_clustering (dict)
            Clustering of existing dataset.
        row_labels (list of tuples)
            Pairs of isolates labelling each distance.
        distances (numpy array)
            Pairwise distances used for defining relationships.
        lineage_index (int)
            Label of current lineage.
        lineage_seed (dict)
            Dict of seeds used to initiate pre-existing lineage definitions.

    Returns:
        seed_isolate (str)
            Isolate to used to initiate next lineage.

    """
    # variable to return
    seed_isolate = None
    # first test if there is an existing seed
    if lineage_index in lineage_seed.keys():
        original_seed_isolate = lineage_seed[lineage_index]
        # if seed now belongs to a different lineage, then lineage has been overwritten
        # seed may be unassigned if neighbours have changed - but this does not overwrite the
        # lineage in the same way
        if lineage_clustering[original_seed_isolate] is None or lineage_clustering[original_seed_isolate] == lineage_index:
            seed_isolate = original_seed_isolate
    # else identify a new seed from the closest pair
    else:
        for index,(distance,(isolate1,isolate2)) in enumerate(zip(distances,row_labels)):
            if lineage_clustering[isolate1] is None:
                seed_isolate = isolate1
                break
            elif lineage_clustering[isolate2] is None:
                seed_isolate = isolate2
                break
    # return identified seed isolate
    return seed_isolate

def generate_nearest_neighbours(distances, row_labels, isolate_indices, rank):
    # data structures
    nn = defaultdict(dict)
    last_dist = {}
    num_ranks = {}
    num_ranks = {i:0 for i in isolate_indices}
    total_isolates = len(isolate_indices)
    num_distances = len(distances)
    completed_isolates = 0
    index = 0
    # iterate through distances until all nearest neighbours identified
    while completed_isolates < total_isolates and index < num_distances:
        distance = distances[index]
        # iterate through all listed isolates
        for isolate in row_labels[index]:
            if isolate in num_ranks.keys() and num_ranks[isolate] < rank:
                # R is the number of unique distances so only increment if
                # different from the last distance
                if isolate in last_dist.keys() and last_dist[isolate] != distance:
                    num_ranks[isolate] = num_ranks[isolate] + 1
                # if maximum number of ranks reached, mark as complete
                if num_ranks[isolate] == rank: # stop at R as counting from 0
                    completed_isolates = completed_isolates + 1
                # if not, add as a nearest neighbour
                else:
                    pair = row_labels[index][0] if row_labels[index][1] == isolate else row_labels[index][1]
                    nn[isolate][pair] = distance
                last_dist[isolate] = distance
        index = index + 1
    # return completed dict
    return nn

def update_nearest_neighbours(distances, row_labels, rank, qlist, nn, lineage_clustering):
    """ Updates the information on nearest neighbours, given
    a new set of ref-query and query-query distances.

    Args:
       distances (numpy array)
        Distances to be used for defining lineages.
       row_labels (list of tuples)
        Pairs of isolates labelling each distance.
       rank (int)
        Maximum rank of distance used to define nearest neighbours.
       qlist (list)
        List of queries to be added to existing dataset.
       nn (nested dict)
        Pre-calculated neighbour relationships.
       lineage_clustering (dict)
        Clustering of existing dataset.

    Returns:
        nn (nested dict)
            Updated neighbour relationships.
        lineage_clustering (dict)
            Updated assignment of isolates to lineage.

    """
    # iterate through isolates and test whether any comparisons with
    # newly-added queries replace or supplement existing neighbours

    # data structures for altered entries
    nn_new = {}
    # pre-process to extract ref-query distances first
    query_match_indices = [n for n, (r, q) in enumerate(row_labels) if q in qlist or r in qlist]
    query_row_names = [row_labels[i] for i in query_match_indices]
    query_distances = np.take(distances, query_match_indices)
    
    # get nn for query sequences
    query_nn = generate_nearest_neighbours(distances, row_labels, qlist, rank)
    # add query-query comparisons
    for query in query_nn.keys():
        nn[query] = query_nn[query]
    
    # calculate max distances for each isolate
    max_distance = {}
    num_distances = {}
    for isolate in nn.keys():
        neighbour_distances = set(nn[isolate].values())
        max_distance[isolate] = max(neighbour_distances)
        num_distances[isolate] = len(neighbour_distances) # query-query comparisons may be < R
        
    # iterate through the ref-query distances
    for index,(distance,(isolate1,isolate2)) in enumerate(zip(query_distances,query_row_names)):
        # identify ref-query matches
        ref = None
        query = None
        if isolate1 in max_distance.keys() and isolate2 not in max_distance.keys():
            ref = isolate1
            query = isolate2
        elif isolate2 in max_distance.keys() and isolate1 not in max_distance.keys():
            ref = isolate2
            query = isolate1
        if ref is not None:
            if distance <= max_distance[ref]:
                # unset isolate and references
                lineage_clustering[ref] = None
                for neighbour in nn[ref]:
                    lineage_clustering[neighbour] = None
                # add neighbours
                nn[ref][query] = distance
                # delete links if no longer high ranked match
                if distance < max_distance[ref]:
                    if num_distances[ref] == rank:
                        to_delete = []
                        for other in nn[ref].keys():
                            if nn[ref][other] == max_distance[ref]:
                                to_delete.append(other)
                        for other in to_delete:
                            del nn[ref][other]
                    else:
                        # if set from query-query distances < R
                        num_distances[ref] = num_distances[ref] + 1
                    max_distance[ref] = max(nn[ref].values())
    # return updated dict
    return nn, lineage_clustering

def cluster_into_lineages(distMat, rank_list = None, output = None, rlist = None, qlist = None, existing_scheme = None, use_accessory = False, num_processes = 1):
    """ Clusters isolates into lineages based on their
    relative distances.
    
    Args:
        distMat (np.array)
            n x 2 array of core and accessory distances for n samples.
            This should not be subsampled.
        rank_list (list of int)
            Integers specifying the maximum rank of neighbours used
            for clustering.
        rlist (list)
            List of reference sequences.
        qlist (list)
            List of query sequences being added to an existing clustering.
            Should be included within rlist.
        existing_scheme (str)
            Path to pickle file containing lineage scheme to which isolates
            should be added.
        use_accessory (bool)
            Option to use accessory distances rather than core distances.
        num_processes (int)
            Number of CPUs to use for calculations.
            
    Returns:
        combined (dict)
            Assignment of each isolate to clusters by all ranks used.
    """
    
    # process distance matrix
    # - this should be sorted (PyTorch allows this to be done on GPUs)
    # - then the functions can be reimplemented to run faster on a
    #   sorted list
    distance_index = 1 if use_accessory else 0
    distances = distMat[:,distance_index]
    
    # sort distances
    distance_ranks = np.argsort(distances)
    distances = distances[distance_ranks]
    
    # determine whether ref-ref or ref-query analysis
    isolate_list = rlist
    
    # determine whether novel analysis or modifying existing analysis
    use_existing = False
    neighbours = {}
    lineage_seed = {}
    lineage_clustering = {}
    previous_lineage_clustering = {}
    
    for rank in rank_list:
        lineage_clustering[rank] = {i:None for i in range(0,len(isolate_list))}
        lineage_seed[rank] = {}
        neighbours[rank] = {}
        previous_lineage_clustering[rank] = {}
    
    # shared memory data structures
    with SharedMemoryManager() as smm:
        # share sorted distances
        distances_raw = smm.SharedMemory(size = distances.nbytes)
        distances_shared_array = np.ndarray(distances.shape, dtype = distances.dtype, buffer = distances_raw.buf)
        distances_shared_array[:] = distances[:]
        distances_shared_array = NumpyShared(name = distances_raw.name, shape = distances.shape, dtype = distances.dtype)

        # share distance ranks
        distance_ranks_raw = smm.SharedMemory(size = distance_ranks.nbytes)
        distance_ranks_shared_array = np.ndarray(distance_ranks.shape, dtype = distance_ranks.dtype, buffer = distance_ranks_raw.buf)
        distance_ranks_shared_array[:] = distance_ranks[:]
        distance_ranks_shared_array = NumpyShared(name = distance_ranks_raw.name, shape = distance_ranks.shape, dtype = distance_ranks.dtype)

        # share isolate list
        isolate_list_shared = smm.ShareableList(isolate_list)
    
        # run clustering for an individual R
        with Pool(processes = num_processes) as pool:
            results = pool.map(partial(run_clustering_for_rank,
                                qlist = qlist,
                                existing_scheme = existing_scheme,
                                distances = distances_shared_array,
                                distance_ranks = distance_ranks_shared_array,
                                isolates = isolate_list_shared),
                                rank_list)
        
        # extract results from multiprocessing pool
        for n,result in enumerate(results):
            rank = rank_list[n]
            lineage_clustering[rank], lineage_seed[rank], neighbours[rank], previous_lineage_clustering[rank] = result

    # store output
    with open(output + "/" + output + '_lineages.pkl', 'wb') as pickle_file:
        pickle.dump([lineage_clustering, lineage_seed, neighbours, rank_list], pickle_file)
    
    # process multirank lineages
    overall_lineages = {}
    overall_lineages = {'Rank_' + str(rank):{} for rank in rank_list}
    overall_lineages['overall'] = {}
    for index,isolate in enumerate(isolate_list):
        overall_lineage = None
        for rank in rank_list:
            overall_lineages['Rank_' + str(rank)][isolate] = lineage_clustering[rank][index]
            if overall_lineage is None:
                overall_lineage = str(lineage_clustering[rank][index])
            else:
                overall_lineage = overall_lineage + '-' + str(lineage_clustering[rank][index])
        overall_lineages['overall'][isolate] = overall_lineage
    
    # print output as CSV
    writeClusterCsv(output + "/" + output + '_lineages.csv',
                    isolate_list,
                    isolate_list,
                    overall_lineages,
                    output_format = 'phandango',
                    epiCsv = None,
                    queryNames = qlist,
                    suffix = '_Lineage')

    # return lineages
    return overall_lineages

def run_clustering_for_rank(rank, qlist = None, existing_scheme = False, distances = None, distance_ranks = None, isolates = None):
    """ Clusters isolates into lineages based on their
    relative distances using a single R to enable
    parallelisation.

    Args:
        rank (int)
            Integer specifying the maximum rank of neighbour used
            for clustering. Should be changed to int list for hierarchical
            clustering.
        qlist (list)
            List of query sequences being added to an existing clustering.
            Should be included within rlist.
        use_existing (bool)
            Whether to extend a previously generated analysis or not.
            
    Returns:
        lineage_clustering (dict)
            Assignment of each isolate to a cluster.
        lineage_seed (dict)
            Seed isolate used to initiate each cluster.
        neighbours (nested dict)
            Neighbour relationships between isolates for R.
    """
    
    # load shared memory objects
    distances_shm = shared_memory.SharedMemory(name = distances.name)
    distances = np.ndarray(distances.shape, dtype = distances.dtype, buffer = distances_shm.buf)
    distance_ranks_shm = shared_memory.SharedMemory(name = distance_ranks.name)
    distance_ranks = np.ndarray(distance_ranks.shape, dtype = distance_ranks.dtype, buffer = distance_ranks_shm.buf)
    isolate_list = isolates
    isolate_indices = range(0,len(isolate_list))
    
    # calculate row labels
    # this is inefficient but there appears to be no way of sharing
    # strings between processes efficiently
    row_labels = listDistInts(isolate_list, isolate_list, self = True)
    # reorder by sorted distances
    row_labels = [row_labels[i] for i in distance_ranks]
    
    lineage_clustering = {i:None for i in range(0,len(isolate_list))}
    previous_lineage_clustering = lineage_clustering
    lineage_seed = {}
    neighbours = {}
    
    if existing_scheme is not None:
        with open(existing_scheme, 'rb') as pickle_file:
            lineage_clustering_overall, lineage_seed_overall, neighbours_overall, rank_list = pickle.load(pickle_file)
        # focus on relevant data
        lineage_clustering = lineage_clustering_overall[rank]
        lineage_seed = lineage_seed_overall[rank]
        neighbours = neighbours_overall[rank]
        # add new queries to lineage clustering
        q_indices = [isolate_list.index(q) for q in qlist]
        for q in q_indices:
            lineage_clustering[q] = None
        previous_lineage_clustering = lineage_clustering
        
        neighbours, lineage_clustering = update_nearest_neighbours(distances,
                                                                row_labels,
                                                                rank,
                                                                q_indices,
                                                                neighbours,
                                                                lineage_clustering)
    else:
        neighbours = generate_nearest_neighbours(distances,
                                                row_labels,
                                                isolate_indices,
                                                rank)

    # run clustering
    lineage_index = 1
    while None in lineage_clustering.values():

        # get seed isolate based on minimum pairwise distances
        seed_isolate = get_seed_isolate(lineage_clustering,
                                        row_labels,
                                        distances,
                                        lineage_index,
                                        lineage_seed)
        lineage_seed[lineage_index] = seed_isolate
        
        # seed isolate is None if a previously-existing cluster has been overwritten
        # in which case pass over the lineage to keep nomenclature consistent
        if seed_isolate is not None:
        
            # record status of seed isolate
            lineage_clustering[seed_isolate] = lineage_index
                                                                                            
            # cluster the lineages
            lineage_clustering = run_lineage_clustering(lineage_clustering,
                                                        neighbours,
                                                        rank,
                                                        lineage_index,
                                                        seed_isolate,
                                                        previous_lineage_clustering)
                                                        
        # increment index for next lineage
        lineage_index = lineage_index + 1

    # return clustering
    return lineage_clustering, lineage_seed, neighbours, previous_lineage_clustering
