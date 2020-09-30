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
import graph_tool.all as gt
from multiprocessing import Pool, RawArray, shared_memory, managers
try:
    from multiprocessing import Pool, shared_memory
    from multiprocessing.managers import SharedMemoryManager
    NumpyShared = collections.namedtuple('NumpyShared', ('name', 'shape', 'dtype'))
except ImportError as e:
    sys.stderr.write("This version of PopPUNK requires python v3.8 or higher\n")
    sys.exit(0)
from functools import partial

import pp_sketchlib

# import poppunk package
from .plot import writeClusterCsv

from .utils import iterDistRows
from .utils import update_distance_matrices

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

def rank_distance_matrix(bounds, distances = None):
    """ Ranks distances between isolates for each index (row)
    isolate.

    Args:
        bounds (2-tuple)
            Range of rows to process in this thread.
        distances (ndarray in shared memory)
            Shared memory object storing pairwise distances.

    Returns:
        ranks (numpy ndarray)
            Ranks of distances for each row.
    """
    # load distance matrix from shared memory
    distances_shm = shared_memory.SharedMemory(name = distances.name)
    distances = np.ndarray(distances.shape, dtype = distances.dtype, buffer = distances_shm.buf)
    # rank relevant slide of distance matrix
    ranks = np.apply_along_axis(rankdata, 1, distances[slice(*bounds),:], method = 'ordinal')
    return ranks

def get_nearest_neighbours(rank, isolates = None, ranks = None):
    """ Identifies sets of nearest neighbours for each isolate.

    Args:
        rank (int)
            Rank used in analysis.
        isolates (int list)
            List of isolate indices.
        ranks (ndarray in shared memory)
            Shared memory object pointing to ndarray of
            ranked pairwise distances.

    Returns:
        nn (default dict of frozensets)
            Dict indexed by isolates, values are a
            frozen set of nearest neighbours.
    """
    # data structure
    nn = set()
    # load shared ranks
    ranks_shm = shared_memory.SharedMemory(name = ranks.name)
    ranks = np.ndarray(ranks.shape, dtype = ranks.dtype, buffer = ranks_shm.buf)
    # apply along axis
    for i in isolates:
        isolate_ranks = ranks[i,:]
        closest_ranked = np.ravel(np.where(isolate_ranks <= rank))
        for j in closest_ranked.tolist():
            nn.add((i,j))
    # return dict
    return nn


def pick_seed_isolate(G, distances = None):
    """ Identifies seed isolate from the closest pair of
    unclustered isolates.

    Args:
        G (network)
            Network with one node per isolate.
        distances (ndarray in shared memory)
            Pairwise distances between isolates.

    Returns:
        seed_isolate (int)
            Index of isolate selected as seed.
    """
    # load distances from shared memory
    distances_shm = shared_memory.SharedMemory(name = distances.name)
    distances = np.ndarray(distances.shape, dtype = distances.dtype, buffer = distances_shm.buf)
    # identify unclustered isolates
    unclustered_isolates = list(nx.isolates(G))
    # select minimum distance between unclustered isolates
    minimum_distance_between_unclustered_isolates = np.amin(distances[unclustered_isolates,unclustered_isolates],axis = 0)
    # select occurrences of this distance
    minimum_distance_coordinates = np.where(distances == minimum_distance_between_unclustered_isolates)
    # identify case where both isolates are unclustered
    for i in range(len(minimum_distance_coordinates[0])):
        if minimum_distance_coordinates[0][i] in unclustered_isolates and minimum_distance_coordinates[1][i] in unclustered_isolates:
            seed_isolate = minimum_distance_coordinates[0][i]
            break
    # return unclustered isolate with minimum distance to another isolate
    return seed_isolate

def get_lineage(G, neighbours, seed_isolate, lineage_index):
    """ Identifies isolates corresponding to a particular
    lineage given a cluster seed.

    Args:
        G (network)
            Network with one node per isolate.
        neighbours (dict of frozen sets)
           Pre-calculated neighbour relationships.
        seed_isolate (int)
           Index of isolate selected as seed.
        lineage_index (int)
           Label of current lineage.

    Returns:
        G (network)
            Network modified with new edges.
    """
    # initiate lineage as the seed isolate and immediate unclustered neighbours
    in_lineage = {seed_isolate}
    G.nodes[seed_isolate]['lineage'] = lineage_index
    for seed_neighbour in neighbours[seed_isolate]:
        if nx.is_isolate(G, seed_neighbour):
            G.add_edge(seed_isolate, seed_neighbour)
            G.nodes[seed_neighbour]['lineage'] = lineage_index
            in_lineage.add(seed_neighbour)
    # iterate through other isolates until converged on a stable clustering
    alterations = len(neighbours.keys())
    while alterations > 0:
        alterations = 0
        for isolate in neighbours.keys():
            if nx.is_isolate(G, isolate):
                intersection_size = in_lineage.intersection(neighbours[isolate])
                if intersection_size is not None and len(intersection_size) > 0:
                    for i in intersection_size:
                        G.add_edge(isolate, i)
                        G.nodes[isolate]['lineage'] = lineage_index
                    in_lineage.add(isolate)
                    alterations = alterations + 1
    # return final clustering
    return G

def cluster_into_lineages(distMat, rank_list = None, output = None,
    isolate_list = None, qlist = None, existing_scheme = None,
    use_accessory = False, num_processes = 1):
    """ Clusters isolates into lineages based on their
    relative distances.

    Args:
        distMat (np.array)
            n x 2 array of core and accessory distances for n samples.
            This should not be subsampled.
        rank_list (list of int)
            Integers specifying the maximum rank of neighbours used
            for clustering.
        output (string)
            Prefix used for printing output files.
        isolate_list (list)
            List of reference sequences.
        qlist (list)
            List of query sequences being added to an existing clustering.
            Should be included within isolate_list.
        existing_scheme (str)
            Path to pickle file containing lineage scheme to which isolates
            should be added.
        use_accessory (bool)
            Option to use accessory distances rather than core distances.
        num_processes (int)
            Number of CPUs to use for calculations.

    Returns:
        overall_lineages (nested dict)
            Dict for each rank listing the lineage of each isolate.
    """

    # data structures
    lineage_assignation = defaultdict(dict)
    overall_lineage_seeds = defaultdict(dict)
    overall_lineages = defaultdict(dict)
    max_existing_cluster = {rank:1 for rank in rank_list}
    
    # load existing scheme if supplied
    if existing_scheme is not None:
        with open(existing_scheme, 'rb') as pickle_file:
            lineage_assignation, overall_lineage_seeds, rank_list = pickle.load(pickle_file)
        for rank in rank_list:
            max_existing_cluster[rank] = max(lineage_assignation[rank].values()) + 1

    # generate square distance matrix
    seqLabels, coreMat, accMat = \
        update_distance_matrices(isolate_list, distMat, threads = num_processes)
    if use_accessory:
        distances = accMat
    else:
        distances = coreMat
    try:
        assert seqLabels == isolate_list
    except:
        sys.stderr.write('Isolates in wrong order?')
        exit(1)

    # list indices and set self-self to Inf
    isolate_indices = [n for n,i in enumerate(isolate_list)]
    for i in isolate_indices:
        distances[i,i] = np.Inf

    # get ranks of distances per row
    chunk_boundaries = get_chunk_ranges(distances.shape[0], num_processes)
    with SharedMemoryManager() as smm:

        # share isolate list
        isolate_list_shared = smm.ShareableList(isolate_indices)

        # create shared memory object for distances
        distances_raw = smm.SharedMemory(size = distances.nbytes)
        distances_shared_array = np.ndarray(distances.shape, dtype = distances.dtype, buffer = distances_raw.buf)
        distances_shared_array[:] = distances[:]
        distances_shared_array = NumpyShared(name = distances_raw.name, shape = distances.shape, dtype = distances.dtype)

        # parallelise ranking of distances across CPUs
        with Pool(processes = num_processes) as pool:
            ranked_array = pool.map(partial(rank_distance_matrix,
                                distances = distances_shared_array),
                                chunk_boundaries)

        # concatenate ranks into shared memory
        distance_ranks = np.concatenate(ranked_array)
        distance_ranks_raw = smm.SharedMemory(size = distance_ranks.nbytes)
        distance_ranks_shared_array = np.ndarray(distance_ranks.shape, dtype = distance_ranks.dtype, buffer = distance_ranks_raw.buf)
        distance_ranks_shared_array[:] = distance_ranks[:]
        distance_ranks_shared_array = NumpyShared(name = distance_ranks_raw.name, shape = distance_ranks.shape, dtype = distance_ranks.dtype)
        
        # build a graph framework for network outputs
        # create graph structure with internal vertex property map
        # storing lineage assignation cannot load boost.python within spawned
        # processes so have to run network analysis separately
        G = gt.Graph(directed = False)
        G.add_vertex(len(isolate_list))
        # add sequence labels for visualisation
        vid = G.new_vertex_property('string',
                                    vals = isolate_list)
        G.vp.id = vid
        
        # parallelise neighbour identification for each rank
        with Pool(processes = num_processes) as pool:
            results = pool.map(partial(get_nearest_neighbours,
                                ranks = distance_ranks_shared_array,
                                isolates = isolate_list_shared),
                                rank_list)

        # extract results from multiprocessing pool and save output network
        nn = defaultdict(dict)

        for n,result in enumerate(results):
            # get results per rank
            rank = rank_list[n]
            # get neigbours
            edges_to_add = result
            # store results in network
            G.add_edge_list(edges_to_add)
            # calculate connectivity of each vertex
            vertex_out_degrees = G.get_out_degrees(G.get_vertices())
            # identify components and rank by frequency
            components, component_frequencies = gt.label_components(G)
            component_frequency_ranks = (len(component_frequencies) - rankdata(component_frequencies, method = 'ordinal').astype(int)).tolist()
            # construct a name translation table
            # begin with previously defined clusters
            component_name = [None] * len(component_frequencies)
            for seed in overall_lineage_seeds[rank]:
                isolate_index = isolate_list.index(seed)
                component_number = components[isolate_index]
                if component_name[component_number] is None or component_name[component_number] > overall_lineage_seeds[rank][seed]:
                    component_name[component_number] = overall_lineage_seeds[rank][seed]
            # name remaining components in rank order
            for component_rank in range(len(component_frequency_ranks)):
#                
                component_number = component_frequency_ranks.index(component_rank)
                if component_name[component_number] is None:
                    component_name[component_number] = max_existing_cluster[rank]
                    # find seed isolate
                    component_max_degree = np.amax(vertex_out_degrees[np.where(components.a == component_number)])
                    seed_isolate_index = int(np.where((components.a == component_number) & (vertex_out_degrees == component_max_degree))[0][0])
                    seed_isolate = isolate_list[seed_isolate_index]
                    overall_lineage_seeds[rank][seed_isolate] = max_existing_cluster[rank]
                    # increment
                    max_existing_cluster[rank] = max_existing_cluster[rank] + 1
            # store assignments
            for isolate_index,isolate_name in enumerate(isolate_list):
                original_component = components.a[isolate_index]
                renamed_component = component_name[original_component]
                lineage_assignation[rank][isolate_name] = renamed_component
            # save network
            G.save(file_name = output + "/" + os.path.basename(output) + '_rank_' + str(rank) + '_lineages.gt', fmt = 'gt')
            # clear edges - nodes in graph can be reused but edges differ between ranks
            G.clear_edges()

    # store output
    with open(output + "/" + output + '_lineages.pkl', 'wb') as pickle_file:
        pickle.dump([lineage_assignation, overall_lineage_seeds, rank_list], pickle_file)
    
    # process multirank lineages
    overall_lineages = {}
    overall_lineages = {'Rank_' + str(rank):{} for rank in rank_list}
    overall_lineages['overall'] = {}
    for index,isolate in enumerate(isolate_list):
        overall_lineage = None
        for rank in rank_list:
            overall_lineages['Rank_' + str(rank)][isolate] = lineage_assignation[rank][isolate]
            if overall_lineage is None:
                overall_lineage = str(lineage_assignation[rank][isolate])
            else:
                overall_lineage = overall_lineage + '-' + str(lineage_assignation[rank][isolate])
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

def run_clustering_for_rank(rank, distances_input = None, distance_ranks_input = None, isolates = None, previous_seeds = None):
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
        lineage_assignation (dict)
            Assignment of each isolate to a cluster.
        lineage_seed (dict)
            Seed isolate used to initiate each cluster.
        connections (set of tuples)
            Edges to add to network describing lineages.
    """    

    # load shared memory objects
    distances_shm = shared_memory.SharedMemory(name = distances_input.name)
    distances = np.ndarray(distances_input.shape, dtype = distances_input.dtype, buffer = distances_shm.buf)
    distance_ranks_shm = shared_memory.SharedMemory(name = distance_ranks_input.name)
    distance_ranks = np.ndarray(distance_ranks_input.shape, dtype = distance_ranks_input.dtype, buffer = distance_ranks_shm.buf)
    isolate_list = isolates
    isolate_indices = range(0,len(isolate_list))

    # load previous scheme
    seeds = {}
    if previous_seeds is not None:
        seeds = previous_seeds[rank]

    # identify nearest neighbours
    nn = get_nearest_neighbours(rank,
                            ranks = distance_ranks_input,
                            isolates = isolate_list)

    # iteratively identify lineages
    lineage_index = 1
    connections = set()
    lineage_assignation = {isolate:None for isolate in isolate_list}
    
    while None in lineage_assignation.values():
        if lineage_index in seeds.keys():
            seed_isolate = seeds[lineage_index]
        else:
            seed_isolate = pick_seed_isolate(lineage_assignation, distances = distances_input)
        # skip over previously-defined seeds if amalgamated into different lineage now
        if lineage_assignation[seed_isolate] is None:
            seeds[lineage_index] = seed_isolate
            lineage_assignation, added_connections = get_lineage(lineage_assignation, nn, seed_isolate, lineage_index)
            connections.update(added_connections)
        lineage_index = lineage_index + 1
    
    # return clustering
    return lineage_assignation, seeds, nn, connections
