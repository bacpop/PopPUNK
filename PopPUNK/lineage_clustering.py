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
from multiprocessing import Pool, Lock, Manager, RawArray, shared_memory, managers
from functools import partial

# import poppunk package
from .utils import iterDistRows
from .utils import readRfile

def run_lineage_clustering(lineage_clustering, lineage_clustering_information, neighbours, R, lineage_index, seed_isolate, previous_lineage_clustering, null_cluster_value):
    """ Identifies isolates corresponding to a particular
    lineage given a cluster seed.
    
    Args:
        lineage_clustering (dict)
           Clustering of existing dataset.
        lineage_clustering_information (dict)
            Dict listing isolates by ranked distance from seed.
        neighbours (nested dict)
           Pre-calculated neighbour relationships.
        R (int)
           Maximum rank of neighbours used for clustering.
        lineage_index (int)
           Label of current lineage.
        seed_isolate (str)
           Isolate to used to initiate next lineage.
        previous_lineage_clustering (dict)
            Clustering of existing dataset in previous iteration.
        null_cluster_value (int)
            Null cluster value used for unsetting lineage assignments
            where this may change due to altered neighbour relationships.
        
    Returns:
        lineage_clustering (dict)
            Assignment of isolates to lineages.
    
    """
    # first make all R neighbours of the seed part of the lineage if unclustered
    for seed_neighbour in neighbours[seed_isolate]:
        if lineage_clustering[seed_neighbour] > lineage_index:
            lineage_clustering[seed_neighbour] = lineage_index
    # iterate through ranks; for each isolate, test if neighbour belongs to this cluster
    # overwrite higher cluster values - default value is higer than number of isolates
    # when querying, allows smaller clusters to be merged into more basal ones as connections
    # made
    for rank in lineage_clustering_information.keys():
        # iterate through isolates of same rank
        for isolate in lineage_clustering_information[rank]:
            # test if clustered/belonging to a more peripheral cluster
            if lineage_clustering[isolate] > lineage_index:
                # get clusters of nearest neighbours
                isolate_neighbour_clusters = [lineage_clustering[isolate_neighbour] for isolate_neighbour in neighbours[isolate].keys()]
                # if a nearest neighbour is in this cluster
                if lineage_index in isolate_neighbour_clusters:
                    # add isolate to lineage
                    lineage_clustering[isolate] = lineage_index
                    # add neighbours of same or lower rank to lineage if unclustered
                    for isolate_neighbour in neighbours[isolate].keys():
                        if lineage_clustering[isolate_neighbour] > lineage_index:
                            for neighbour_rank in lineage_clustering_information.keys():
                                if neighbour_rank <= rank:
                                    if isolate_neighbour in lineage_clustering_information[neighbour_rank]:
                                        lineage_clustering[isolate_neighbour] = lineage_index
                                else:
                                    break
            # if this represents a change from the previous iteration of lineage definitions
            # then the change needs to propagate through higher-ranked members
            if isolate in previous_lineage_clustering:
                if previous_lineage_clustering[isolate] == lineage_index and lineage_clustering[isolate] != lineage_index:
                    for higher_rank in lineage_clustering_information.keys():
                        if higher_rank > rank:
                            for higher_ranked_isolate in lineage_clustering_information[rank]:
                                if lineage_clustering[isolate] == lineage_index:
                                    lineage_clustering[isolate] = null_cluster_value
                
                        
    return lineage_clustering

def get_seed_isolate(lineage_clustering, row_labels, distances, null_cluster_value, lineage_index, lineage_seed):
    """ Identifies the isolate used to initiate a cluster.

    Args:
        lineage_clustering (dict)
            Clustering of existing dataset.
        row_labels (list of tuples)
            Pairs of isolates labelling each distance.
        distances (numpy array)
            Pairwise distances used for defining relationships.
        null_cluster_value (int)
           Null cluster value used for unsetting lineage assignments
           where this may change due to altered neighbour relationships.
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
        if lineage_clustering[original_seed_isolate] == null_cluster_value or lineage_clustering[original_seed_isolate] == lineage_index:
            seed_isolate = original_seed_isolate
    # else identify a new seed from the closest pair
    else:
        for index,distance in enumerate(distances):
            for isolate in row_labels[index]:
                if lineage_clustering[isolate] == null_cluster_value:
                    seed_isolate = isolate
                break
    # return identified seed isolate
    return seed_isolate

def get_lineage_clustering_information(seed_isolate, row_labels, distances):
    """ Generates the ranked distances needed for cluster
    definition.
    
    Args:
        seed_isolate (str)
            Isolate used to initiate lineage.
        row_labels (list of tuples)
            Pairs of isolates labelling each distance.
        distances (numpy array)
            Pairwise distances used for defining relationships.
    
    Returns:
        lineage_info (dict)
            Dict listing isolates by ranked distance from seed.
    
    """
    # data structure
    lineage_info = defaultdict(list)
    rank = 0
    last_dist = -1
    # iterate through distances recording rank
    for index,distance in enumerate(distances):
        if seed_isolate in row_labels[index]:
            if distance > last_dist:
                rank = rank + 1
                last_dist = distance
            pair = row_labels[index][0] if row_labels[index][1] == seed_isolate else row_labels[index][1]
            lineage_info[rank].append(pair)
    # return information
    return lineage_info

#def old_get_lineage_clustering_information(seed_isolate, row_labels, distances):
#    """ Generates the ranked distances needed for cluster
#    definition.
#
#    Args:
#        seed_isolate (str)
#            Isolate used to initiate lineage.
#        row_labels (list of tuples)
#            Pairs of isolates labelling each distance.
#        distances (numpy array)
#            Pairwise distances used for defining relationships.
#
#    Returns:
#        lineage_info (dict)
#            Dict listing isolates by ranked distance from seed.
#
#    """
#    # data structure
#    lineage_info = defaultdict(list)
#    # get subset of relevant distances
#    comparisons_involving_seed = [n for n,(pair) in enumerate(row_labels) if seed_isolate in pair]
#    distances_to_seed = distances[comparisons_involving_seed]
#    # get ranks of data
#    distance_ranks = rankdata(distances_to_seed)
#    # get partners of seed isolate in each comparison
#    pairs_involving_seed = [row_labels[x] for x in comparisons_involving_seed]
#    seed_partners = [r if q == seed_isolate else q for (r,q) in pairs_involving_seed]
#    # create a dict of lists of isolates for a given rank
#    # enables later easy iterating through ranked distances
#    for rank in np.unique(distance_ranks):
#        lineage_info[rank] = [seed_partners[n] for n,r in enumerate(distance_ranks) if r == rank]
#        # debug
#        if seed_isolate in {'EPI_ISL_416602','EPI_ISL_416650','EPI_ISL_417950','EPI_ISL_417931','EPI_ISL_416649','EPI_ISL_417975','EPI_ISL_413593','EPI_ISL_416633','EPI_ISL_416643','EPI_ISL_413592','EPI_ISL_416653'} and rank < 10:
#            print('Rank: ' + str(rank) + '\tpair: ' + str(lineage_info[rank]) + '\tseed: ' + seed_isolate)
#    return lineage_info

def generate_nearest_neighbours(distances, row_labels, isolate_list, R):
    # clade
    in_lineage = {'EPI_ISL_419768','EPI_ISL_420716','EPI_ISL_417108','EPI_ISL_418430','EPI_ISL_417105','EPI_ISL_417102','EPI_ISL_419747','EPI_ISL_419749','EPI_ISL_419762','EPI_ISL_418406','EPI_ISL_421207','EPI_ISL_419733','EPI_ISL_417100','EPI_ISL_419744','EPI_ISL_418408','EPI_ISL_418416','EPI_ISL_417104','EPI_ISL_420721','EPI_ISL_418418','EPI_ISL_420717','EPI_ISL_419739','EPI_ISL_414045','EPI_ISL_419731','EPI_ISL_419729','EPI_ISL_419741','EPI_ISL_417107','EPI_ISL_419748','EPI_ISL_420727','EPI_ISL_420750','EPI_ISL_418405','EPI_ISL_418412','EPI_ISL_419765','EPI_ISL_418413','EPI_ISL_420723','EPI_ISL_420742','EPI_ISL_419725','EPI_ISL_419763','EPI_ISL_418432','EPI_ISL_418431','EPI_ISL_420745','EPI_ISL_418403','EPI_ISL_420751','EPI_ISL_419760','EPI_ISL_418409','EPI_ISL_418419','EPI_ISL_418414','EPI_ISL_419743','EPI_ISL_418400','EPI_ISL_419745','EPI_ISL_420722','EPI_ISL_418402','EPI_ISL_418417','EPI_ISL_419746','EPI_ISL_420724','EPI_ISL_420746','EPI_ISL_419761','EPI_ISL_419764','EPI_ISL_417101','EPI_ISL_420714','EPI_ISL_420719','EPI_ISL_419740','EPI_ISL_417148','EPI_ISL_420748','EPI_ISL_421212','EPI_ISL_419767','EPI_ISL_420747','EPI_ISL_419783','EPI_ISL_419726','EPI_ISL_419720','EPI_ISL_416691','EPI_ISL_419713','EPI_ISL_416693','EPI_ISL_416698','EPI_ISL_417197','EPI_ISL_421213','EPI_ISL_418404','EPI_ISL_419730','EPI_ISL_417122','EPI_ISL_420720','EPI_ISL_420785','EPI_ISL_421194','EPI_ISL_418426','EPI_ISL_417988','EPI_ISL_416656','EPI_ISL_416636','EPI_ISL_413574','EPI_ISL_421191','EPI_ISL_416631','EPI_ISL_416603','EPI_ISL_419766','EPI_ISL_420729','EPI_ISL_417103','EPI_ISL_420718','EPI_ISL_419728','EPI_ISL_417192','EPI_ISL_416685','EPI_ISL_419719','EPI_ISL_416658','EPI_ISL_420749','EPI_ISL_421210','EPI_ISL_419717','EPI_ISL_416686','EPI_ISL_419723','EPI_ISL_413572','EPI_ISL_414027','EPI_ISL_414023','EPI_ISL_419727','EPI_ISL_414020','EPI_ISL_416680','EPI_ISL_417196','EPI_ISL_421203','EPI_ISL_419708','EPI_ISL_419712','EPI_ISL_419710','EPI_ISL_419716','EPI_ISL_416626','EPI_ISL_416625','EPI_ISL_417121','EPI_ISL_416688','EPI_ISL_420715','EPI_ISL_419736','EPI_ISL_418401','EPI_ISL_417987','EPI_ISL_419737','EPI_ISL_420786','EPI_ISL_417109','EPI_ISL_419718','EPI_ISL_416682','EPI_ISL_417194','EPI_ISL_419769','EPI_ISL_417106','EPI_ISL_416690','EPI_ISL_421204','EPI_ISL_419711','EPI_ISL_419714','EPI_ISL_420728'}
    # data structures
    nn = {}
    last_dist = {}
    num_ranks = {}
    for i in isolate_list:
        nn[i] = {}
        num_ranks[i] = 0
    total_isolates = len(isolate_list)
    completed_isolates = 0
    index = 0
    # iterate through distances until all nearest neighbours identified
    while completed_isolates < total_isolates:
        try:
            distance = distances[index]
        except:
            sys.stderr.write('Not enough distances! Try reducing R\n')
            exit(1)
        for isolate in row_labels[index]:
            if isolate in num_ranks.keys() and num_ranks[isolate] < R:
                if isolate in last_dist.keys() and last_dist[isolate] != distance:
                    num_ranks[isolate] = num_ranks[isolate] + 1
                if num_ranks[isolate] >= R:
                    completed_isolates = completed_isolates + 1
                else:
                    pair = row_labels[index][0] if row_labels[index][1] == isolate else row_labels[index][1]
                    nn[isolate][pair] = distance
                    # debug
                    if R == 5 and (isolate in in_lineage and pair not in in_lineage):
                        print('In: ' + isolate + '\tOut: ' + pair)
                    elif R == 5 and (pair in in_lineage and isolate not in in_lineage):
                        print('In: ' + pair + '\tOut: ' + isolate)
                last_dist[isolate] = distance
        index = index + 1
    # return completed dict
    return nn

def update_nearest_neighbours(distances, row_labels, R, qlist, nn, lineage_clustering, null_cluster_value):
    """ Updates the information on nearest neighbours, given
    a new set of ref-query and query-query distances.

    Args:
       distances (numpy array)
        Distances to be used for defining lineages.
       row_labels (list of tuples)
        Pairs of isolates labelling each distance.
       R (int)
        Maximum rank of distance used to define nearest neighbours.
       qlist (list)
        List of queries to be added to existing dataset.
       nn (nested dict)
        Pre-calculated neighbour relationships.
       lineage_clustering (dict)
        Clustering of existing dataset.
       null_cluster_value (int)
        Null cluster value used for unsetting lineage assignments
        where this may change due to altered neighbour relationships.

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

    # calculate max distances for each isolate
    max_distance = {}
    for existing_isolate in nn.keys():
        max_distance[existing_isolate] = max(nn[existing_isolate].values())
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
                lineage_clustering[ref] = null_cluster_value
                for neighbour in nn[ref]:
                    lineage_clustering[neighbour] = null_cluster_value
                # add neighbours
                nn[ref][query] = distance
                # delete links if no longer high ranked match
                if distance < max_distance[ref]:
                    to_delete = []
                    for other in nn[ref].keys():
                        if nn[ref][other] == max_distance[ref]:
                            to_delete.append(other)
                    for other in to_delete:
                        del nn[ref][other]
                    max_distance[ref] = max(nn[ref].values())
    # get nn for query sequences
    query_nn = generate_nearest_neighbours(distances, row_labels, qlist, R)
    # merge dicts
    for query in query_nn.keys():
        nn[query] = query_nn[query]
    # return updated dict
    return nn, lineage_clustering

def cluster_into_lineages(X, rank_list = None, output = None, rlist = None, qlist = None, existing_scheme = None, use_accessory = False, num_processes = 1):
    """ Clusters isolates into lineages based on their
    relative distances.
    
    Args:
        X (np.array)
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
    distances = X[:,distance_index]
    
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
    
    null_cluster_value = len(isolate_list) + 1
    for R in rank_list:
        lineage_clustering[R] = {i:null_cluster_value for i in isolate_list}
        lineage_seed[R] = {}
        neighbours[R] = {}
        previous_lineage_clustering[R] = {}
    
    # shared memory data structures
    shm_distances = shared_memory.SharedMemory(create = True, size = distances.nbytes, name = 'shm_distances')
    distances_shared = np.ndarray(distances.shape, dtype = distances.dtype, buffer = shm_distances.buf)
    distances_shared[:] = distances[:]

    shm_distance_ranks = shared_memory.SharedMemory(create = True, size = distance_ranks.nbytes, name = 'shm_distance_ranks')
    distance_ranks_shared = np.ndarray(distance_ranks.shape, dtype = distance_ranks.dtype, buffer = shm_distance_ranks.buf)
    distance_ranks_shared[:] = distance_ranks[:]

    isolate_list_shared = shared_memory.ShareableList(isolate_list, name = 'shm_isolate_list')
    
    # run clustering for an individual R
    if num_processes == 1:
        for R in rank_list:
            lineage_clustering[R], lineage_seed[R], neighbours[R], previous_lineage_clustering[R] = run_clustering_for_R(R,
                                                                                        null_cluster_value = null_cluster_value,
                                                                                        qlist = qlist,
                                                                                        existing_scheme = existing_scheme,
                                                                                        distances_length = distances.shape,
                                                                                        distances_type = distances.dtype)


    else:

        with Pool(processes = num_processes) as pool:
            results = pool.map(partial(run_clustering_for_R,
                                null_cluster_value = null_cluster_value,
                                qlist = qlist,
                                existing_scheme = existing_scheme,
                                distances_length = distances.shape,
                                distances_type = distances.dtype),
                                rank_list)
        
        for n,result in enumerate(results):
            R = rank_list[n]
            lineage_clustering[R], lineage_seed[R], neighbours[R], previous_lineage_clustering[R] = result
        
    # manage memory
    shm_distances.close()
    shm_distances.unlink()
    del shm_distances
    shm_distance_ranks.close()
    shm_distance_ranks.unlink()
    del shm_distance_ranks
    isolate_list_shared.shm.close()
    isolate_list_shared.shm.unlink()
    del isolate_list_shared

    # store output
    with open(output + "/" + output + '_lineageClusters.pkl', 'wb') as pickle_file:
        pickle.dump([lineage_clustering, lineage_seed, neighbours, rank_list], pickle_file)
    
    # print output
    combined = {}
    titles_list = ['Lineage_R' + str(R)  for R in rank_list]
    lineage_output_name = output + "/" + output + "_lineage_clusters.csv"
    with open(lineage_output_name, 'w') as lFile:
        # print header
        lFile.write('Id')
        for t in titles_list:
            lFile.write(',' + t + '__autocolor')
            combined[t] = {}
        lFile.write(',Overall_lineage')
        combined['Overall_lineage'] = {}
        if qlist is not None:
            lFile.write(',Status')
        lFile.write('\n')

        # print lines for each isolate
        for isolate in lineage_clustering[R].keys():
            lFile.write(isolate)
            for n,R in enumerate(rank_list):
                lFile.write(',' + str(lineage_clustering[R][isolate]))
                lineage_string = str(lineage_clustering[R][isolate])
                # include information on lineage clustering
                combined[titles_list[n]][isolate] = lineage_string
                if lineage_clustering[R][isolate] != previous_lineage_clustering[R][isolate] and previous_lineage_clustering[R][isolate] != null_cluster_value:
                    lineage_string = str(previous_lineage_clustering[R][isolate]) + ':' + lineage_string
                if isolate in combined['Overall_lineage'].keys():
                    combined['Overall_lineage'][isolate] = combined['Overall_lineage'][isolate] + '-' + lineage_string
                else:
                    combined['Overall_lineage'][isolate] = lineage_string
            lFile.write(',' + combined['Overall_lineage'][isolate])
            if qlist is not None:
                if isolate in qlist:
                    lFile.write(',Added')
                else:
                    lFile.write(',Existing')
            lFile.write('\n')

    return combined

def run_clustering_for_R(R, null_cluster_value = None, qlist = None, existing_scheme = False, distances_length = None, distances_type = None):
    """ Clusters isolates into lineages based on their
    relative distances using a single R to enable
    parallelisation.

    Args:
        R (int)
            Integer specifying the maximum rank of neighbour used
            for clustering. Should be changed to int list for hierarchical
            clustering.
        null_cluster_value (int)
            Used to denote as-yet-unclustered isolates.
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
    existing_distance_shm = shared_memory.SharedMemory(name = 'shm_distances')
    distances = np.ndarray(distances_length, dtype = distances_type, buffer = existing_distance_shm.buf)
    existing_rank_shm = shared_memory.SharedMemory(name = 'shm_distance_ranks')
    distance_ranks = np.ndarray(distances_length, dtype = 'int64', buffer = existing_rank_shm.buf)
    isolate_list = shared_memory.ShareableList(name = 'shm_isolate_list')
    
    # calculate row labels
    # this is inefficient but there appears to be no way of sharing
    # strings between processes efficiently
    row_labels = list(iter(iterDistRows(isolate_list, isolate_list, self = True)))
    # reorder by sorted distances
    row_labels = [row_labels[i] for i in distance_ranks]
    
    lineage_clustering = {i:null_cluster_value for i in isolate_list}
    previous_lineage_clustering = lineage_clustering
    lineage_seed = {}
    neighbours = {}
    
    if existing_scheme is not None:
        with open(existing_scheme, 'rb') as pickle_file:
            lineage_clustering_overall, lineage_seed_overall, neighbours_overall, rank_list = pickle.load(pickle_file)
        # focus on relevant data
        lineage_clustering = lineage_clustering_overall[R]
        lineage_seed = lineage_seed_overall[R]
        neighbours = neighbours_overall[R]
        # add new queries to lineage clustering
        for q in qlist:
            lineage_clustering[q] = null_cluster_value
        previous_lineage_clustering = lineage_clustering
        
        neighbours, lineage_clustering = update_nearest_neighbours(distances,
                                                                row_labels,
                                                                R,
                                                                qlist,
                                                                neighbours,
                                                                lineage_clustering,
                                                                null_cluster_value)
    else:
        neighbours = generate_nearest_neighbours(distances,
                                                row_labels,
                                                isolate_list,
                                                R)

    # run clustering
    lineage_index = 1
    while null_cluster_value in lineage_clustering.values():

        # get seed isolate based on minimum pairwise distances
        seed_isolate = get_seed_isolate(lineage_clustering,
                                        row_labels,
                                        distances,
                                        null_cluster_value,
                                        lineage_index,
                                        lineage_seed)
        lineage_seed[lineage_index] = seed_isolate
        
        # seed isolate is None if a previously-existing cluster has been overwritten
        # in which case pass over the lineage to keep nomenclature consistent
        if seed_isolate is not None:
        
            # record status of seed isolate
            lineage_clustering[seed_isolate] = lineage_index
            
            # get information for lineage clustering
            lineage_clustering_information = get_lineage_clustering_information(seed_isolate,
                                                                                row_labels,
                                                                                distances)
                                                                                
            # cluster the lineages
            lineage_clustering = run_lineage_clustering(lineage_clustering,
                                                        lineage_clustering_information,
                                                        neighbours,
                                                        R,
                                                        lineage_index,
                                                        seed_isolate,
                                                        previous_lineage_clustering,
                                                        null_cluster_value)
                                                        
        # increment index for next lineage
        lineage_index = lineage_index + 1

    # return clustering
    return lineage_clustering, lineage_seed, neighbours, previous_lineage_clustering

def calculateQueryDistances(dbFuncs, rlist, qfile, kmers, estimated_length,
                    queryDB, use_mash = False, threads = 1):
    """Finds edges between queries and items in the reference database,
    and modifies the network to include them.

    Args:
        dbFuncs (list)
            List of backend functions from :func:`~PopPUNK.utils.setupDBFuncs`
        rlist (list)
            List of reference names
        qfile (str)
            File containing queries
        kmers (list)
            List of k-mer sizes
        estimated_length (int)
            Estimated length of genome, if not calculated from data
        queryDB (str)
            Query database location
        use_mash (bool)
            Use the mash backend
        threads (int)
            Number of threads to use if new db created
            (default = 1)
    Returns:
        qlist1 (list)
            Ordered list of queries
        distMat (numpy.array)
            Query-query distances
    """
    
    constructDatabase = dbFuncs['constructDatabase']
    queryDatabase = dbFuncs['queryDatabase']
    readDBParams = dbFuncs['readDBParams']

    # These are returned
    qlist1 = None
    distMat = None

    # Set up query names
    qList, qSeqs = readRfile(qfile, oneSeq = use_mash)
    queryFiles = dict(zip(qList, qSeqs))
    if use_mash == True:
        rNames = None
        qNames = qSeqs
    else:
        rNames = qList
        qNames = rNames

    # Calculate all query-query distances too, if updating database
    qlist1, qlist2, distMat = queryDatabase(rNames = rNames,
                                            qNames = qNames,
                                            dbPrefix = queryDB,
                                            queryPrefix = queryDB,
                                            klist = kmers,
                                            self = True,
                                            number_plot_fits = 0,
                                            threads=threads)

    return qlist1, distMat


def readLineages(clustCSV):
    """Read a previous reference clustering from CSV

    Args:
        clustCSV (str)
            File name of CSV with previous cluster assignments

    Returns:
        clusters (dict)
            Or if return_dict is set keys are sample names,
            values are cluster assignments.
    """
    clusters = {}
    relevant_headers = []
    header_elements = []

    with open(clustCSV, 'r') as csv_file:
        header = csv_file.readline()
        # identify columns to include
        header_elements = header.rstrip().split(",")
        relevant_headers.append(header_elements.index('Overall_lineage'))
        relevant_headers.extend([n for n,i in enumerate(header_elements) if re.search('Lineage_R',i)])
        for h in relevant_headers:
            clusters[header_elements[h]] = {}
        for line in csv_file:
            elements = line.rstrip().split(",")
            if elements[0] != header_elements[0]:
                for h in relevant_headers:
                    clusters[header_elements[h]][elements[0]] = elements[h]

    return clusters
