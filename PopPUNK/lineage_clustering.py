#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018-2020 John Lees and Nick Croucher

# universal
import os
import sys
# additional
import numpy as np
from scipy.stats import rankdata
from collections import defaultdict
import pickle

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
        # extract all pairwise distances between isolates that are not yet clustered
        clustered_isolates = frozenset([isolate for isolate in lineage_clustering.keys() if lineage_clustering[isolate] != null_cluster_value])
        unclustered_pair_indices = [n for n,pair in enumerate(row_labels) if not set(pair).issubset(clustered_isolates)]
        unclustered_distances = distances[unclustered_pair_indices]
        # start at zero as the minimum if present in the unclustered set
        minimum_unclustered_distance = 0
        # otherwise go through the bother of picking a minimum
        if minimum_unclustered_distance not in unclustered_distances:
            if len(unclustered_distances) == 1:
                minimum_unclustered_distance = unclustered_distances[0]
            elif len(unclustered_distances) > 1:
                minimum_unclustered_distance = np.amin(unclustered_distances)
            else:
                sys.stderr.write('Cannot find any more isolates not assigned to lineages; unclustered pair indices are ' + str(unclustered_pair_indices))
                exit(1)
        # identify which entries in the full distance set have this value
        # unfortunately these distances are not necessarily neighbours
        distance_indices = np.where(distances == minimum_unclustered_distance)[0].tolist()
        # identify an unclustered isolate from the pair that have this value
        closest_unclustered_pair_index =                                           int(set(unclustered_pair_indices).intersection(set(distance_indices)).pop())
        seed_isolate = row_labels[closest_unclustered_pair_index][0] if lineage_clustering[row_labels[closest_unclustered_pair_index][0]] == null_cluster_value else row_labels[closest_unclustered_pair_index][1]
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
    # get subset of relevant distances
    comparisons_involving_seed = [n for n,(pair) in enumerate(row_labels) if seed_isolate in pair]
    distances_to_seed = distances[comparisons_involving_seed]
    # get ranks of data
    distance_ranks = rankdata(distances_to_seed)
    # get partners of seed isolate in each comparison
    pairs_involving_seed = [row_labels[x] for x in comparisons_involving_seed]
    seed_partners = [r if q == seed_isolate else q for (r,q) in pairs_involving_seed]
    # create a dict of lists of isolates for a given rank
    # enables later easy iterating through ranked distances
    for rank in np.unique(distance_ranks):
        lineage_info[rank] = [seed_partners[n] for n,r in enumerate(distance_ranks) if r == rank]
    return lineage_info

def generate_nearest_neighbours(distances, row_labels, isolate_list, R):
    """ Identifies the nearest neighbours from the core
    genome distances.
    
    Args:
        distances (numpy array)
            Pairwise distances used for defining relationships.
        row_labels (list of tuples)
            Pairs of isolates labelling each distance.
        isolate_list (list)
            List of isolates for lineage assignment.
        R (int)
            Maximum rank of neighbours used for clustering.
    
    Returns:
        nn
            Dict of dict of neighbours per isolate, and pairwise distances
    """
    # data structures
    nn = {}
    # iterate through isolates
    for isolate in isolate_list:
        # initiate dict
        nn[isolate] = {}
        # get indices of rows involving the considered isolate
        indices = [n for n, (r, q) in enumerate(row_labels) if r == isolate or q == isolate]
        # get the corresponding distances
        #ref_distances = np.take(X, np.array(indices, dtype = 'int64'))
        ref_distances = np.take(distances, indices)
        # get unique distances and pick the R smallest values
        unique_isolate_distances = np.unique(ref_distances)
        R_lowest_distances = unique_isolate_distances[np.argpartition(unique_isolate_distances, R)]
        # get threshold distance for neighbour definition - np.maximum fails for R=1
        threshold_distance = R_lowest_distances[0]
        if R > 1:
            threshold_distance = np.maximum(R_lowest_distances)
        # identify all distances below this threshold and store relative to index
        for i,d in enumerate(ref_distances):
            if d <= threshold_distance:
                # get index of row in full list
                index = indices[i]
                # get name of other isolate in pair
                name = row_labels[index][1] if row_labels[index][0] == isolate else row_labels[index][0]
                nn[isolate][name] = ref_distances[i]
    # return completed dictionary
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
    # iterate through references to update existing entries
    existing_isolates = nn.keys()
    
    for isolate in existing_isolates:
        # get max distance to existing neighbours
        max_distance_to_ref = max(nn[isolate].values())
        # get min distance to queries
        ref_query_match_indices = [n for n, (r, q) in enumerate(query_row_names) if r == isolate or q == isolate]
        ref_query_distances = np.take(query_distances, ref_query_match_indices)
        min_distance_to_query = np.amin(ref_query_distances)
        # if closer queries, find distances and replace
        if min_distance_to_query < max_distance_to_ref:
            # unset clustering of isolate and neighbours
            lineage_clustering[isolate] = null_cluster_value
            for neighbour in nn[isolate].keys():
                lineage_clustering[neighbour] = null_cluster_value
            # update neighbour information
            nn_old = nn[isolate]
            nn_new[isolate] = {}
            # old neighbours from existing dict
            old_distances = list(set(nn[isolate].values()))
            # new neighbours from new distances in X
            new_distances = np.unique(ref_query_distances)
            # combine old and new distances
            combined_ref_distances = list(nn[isolate].values()) + ref_query_distances.tolist()
            # take top ranked distances from both
            ranked_unique_combined_ref_distances = sorted(set(combined_ref_distances))[:R]
            for distance in ranked_unique_combined_ref_distances:
                # add old neighbours back in
                if distance in old_distances:
                    for neighbour,d in nn[isolate].items():
                        if d == distance:
                            nn_new[isolate][neighbour] = distance
                # add new neighbours
                if distance in new_distances:
                    for n,d in enumerate(ref_query_distances):
                        if d == distance:
                            row_label_index = ref_query_match_indices[n]
                            neighbour = row_labels[row_label_index][0] if row_labels[row_label_index][1] == isolate else row_labels[row_label_index][1]
                            nn_new[isolate][neighbour] = d
    # get nn for query sequences
    query_nn = generate_nearest_neighbours(distances, row_labels, qlist, R)
    # merge dicts
    for query in query_nn.keys():
        nn[query] = query_nn[query]
    for updated in nn_new.keys():
        nn[updated] = nn_new[updated]
    # return updated dict
    return nn, lineage_clustering


def cluster_into_lineages(X, R = 1, output = None, rlist = None, qlist = None, existing_scheme = None, use_accessory = False):
    """ Clusters isolates into lineages based on their
    relative distances.
    
    Args:
        X (np.array)
            n x 2 array of core and accessory distances for n samples.
            This should not be subsampled.
        R (int)
            Integer specifying the maximum rank of neighbour used
            for clustering. Should be changed to int list for hierarchical
            clustering.
        rlist (list)
            List of reference sequences.
        qlist (list)
            List of query sequences.
        existing_scheme (str)
            Path to pickle file containing lineage scheme to which isolates
            should be added.
        use_accessory (bool)
            Option to use accessory distances rather than core distances.
            
    Returns:
        completion_status (int)
            0 if successful
    """
    # process distance matrix
    distance_index = 1 if use_accessory else 0
    distances = X[:,distance_index]
    
    # determine whether ref-ref or ref-query analysis
    isolate_list = rlist + qlist
    row_labels = list(iter(iterDistRows(isolate_list, isolate_list, self = True)))
#    if qlist is None:
#        sys.stderr.write("Identifying lineages using reference sequences\n")
#        isolate_list = rlist
#        row_labels = list(iter(iterDistRows(rlist, rlist, self = True)))
#    else:
#        sys.stderr.write("Identifying lineages from using reference and query sequences\n")
#        row_labels = list(iter(iterDistRows(rlist, qlist, self = False)))
#        isolate_list = rlist + qlist
    
    # determine whether novel analysis or modifying existing analysis
    neighbours = {}
    lineage_seed = {}
    null_cluster_value = len(isolate_list) + 1
    lineage_clustering = {i:null_cluster_value for i in isolate_list}
    previous_lineage_clustering = lineage_clustering
    
    if existing_scheme is None:
        neighbours = generate_nearest_neighbours(distances,
                                                row_labels,
                                                isolate_list,
                                                R)
    else:
        with open(existing_scheme, 'rb') as pickle_file:
            sys.stderr.write('Loading from stored file ' + str(existing_scheme) + '\n')
            lineage_clustering, lineage_seed, neighbours, R = pickle.load(pickle_file)
        previous_lineage_clustering = lineage_clustering # use for detecting changes
        # add new queries to lineage clustering
        for q in qlist:
            lineage_clustering[q] = null_cluster_value
        neighbours, lineage_clustering = update_nearest_neighbours(distances,
                                                                    row_labels,
                                                                    R,
                                                                    qlist,
                                                                    neighbours,
                                                                    lineage_clustering,
                                                                    null_cluster_value)

    # run clustering
    lineage_index = 1
    while null_cluster_value in lineage_clustering.values():
        
        # get seed isolate based on minimum pairwise distances
        seed_isolate = get_seed_isolate(lineage_clustering,     # need to add in lineage seed hash
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
    
    # store output
    with open(output + "/" + output + '_lineageClusters.pkl', 'wb') as pickle_file:
        pickle.dump([lineage_clustering, lineage_seed, neighbours, R], pickle_file)
    
    # print output
    lineage_output_name = output + "/" + output + "_lineage_clusters.csv"
    with open(lineage_output_name, 'w') as lFile:
        if qlist is None:
            print('Id,Lineage__autocolor', file = lFile)
            for isolate in lineage_clustering.keys():
                print(isolate + ',' + str(lineage_clustering[isolate]), file = lFile)
        else:
            print('Id,Lineage__autocolor,QueryStatus', file = lFile)
            for isolate in rlist:
                print(isolate + ',' + str(lineage_clustering[isolate]) + ',' + 'Reference', file = lFile)
            for isolate in qlist:
                print(isolate + ',' + str(lineage_clustering[isolate]) + ',' + 'Query', file = lFile)

    return lineage_clustering

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


