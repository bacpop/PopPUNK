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

# import poppunk package
from .utils import iterDistRows

def run_lineage_clustering(lineage_clustering, lineage_clustering_information, neighbours, R, lineage_index, seed_isolate):
    """ Identifies isolates corresponding to a particular
    lineage given a cluster seed.
    
    Args:
        ranked_information
        nearest_neighbours (dict)
            
        R (int)
            Maximum rank of neighbours used for clustering.
    
    Returns:
        ranked_information
    
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
#            print("Rank: " + str(rank) + " isolate: " + isolate)
            if lineage_clustering[isolate] > lineage_index:
                # get clusters of nearest neighbours
                isolate_neighbour_clusters = [lineage_clustering[isolate_neighbour] for isolate_neighbour in neighbours[isolate].keys()]
#                print('Neighbours: ' + str(neighbours[isolate].keys()) + '\nNeighbour clusters: ' + str(isolate_neighbour_clusters))
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
    return lineage_clustering

def get_seed_isolate(lineage_clustering, row_labels, distances, null_cluster_value):
    """ Identifies the isolate used to initiate a cluster
    """
    # extract all pairwise distances between isolates that are not yet clustered
    clustered_isolates = frozenset([isolate for isolate in lineage_clustering.keys() if lineage_clustering[isolate] != null_cluster_value])
    unclustered_pair_indices = [n for n,pair in enumerate(row_labels) if not set(pair).issubset(clustered_isolates)]
    unclustered_distances = distances[unclustered_pair_indices]
#    print("Unclustered distances are: " + str(unclustered_distances))
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
    new_seed_isolate = row_labels[closest_unclustered_pair_index][0] if lineage_clustering[row_labels[closest_unclustered_pair_index][0]] == null_cluster_value else row_labels[closest_unclustered_pair_index][1]
    return new_seed_isolate
    

def get_lineage_clustering_information(seed_isolate, row_labels, distances):
    """ Generates the ranked distances needed for cluster
    definition.
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

def generate_nearest_neighbours(X, row_labels, isolate_list, R):
    """ Identifies the nearest neighbours from the core
    genome distances.
    
    Args
        X (np.array)
            n x 2 array of core and accessory distances for n samples.
        row_labels (list of tuples)
            List of pairs of isolates
        isolate_list (list)
            List of isolates
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
        #ref_distances = X[np.array(indices):1]
        ref_distances = np.take(X, np.array(indices))
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


def update_nearest_neighbours():
    """ Updates the information on nearest neighbours, given
    a new set of ref-query and query-query distances.
    """
    return 0


def cluster_into_lineages(X, R, output, rlist = None, qlist = None, existing_model = None, use_accessory = False):
    """ Clusters isolates into lineages based on their
    relative distances.
    
    Args
        X (np.array)
            n x 2 array of core and accessory distances for n samples.
            This should not be subsampled.
        R (int)
            Integer specifying the maximum rank of neighbour used
            for clustering.
        rlist (list)
            List of reference sequences.
        qlist (list)
            List of query sequences.
        existing_model (?)
            Existing lineage clustering model.
    Returns:
        ret_vec
            An n-vector with the most likely cluster memberships
    """
    # process data lineage identification
    sorted_distances = np.empty(shape = X.shape[0])
    row_labels = []
    isolate_list = []
    if qlist is None:
        isolate_list = rlist
        distance_index = 1 if use_accessory else 0
        distances = X[:,distance_index]
        row_labels = list(iter(iterDistRows(rlist, rlist, self=True)))
    else:
        sys.stderr.write("Adding to lineage not yet implemented\n")
    # identify nearest neighbours
    neighbours = {}
    neighbours = generate_nearest_neighbours(distances,
                                            row_labels,
                                            rlist,
                                            R)
    # run clustering
    null_cluster_value = len(isolate_list) + 1
    lineage_clustering = {i:null_cluster_value for i in isolate_list}
    lineage_index = 1
    lineage_seed = {}
    while null_cluster_value in lineage_clustering.values():
        # get seed isolate based on minimum pairwise distances
        seed_isolate = get_seed_isolate(lineage_clustering,
                                        row_labels,
                                        distances,
                                        null_cluster_value)
        # record status of seed isolate
        lineage_seed[lineage_index] = seed_isolate
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
                                                    seed_isolate)
        # increment index for next lineage
        lineage_index = lineage_index + 1
    
    # print output
    lineage_output_name = output + "/" + output + "_lineage_clusters.csv"
    with open(lineage_output_name, 'w') as lFile:
        print('Id,Lineage__autocolor', file = lFile)
        for isolate in lineage_clustering.keys():
            print(isolate + ',' + str(lineage_clustering[isolate]), file = lFile)
    
    return 0

