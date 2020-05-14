#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018 John Lees and Nick Croucher

import pickle
import sys
import numpy as np
import argparse
import dendropy

# command line parsing
def get_options():

    parser = argparse.ArgumentParser(description='Extract tab-separated file of distances from pkl and npy files', prog='extract_distances')

    # input options
    parser.add_argument('--distances', required=True, help='Prefix of input pickle and numpy file of pre-calculated distances (required)')
    parser.add_argument('--tree', required=False, help='Newick file containing phylogeny of isolates', default = None)
    parser.add_argument('--output', required=True, help='Name of output file')

    return parser.parse_args()

def listDistInts(refSeqs, querySeqs, self=True):
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
    n = 0
    num_ref = len(refSeqs)
    num_query = len(querySeqs)
    if self:
        comparisons = [(0,0)] * int((num_ref * (num_ref-1)) * 0.5)
        assert refSeqs == querySeqs
        for i in range(num_ref):
            for j in range(i + 1, num_ref):
                comparisons[n] = (j, i)
                n = n + 1
    else:
        comparisons = [(0,0)] * (len(refSeqs) * len(querySeqs))
        for i in range(num_query):
            for j in range(num_ref):
                comparisons[n] = (j, i)
                n = n + 1
    return comparisons

def isolateNameToLabel(names):
    """Function to process isolate names to labels
    appropriate for visualisation.

    Args:
        names (list)
            List of isolate names.
    Returns:
        labels (list)
            List of isolate labels.
    """
    # useful to have as a function in case we
    # want to remove certain characters
    labels = [name.split('/')[-1].split('.')[0] for name in names]
    return labels

# main code
if __name__ == "__main__":

    # Check input ok
    args = get_options()

    # open stored distances
    with open(args.distances + ".pkl", 'rb') as pickle_file:
        rlist, qlist, self = pickle.load(pickle_file)
    X = np.load(args.distances + ".npy")
    
    # get names order
    r_names = isolateNameToLabel(rlist)
    q_names = isolateNameToLabel(qlist)
    names = listDistInts(r_names, q_names, self)

    # parse distances from tree, if supplied
    if args.tree is not None:
        # only calculate if all v all
        assert r_names == q_names, 'Using a phylogeny requires an all-v-all distance matrix'
        # load tree
        tree = dendropy.Tree.get(path = args.tree, schema = 'newick')
        # calculate distance matrix
        pdc = tree.phylogenetic_distance_matrix()
        # dict for identifying nodes from names
        tip_index = {}
        for t in tree.taxon_namespace:
            taxon_name = t.label.replace(' ','_')
            tip_index[r_names.index(taxon_name)] = t

    # open output file
    with open(args.output, 'w') as oFile:
        oFile.write("\t".join(['Query', 'Reference', 'Core', 'Accessory']))
        if args.tree is not None:
            oFile.write("\t" + 'Patristic')
        oFile.write("\n")
        for i, (r_index, q_index) in enumerate(names):
            oFile.write("\t".join([q_names[q_index], r_names[r_index], str(X[i,0]), str(X[i,1])]))
            if args.tree is not None:
                oFile.write("\t" + str(pdc(tip_index[r_index], tip_index[q_index])))
            oFile.write("\n")
