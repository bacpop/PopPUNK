#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018 John Lees and Nick Croucher

import pickle
import sys, os
import numpy as np
import argparse
import dendropy
from scipy import sparse

# command line parsing
def get_options():

    parser = argparse.ArgumentParser(description='Extract tab-separated file of distances from pkl and npy files', prog='extract_distances')

    # input options
    parser.add_argument('--distances', help='Prefix of input pickle (and optionally,'
    '  numpy file) of pre-calculated distances (required)',
                                    required=True)
    parser.add_argument('--sparse', help='Sparse distance matrix file name',
                                    default = None,
                                    required = False)
    parser.add_argument('--tree', help='Newick file containing phylogeny of isolates',
                                    required = False,
                                    default = None)
    parser.add_argument('--output', help='Name of output file',
                                    required = True)

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
    num_ref = len(refSeqs)
    num_query = len(querySeqs)
    if self:
        if refSeqs != querySeqs:
            raise RuntimeError('refSeqs must equal querySeqs for db building (self = true)')
        for i in range(num_ref):
            for j in range(i + 1, num_ref):
                yield(j, i)
    else:
        comparisons = [(0,0)] * (len(refSeqs) * len(querySeqs))
        for i in range(num_query):
            for j in range(num_ref):
                yield(j, i)

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
    labels = [os.path.splitext(os.path.basename(name))[0] for name in names]
    return labels

# main code
if __name__ == "__main__":

    # Check input ok
    args = get_options()

    # open stored distances
    with open(args.distances + ".pkl", 'rb') as pickle_file:
        rlist, qlist, self = pickle.load(pickle_file)

    # get names order
    r_names = isolateNameToLabel(rlist)
    q_names = isolateNameToLabel(qlist)

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

    # Load sparse matrix
    if args.sparse is not None:
        sparse_mat = sparse.load_npz(args.sparse)
    else:
        X = np.load(args.distances + ".npy")

    # open output file
    with open(args.output, 'w') as oFile:
        # Write header of output file
        if args.sparse is not None:
            oFile.write("\t".join(['Query', 'Reference', 'Core']))
        else:
            oFile.write("\t".join(['Query', 'Reference', 'Core', 'Accessory']))
        if args.tree is not None:
            oFile.write("\t" + 'Patristic')
        oFile.write("\n")
        # Write distances
        if args.sparse is not None:
            for (r_index, q_index, dist) in zip(sparse_mat.col, sparse_mat.row, sparse_mat.data):
                oFile.write("\t".join([q_names[q_index], r_names[r_index], str(dist)]))
                if args.tree is not None:
                    oFile.write("\t" + str(pdc(tip_index[r_index], tip_index[q_index])))
                oFile.write("\n")
        else:
            for i, (r_index, q_index) in enumerate(listDistInts(r_names, q_names, r_names == q_names)):
                oFile.write("\t".join([q_names[q_index], r_names[r_index], str(X[i,0]), str(X[i,1])]))
                if args.tree is not None:
                    oFile.write("\t" + str(pdc(tip_index[r_index], tip_index[q_index])))
                oFile.write("\n")
