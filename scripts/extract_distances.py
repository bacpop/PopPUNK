#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018 John Lees and Nick Croucher

import pickle
import sys
import numpy as np
import argparse

# command line parsing
def get_options():

    parser = argparse.ArgumentParser(description='Extract tab-separated file of distances from pkl and npy files', prog='extract_distances')

    # input options
    parser.add_argument('--distances', required=True, help='Prefix of input pickle and numpy file of pre-calculated distances (required)')
    parser.add_argument('--output', required=True, help='Name of output file')

    return parser.parse_args()

def iterDistRows(refSeqs, querySeqs, self=True):
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
    if self:
        if refSeqs != querySeqs:
            raise RuntimeError('refSeqs must equal querySeqs for db building (self = true)')
        for i, ref in enumerate(refSeqs):
            for j in range(i + 1, len(refSeqs)):
                yield(refSeqs[j], ref)
    else:
        for query in querySeqs:
            for ref in refSeqs:
                yield(ref, query)

# main code
if __name__ == "__main__":

    # Check input ok
    args = get_options()

    # open stored distances
    with open(args.distances + ".pkl", 'rb') as pickle_file:
        rlist, qlist, self = pickle.load(pickle_file)
    X = np.load(args.distances + ".npy")

    # get names order
    names = iterDistRows(rlist, qlist, self)

    # open output file
    with open(args.output, 'w') as oFile:
        oFile.write("\t".join(['Query', 'Reference', 'Core', 'Accessory']) + "\n")
        for i, (ref, query) in enumerate(names):
            oFile.write("\t".join([query, ref, str(X[i,0]), str(X[i,1])]) + "\n")
