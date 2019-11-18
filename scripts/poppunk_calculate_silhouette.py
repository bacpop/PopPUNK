#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018 John Lees and Nick Croucher

import os
import sys
import pickle
import re
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.metrics import silhouette_score

#############
# functions #
#############

# command line parsing
def get_options():

    import argparse

    parser = argparse.ArgumentParser(description='Calculate Silhouette Coefficient for a clustering',
                                     prog='calculate_silhouette')

    # input options
    parser.add_argument('--distances', required=True, help='Prefix of input pickle of pre-calculated distances (required)')
    parser.add_argument('--cluster-csv', required=True, help='Cluster assignments')
    parser.add_argument('--cluster-col', type = int, default = 2, help="Column with cluster assignment (default = 2)")
    parser.add_argument('--id-col', type = int, default = 1, help="Column with sample names (default = 1)")
    parser.add_argument('--sub', type = str, default = None, help="Remove string from sample names")


    return parser.parse_args()

#################
# run main code #
#################

if __name__ == "__main__":

    # Check input ok
    args = get_options()

    # Read in old distances
    with open(args.distances + ".pkl", 'rb') as pickle_file:
        rlist, qlist, self = pickle.load(pickle_file)
    distMat = np.load(args.distances  + ".npy")
    if not self:
        raise RuntimeError("Distance DB should be self-self distances")

    if args.sub is not None:
        for ref_idx, ref_name in enumerate(rlist):
            rlist[ref_idx] = re.sub(pattern=args.sub, repl='', string=ref_name)

    # Convert dists into N x N matrix
    i = 0
    j = 1
    X_mat = np.zeros((len(rlist), len(rlist)))
    for distRow in distMat:
        X_mat[i, j] = euclidean(distRow[0], distRow[1])
        X_mat[j, i] = X_mat[i, j]
        if j == (len(rlist) - 1):
            i += 1
            j = i + 1
        else:
            j += 1

    # Read in clustering
    clustering = pd.read_csv(args.cluster_csv, index_col = args.id_col - 1, quotechar='"')
    name_map = {}
    if args.sub is not None:
        for ind_name in clustering.index:
            name_map[ind_name] = re.sub(pattern=args.sub, repl='', string=ind_name)
        clustering.rename(index = name_map, inplace = True)

    labels = []
    for sample in rlist:
        labels.append(clustering.loc[sample][args.cluster_col - 2])
    labels = np.array(labels)

    print(silhouette_score(X_mat, labels, metric='precomputed'))
