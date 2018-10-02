#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018 John Lees and Nick Croucher

import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster.supervised import check_clusterings
from sklearn.metrics.cluster.supervised import contingency_matrix
from scipy.special import comb

#############
# functions #
#############

# using code from https://stats.stackexchange.com/questions/89030/rand-index-calculation,
# sklearn and arandi - validated against R package mcclust
def rand_index_score(labels_true, labels_pred):

    # check clusterings
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    
    # initial statistics calculations
    n_samples = labels_true.shape[0]
    n_samples_comb = comb(n_samples,2)
    n_classes = np.unique(labels_true).shape[0]
    n_clusters = np.unique(labels_pred).shape[0]
    class_freq = np.bincount(labels_true)
    cluster_freq = np.bincount(labels_pred)
    
    # Special limit cases: no clustering since the data is not split;
    # or trivial clustering where each document is assigned a unique cluster.
    # These are perfect matches hence return 1.0.
    if (n_classes == n_clusters == 1 or
        n_classes == n_clusters == 0 or
        n_classes == n_clusters == n_samples):
        return 1.0

    # Compute the RI using the contingency data
    contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    sum_comb_c = sum((n_c**2) for n_c in cluster_freq)
    sum_comb_k = sum((n_k**2) for n_k in class_freq)
    sum_comb = sum((n_ij**2) for n_ij in contingency.data)
    
    return (1 + (sum_comb - 0.5 * sum_comb_k - 0.5 * sum_comb_c)/n_samples_comb)

# command line parsing
def get_options():

    import argparse

    parser = argparse.ArgumentParser(description='Calculate Rand index and adjusted Rand index '
                                                 'between two clusterings',
                                     prog='calculate_rand_indices')

    # input options
    parser.add_argument('--input',
                        required = True,
                        type = str,
                        help = 'Comma separated list of clustering CSVs between which '
                                'indices should be calculated (required)')
    parser.add_argument('--output', help = 'Name of output file [default = rand.out]',
                        type = str,
                        default = 'rand.out')
    parser.add_argument('--subset',
                        help = 'File with list of sequences to extract for comparison, '
                        'one per line, no header; must be present in all CSVs',
                        type = str)

    return parser.parse_args()

#################
# run main code #
#################

if __name__ == "__main__":

    # Check input ok
    args = get_options()

    # check on input files
    input_files = args.input.split(',')
    names_list = defaultdict(list)
    cluster_list = defaultdict(list)
    for input_file in input_files:
        if os.path.isfile(input_file):
            epiData = pd.read_csv(input_file, quotechar='"')
            bn = os.path.basename(input_file)
            if "Taxon" in epiData.columns.values and "Cluster" in epiData.columns.values:
                names_list[bn] = epiData["Taxon"]
                cluster_list[bn] = pd.factorize(epiData["Cluster"])[0]
            else:
                sys.stderr.write("Input files need to be PopPUNK clustering "
                                 "CSVs; " + input_file + " appears to be misformatted\n")
        else:
            sys.stderr.write("File does not exist: " + input_file + "\n")
            sys.exit(1)

    # parse subset if provided
    subset_seq = []
    if args.subset is not None and os.path.isfile(args.subset):
        with open(args.subset, 'r') as subset_file:
            for line in subset_file:
                subset_seq.append(line.rstrip())

    # open output and print header
    with open(args.output, 'w') as output_file:
        output_file.write("\t".join(["File_1","File_2","n_1","n_2",
                                     "n_compared","Rand_index","Adjusted_Rand_index"]) + "\n")

        # iterate through clusterings
        for x,input_x in enumerate(names_list):
            x_set = set(names_list[input_x])
            for y,input_y in enumerate(names_list):
                if (x < y):

                    # get intersection of names
                    indices_y = []
                    indices_x = []
                    if args.subset is None:
                        for i,name in enumerate(names_list[input_y]):
                            if name in x_set:
                                indices_y.append(i)
                                indices_x.append(np.asscalar(np.where(names_list[input_x].values == name)[0]))
                    else:
                        for name in subset_seq:
                            if name in x_set:
                                indices_y.append(np.asscalar(np.where(names_list[input_y].values == name)[0]))
                                indices_x.append(np.asscalar(np.where(names_list[input_x].values == name)[0]))
                            else:
                                sys.stderr.write("Cannot find " + name + " in both datasets")

                    # calculate indices
                    if len(indices_x) > 0:
                        numpy_x = np.asarray(cluster_list[input_x][indices_x],dtype=int)
                        numpy_y = np.asarray(cluster_list[input_y][indices_y],dtype=int)
                        ri = rand_index_score(numpy_x,numpy_y)
                        ari = adjusted_rand_score(cluster_list[input_x][indices_x],cluster_list[input_y][indices_y])

                        # print output line
                        output_file.write("\t".join([input_x, input_y, str(len(cluster_list[input_x])),
                                                     str(len(cluster_list[input_y])), str(len(indices_x)),
                                                     str(ri), str(ari)]) + "\n")
                    else:
                        sys.stderr.write("No overlapping sequences for files " + input_x +
                                         " and " + input_y + "\n")

    sys.exit(0)
