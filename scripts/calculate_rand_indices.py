#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018 John Lees and Nick Croucher

import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import adjusted_rand_score
from scipy.misc import comb

#############
# functions #
#############

# from https://stats.stackexchange.com/questions/89030/rand-index-calculation
def rand_index_score(clusters, classes):

    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)

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
                cluster_list[bn] = epiData["Cluster"]
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
                                     "n_compared","Rand_index","Adjusted_Rand_index"]))

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
                            indices_y.append(np.asscalar(np.where(names_list[input_y].values == name)[0]))
                            indices_x.append(np.asscalar(np.where(names_list[input_x].values == name)[0]))

                    # calculate indices
                    if len(indices_x) > 0:
                        numpy_x = np.asarray(cluster_list[input_x].values[indices_x],dtype=int)
                        numpy_y = np.asarray(cluster_list[input_y].values[indices_y],dtype=int)
                        ri = rand_index_score(numpy_x,numpy_y)
                        ari = adjusted_rand_score(cluster_list[input_x][indices_x],cluster_list[input_y][indices_y])

                        # print output line
                        output_file.write("\t".join([input_x, input_y, str(len(cluster_list[input_x])),
                                                     str(len(cluster_list[input_y])), str(len(indices_x)),
                                                     str(ri), str(ari)]))
                    else:
                        sys.stderr.write("No overlapping sequences for files " + input_x +
                                         " and " + input_y + "\n")

    sys.exit(0)
