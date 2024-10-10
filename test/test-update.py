#!/usr/bin/env python
# Copyright 2018-2021 John Lees and Nick Croucher

"""Tests for PopPUNK --update-db order"""

import subprocess
import os, sys
import sys
import shutil
import pickle

import numpy as np
from scipy import stats
import h5py
import scipy.sparse

import pp_sketchlib

if os.environ.get("POPPUNK_PYTHON"):
    python_cmd = os.environ.get("POPPUNK_PYTHON")
else:
    python_cmd = "python"

def run_regression(x, y, threshold = 0.99):
    res = stats.linregress(x, y)
    print("R^2: " + str(res.rvalue**2))
    if res.rvalue**2 < threshold:
        sys.stderr.write("Distance matrix order failed!\n")
        sys.exit(1)

def compare_sparse_matrices(d1,d2,r1,r2,flag):
    d1_pairs = get_seq_tuples(d1.row,d1.col,r1)
    d2_pairs = get_seq_tuples(d2.row,d2.col,r2)
    d1_dists = []
    d2_dists = []
    if (len(d1_pairs) != len(d2_pairs) and flag == " "): # May not be equal if reciprocal/unique count
        sys.stderr.write("Distance matrix number of entries differ!\n")
        print(d1_pairs)
        print(d2_pairs)
        sys.exit(1)

    for (pair1,dist1) in zip(d1_pairs,d1.data):
        for (pair2,dist2) in zip(d2_pairs,d2.data):
#            print('Pair1: ' + str(pair1) + ' Dist1: ' + str(dist1) + 'Pair2: ' + str(pair2) + ' Dist2: ' + str(dist2))
            if pair1 == pair2:
                d1_dists.append(dist1)
                d2_dists.append(dist2)
                break

    run_regression(np.asarray(d1_dists),np.asarray(d2_dists))

def get_seq_tuples(rows,cols,names):
    tuple_list = []
    for (i,j) in zip(rows,cols):
        sorted_pair = tuple(sorted((names[i],names[j])))
        tuple_list.append(sorted_pair)
    return tuple_list

def old_get_seq_tuples(rows,cols):
    max_seqs = np.maximum(rows,cols)
    min_seqs = np.minimum(rows,cols)
    concat_seqs = np.vstack((max_seqs,min_seqs))
    seq_pairs = concat_seqs.T
    seq_tuples = [tuple(row) for row in seq_pairs]
    return seq_tuples

for lineage_option_string in [" "," --count-unique-distances ", " --reciprocal-only "," --count-unique-distances --reciprocal-only "]:

    if lineage_option_string != " ":
      print("\n*** Now running tests with lineage option" + lineage_option_string + "***\n")

    # Check distances after one query

    # Check that order is the same after doing 1 + 2 with --update-db, as doing all of 1 + 2 together
    subprocess.run(python_cmd + " ../poppunk-runner.py --create-db --r-files rfile12.txt --output batch12 --overwrite", shell=True, check=True)
    subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model lineage --ref-db batch12 --ranks 1,2 --overwrite" + lineage_option_string,shell=True, check=True)
    print(" ../poppunk-runner.py --fit-model lineage --ref-db batch12 --ranks 1,2 --overwrite\n\n")
    subprocess.run(python_cmd + " ../poppunk-runner.py --create-db --r-files rfile1.txt --output batch1 --overwrite", shell=True, check=True)
    subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model lineage --ref-db batch1 --ranks 1,2 --overwrite" + lineage_option_string, shell=True, check=True)
    print("../poppunk-runner.py --fit-model lineage --ref-db batch1 --ranks 1,2 --overwrite\n\n")
    subprocess.run(python_cmd + " ../poppunk_assign-runner.py --db batch1 --query rfile2.txt --output batch2 --update-db --overwrite --max-a-dist 1", shell=True, check=True)
    print(" ../poppunk_assign-runner.py --db batch1 --query rfile2.txt --output batch2 --update-db --overwrite --max-a-dist 1\n\n")

    # Load updated distance order
    with open("batch2/batch2.dists.pkl", 'rb') as pickle_file:
        rlist2, qlist, self = pickle.load(pickle_file)

  # Check sparse distances after one query
    with open("batch12/batch12.dists.pkl", 'rb') as pickle_file:
        rlist1, qlist1, self = pickle.load(pickle_file)
    S1 = scipy.sparse.load_npz("batch12/batch12_rank_2_fit.npz")
    S2 = scipy.sparse.load_npz("batch2/batch2_rank_2_fit.npz")
    sys.stderr.write("Comparing sparse matrices at rank 2 after first query calculated with options " + lineage_option_string + "\n")
    compare_sparse_matrices(S1,S2,rlist1,rlist2,lineage_option_string)

    # Check rank 1
    S3 = scipy.sparse.load_npz("batch12/batch12_rank_1_fit.npz")
    S4 = scipy.sparse.load_npz("batch2/batch2_rank_1_fit.npz")
    sys.stderr.write("Comparing sparse matrices at rank 1 after first query calculated with options " + lineage_option_string + "\n")
    compare_sparse_matrices(S3,S4,rlist1,rlist2,lineage_option_string)

    # Check distances after second query

    # Check that order is the same after doing 1 + 2 + 3 with --update-db, as doing all of 1 + 2 + 3 together
    subprocess.run(python_cmd + " ../poppunk-runner.py --create-db --r-files rfile123.txt --output batch123 --overwrite", shell=True, check=True)
    subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model lineage --ref-db batch123 --ranks 1,2 --overwrite" + lineage_option_string, shell=True, check=True)
    print("../poppunk-runner.py --fit-model lineage --ref-db batch123 --ranks 1,2 --overwrite\n\n")
    subprocess.run(python_cmd + " ../poppunk_assign-runner.py --db batch2 --query rfile3.txt --output batch3 --update-db --overwrite", shell=True, check=True)
    print(python_cmd + " ../poppunk_assign-runner.py --db batch2 --query rfile3.txt --output batch3 --update-db --overwrite\n\n")

    # Load updated distances order
    with open("batch3/batch3.dists.pkl", 'rb') as pickle_file:
        rlist4, qlist, self = pickle.load(pickle_file)

    # Check sparse distances after second query
    with open("batch123/batch123.dists.pkl", 'rb') as pickle_file:
        rlist3, qlist, self = pickle.load(pickle_file)
    S5 = scipy.sparse.load_npz("batch123/batch123_rank_2_fit.npz")
    S6 = scipy.sparse.load_npz("batch3/batch3_rank_2_fit.npz")
    sys.stderr.write("Comparing sparse matrices at rank 2 after second query calculated with options " + lineage_option_string + "\n")
    compare_sparse_matrices(S5,S6,rlist3,rlist4,lineage_option_string)

    # Check rank 1
    S7 = scipy.sparse.load_npz("batch123/batch123_rank_1_fit.npz")
    S8 = scipy.sparse.load_npz("batch3/batch3_rank_1_fit.npz")
    sys.stderr.write("Comparing sparse matrices at rank 1 after second query calculated with options " + lineage_option_string + "\n")
    compare_sparse_matrices(S7,S8,rlist3,rlist4,lineage_option_string)
