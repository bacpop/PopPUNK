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

def compare_sparse_matrices(d1,d2,r1,r2):
    d1_pairs = get_seq_tuples(d1.row,d1.col,r1)
    d2_pairs = get_seq_tuples(d2.row,d2.col,r2)
    d1_dists = []
    d2_dists = []

    for (pair1,dist1) in zip(d1_pairs,d1.data):
        for (pair2,dist2) in zip(d2_pairs,d2.data):
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

# Check distances after one query

# Check that order is the same after doing 1 + 2 with --update-db, as doing all of 1 + 2 together
subprocess.run(python_cmd + " ../poppunk-runner.py --create-db --r-files rfile12.txt --output batch12 --overwrite", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model lineage --ref-db batch12 --ranks 1,2", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --create-db --r-files rfile1.txt --output batch1 --overwrite", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model lineage --ref-db batch1 --ranks 1,2", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_assign-runner.py --db batch1 --query rfile2.txt --output batch2 --update-db --overwrite", shell=True, check=True)

# Load updated distances
X2 = np.load("batch2/batch2.dists.npy")
with open("batch2/batch2.dists.pkl", 'rb') as pickle_file:
    rlist2, qlist, self = pickle.load(pickle_file)

# Get same distances from the full database
ref_db = "batch12/batch12"
ref_h5 = h5py.File(ref_db + ".h5", 'r')
db_kmers = sorted(ref_h5['sketches/' + rlist2[0]].attrs['kmers'])
ref_h5.close()
X1 = pp_sketchlib.queryDatabase(ref_db, ref_db, rlist2, rlist2, db_kmers,
                                True, False, 1, False, 0)

# Check distances match
run_regression(X1[:, 0], X2[:, 0])
run_regression(X1[:, 1], X2[:, 1])

# Check sparse distances after one query
with open("batch12/batch12.dists.pkl", 'rb') as pickle_file:
    rlist1, qlist1, self = pickle.load(pickle_file)
S1 = scipy.sparse.load_npz("batch12/batch12_rank2_fit.npz")
S2 = scipy.sparse.load_npz("batch2/batch2_rank2_fit.npz")
compare_sparse_matrices(S1,S2,rlist1,rlist2)

# Check distances after second query

# Check that order is the same after doing 1 + 2 + 3 with --update-db, as doing all of 1 + 2 + 3 together
subprocess.run(python_cmd + " ../poppunk-runner.py --create-db --r-files rfile123.txt --output batch123 --overwrite", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model lineage --ref-db batch123 --ranks 1,2", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_assign-runner.py --db batch2 --query rfile3.txt --output batch3 --update-db --overwrite", shell=True, check=True)

# Load updated distances
X2 = np.load("batch3/batch3.dists.npy")
with open("batch3/batch3.dists.pkl", 'rb') as pickle_file:
    rlist4, qlist, self = pickle.load(pickle_file)

# Get same distances from the full database
ref_db = "batch123/batch123"
ref_h5 = h5py.File(ref_db + ".h5", 'r')
db_kmers = sorted(ref_h5['sketches/' + rlist4[0]].attrs['kmers'])
ref_h5.close()
X1 = pp_sketchlib.queryDatabase(ref_db, ref_db, rlist4, rlist4, db_kmers,
                                True, False, 1, False, 0)

# Check distances match
run_regression(X1[:, 0], X2[:, 0])
run_regression(X1[:, 1], X2[:, 1])

# Check sparse distances after second query
with open("batch123/batch123.dists.pkl", 'rb') as pickle_file:
    rlist3, qlist, self = pickle.load(pickle_file)
S3 = scipy.sparse.load_npz("batch123/batch123_rank2_fit.npz")
S4 = scipy.sparse.load_npz("batch3/batch3_rank2_fit.npz")

compare_sparse_matrices(S3,S4,rlist3,rlist4)
