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

sys.path.insert(0, '/Users/jlees/Documents/Imperial/pp-sketchlib/build/lib.macosx-10.9-x86_64-3.8')
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

# Check that order is the same after doing 1 + 2 with --update-db, as doing all of 1 + 2 together
subprocess.run(python_cmd + " ../poppunk-runner.py --create-db --r-files rfile12.txt --output batch12 --overwrite", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --create-db --r-files rfile1.txt --output batch1 --overwrite", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model lineage --ref-db batch1 --ranks 1", shell=True, check=True)
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

# Check that order is the same after doing 1 + 2 + 3 with --update-db, as doing all of 1 + 2 + 3 together
subprocess.run(python_cmd + " ../poppunk-runner.py --create-db --r-files rfile123.txt --output batch123 --overwrite", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model lineage --ref-db batch123 --ranks 1", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_assign-runner.py --db batch2 --query rfile3.txt --output batch3 --update-db --overwrite", shell=True, check=True)

# Load updated distances
X2 = np.load("batch3/batch3.dists.npy")
with open("batch3/batch3.dists.pkl", 'rb') as pickle_file:
    rlist3, qlist, self = pickle.load(pickle_file)

# Get same distances from the full database
ref_db = "batch123/batch123"
ref_h5 = h5py.File(ref_db + ".h5", 'r')
db_kmers = sorted(ref_h5['sketches/' + rlist3[0]].attrs['kmers'])
ref_h5.close()
X1 = pp_sketchlib.queryDatabase(ref_db, ref_db, rlist3, rlist3, db_kmers,
                                True, False, 1, False, 0)

# Check distances match
run_regression(X1[:, 0], X2[:, 0])
run_regression(X1[:, 1], X2[:, 1])
