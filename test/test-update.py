#!/usr/bin/env python
# Copyright 2018-2020 John Lees and Nick Croucher

"""Tests for PopPUNK"""

import subprocess
import os
import sys
import shutil

import numpy as np
from scipy import stats

if os.environ.get("POPPUNK_PYTHON"):
    python_cmd = os.environ.get("POPPUNK_PYTHON")
else:
    python_cmd = "python"

def run_regression(x, y, threshold = 0.99):
    res = stats.linregress(x, y)
    if res.rvalue**2 < threshold:
        sys.stderr.write("Dist order failed: R^2 = " + str(res.rvalue**2) + "\n")
        sys.exit(1)

# Check that order is the same after doing 1 + 2 with --update-db, as doing all of 1 + 2 together
subprocess.run(python_cmd + " ../poppunk-runner.py --create-db --r-files rfile12.txt --output batch12 --overwrite", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --create-db --r-files rfile1.txt --output batch1 --overwrite", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model lineage --ref-db batch1 --ranks 1", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_assign-runner.py --db batch1 --query rfile2.txt --output batch2 --update-db --overwrite", shell=True, check=True)

X1 = np.load("batch12/batch12.dists.npy")
X2 = np.load("batch2/batch2.dists.npy")

run_regression(X1[:, 0], X2[:, 0])
run_regression(X1[:, 1], X2[:, 1])


