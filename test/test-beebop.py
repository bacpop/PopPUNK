#!/usr/bin/env python
# Copyright 2018-2024 John Lees and Nick Croucher

"""Tests for PopPUNK --update-db order"""

import subprocess
import os, sys
import sys
import shutil

if os.environ.get("POPPUNK_PYTHON"):
    python_cmd = os.environ.get("POPPUNK_PYTHON")
else:
    python_cmd = "python"

subprocess.run(python_cmd + " ../poppunk-runner.py --create-db --r-files rfile12.txt --output batch12 --overwrite", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model bgmm --D 2 --ref-db batch12 --overwrite", shell=True, check=True)
#subprocess.run(python_cmd + " ../poppunk-runner.py --create-db --r-files rfile3.txt --output batch3 --overwrite", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_assign-runner.py --db batch12 --query rfile3.txt --output batch3 --overwrite", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_visualise-runner.py --ref-db batch12 --query-db batch3 --output batch123_viz --external-clustering example_external_clusters.csv --cytoscape --rapidnj rapidnj --network-file ./batch12/batch12_graph.gt", shell=True, check=True)
