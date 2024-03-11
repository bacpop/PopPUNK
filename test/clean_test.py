#!/usr/bin/env python
# Copyright 2018 John Lees and Nick Croucher

"""Clean test files"""

import os
import sys
import shutil

def deleteDir(dirname):
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)

sys.stderr.write("Cleaning up tests\n")
dirty_files = ['example_db.info.csv']
with open("references.txt", 'r') as ref_file:
    for line in ref_file:
        dirty_files.append(line.rstrip().split("\t")[1])

# clean up
outputDirs = [
    "example_db",
    "example_qc",
    "example_dbscan",
    "example_refine",
    "example_threshold",
    "example_lineages",
    "example_use",
    "example_query",
    "example_single_query",
    "example_query_update",
    "example_lineage_query",
    "example_viz",
    "example_viz_subset",
    "example_viz_query",
    "example_lineage_viz",
    "example_viz_query_lineages",
    "example_mst",
    "example_sparse_mst",
    "example_mandrake",
    "example_iterate",
    "example_refs",
    "example_api",
    "batch1",
    "batch2",
    "batch3",
    "batch12",
    "batch123",
    "strain_1_lineage_db",
    "strain_2_lineage_db"
]
for outDir in outputDirs:
    deleteDir(outDir)

for ref in dirty_files:
    if os.path.isfile(ref):
        os.remove(ref)

