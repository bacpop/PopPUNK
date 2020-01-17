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
refs = []
with open("references.txt", 'r') as ref_file:
    for line in ref_file:
        refs.append(line.rstrip().split("\t")[1])

# clean up
outputDirs = [
    "example_db",
    "example_refine",
    "example_query",
    "example_use",
    "example_db_mash",
    "example_refine_mash",
    "example_query_mash",
    "example_use_mash", 
    "example_viz"
]
for outDir in outputDirs:
    deleteDir(outDir)

for ref in refs:
    os.remove(ref)

