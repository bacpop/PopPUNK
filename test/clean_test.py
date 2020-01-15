#!/usr/bin/env python
# Copyright 2018 John Lees and Nick Croucher

"""Clean test files"""

import os
import sys
import shutil

sys.stderr.write("Cleaning up tests\n")
refs = []
with open("references.txt", 'r') as ref_file:
    for line in ref_file:
        refs.append(line.rstrip().split("\t")[1])

# clean up
if os.path.isdir("example_db"):
    shutil.rmtree("example_db")
if os.path.isdir("example_refine"):
    shutil.rmtree("example_refine")
if os.path.isdir("example_query"):
    shutil.rmtree("example_query")
if os.path.isdir("example_viz"):
    shutil.rmtree("example_viz")
if os.path.isdir("example_use"):
    shutil.rmtree("example_use")
for ref in refs:
    os.remove(ref)

