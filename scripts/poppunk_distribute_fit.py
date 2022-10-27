#!/usr/bin/env python
# Copyright 2022-2023 John Lees, Nick Croucher and Samuel Horsfield

import shutil
import os
import sys
import argparse
import tarfile
import re

def get_options():
    description = 'Generates distributable fits from PopPUNK'
    parser = argparse.ArgumentParser(description=description, prog='python poppunk_distribute_fit.py')

    IO = parser.add_argument_group('Input/Output options')
    IO.add_argument('--dbdir', required=True, help='PopPUNK Database Directory. ')
    IO.add_argument('--fitdir', required=True, help='PopPUNK fit Directory. ')
    IO.add_argument('--outpref', default="PopPUNK", help='Output file prefix. [Default = "PopPUNK"]')
    IO.add_argument('--lineage', default=False, action="store_true", help='Specify if lineage used for fit. [Default = False]')
    IO.add_argument('--no-compress', default=False, action="store_true", help='No compression of fits. [Default = False] ')


    return parser.parse_args()

if __name__ == "__main__":
    options = get_options()

    db_dir = options.dbdir
    fit_dir = options.fitdir
    out_full = options.outpref + "_full"
    out_refs = options.outpref + "_refs"
    lineage = options.lineage


    # ensure trailing slash present
    db_dir = os.path.join(db_dir, "")
    fit_dir = os.path.join(fit_dir, "")
    out_full = os.path.join(out_full, "")
    out_refs = os.path.join(out_refs, "")

    if not os.path.exists(out_full):
        os.mkdir(out_full)

    if not os.path.exists(out_refs):
        os.mkdir(out_refs)

    # get absolute paths
    db_dir = os.path.abspath(db_dir)
    fit_dir = os.path.abspath(fit_dir)
    out_full = os.path.abspath(out_full)
    out_refs = os.path.abspath(out_refs)

    # check if directories are real
    dir_check = True
    for dir in (db_dir, fit_dir, out_full, out_refs):
        if not os.path.isdir(dir):
            print("Directory {} not found".format(dir))
            dir_check = False

    if not dir_check:
        sys.exit(1)

    # database extensions
    db_exts = (".dists.npy", ".dists.pkl", ".h5", ".png", "_qcreport.txt")
    if lineage:
        fit_exts = ("_fit.npz", "_fit.pkl", "_graph.gt", ".csv", ".png", "rank_k_fit.npz")
    else:
        fit_exts = ("_fit.npz", "_fit.pkl", "_graph.gt", ".csv", ".png")


    #set current dir
    curr_dir = db_dir
    out_dir = out_full

    # get files in db_dir
    onlyfiles = [os.path.join(curr_dir, f) for f in os.listdir(curr_dir) if os.path.isfile(os.path.join(curr_dir, f))]
    for file in onlyfiles:
        if any(s in file for s in db_exts):
            shutil.copy(file, out_dir)
            new_name = re.sub(rf"^{db_dir}", out_dir, os.path.basename(file))
            os.rename(os.path.join(out_dir, file), os.path.join(out_dir, new_name))

    # get files in fit_dir
    curr_dir = fit_dir

    onlyfiles = [os.path.join(curr_dir, f) for f in os.listdir(curr_dir) if os.path.isfile(os.path.join(curr_dir, f))]
    for file in onlyfiles:
        if any(s in file for s in fit_exts):
            shutil.copy(file, out_dir)
            new_name = re.sub(rf"^{db_dir}", out_dir, os.path.basename(file))
            os.rename(os.path.join(out_dir, file), os.path.join(out_dir, new_name))

    # repeat for refs, will be in fit_dir
    out_dir = out_refs

    if lineage:
        fit_exts = ("_fit.npz", "_fit.pkl", ".csv", ".png", "_qcreport.txt", "rank_k_fit.npz")
    else:
        fit_exts = ("_fit.npz", "_fit.pkl", ".csv", ".png", "_qcreport.txt")

    # get files in db_dir
    onlyfiles = [os.path.join(curr_dir, f) for f in os.listdir(curr_dir) if os.path.isfile(os.path.join(curr_dir, f))]
    for file in onlyfiles:
        #print(file)
        if ".refs" in file:
            new_name = re.sub(rf"^{db_dir}", out_dir, os.path.basename(file))
            new_name = re.sub(rf"\.refs\.", ".", new_name)
            os.rename(os.path.join(out_dir, file), os.path.join(out_dir, new_name))


    # get files in fit_dir
    onlyfiles = [os.path.join(curr_dir, f) for f in os.listdir(curr_dir) if os.path.isfile(os.path.join(curr_dir, f))]
    for file in onlyfiles:
        #print(file)
        if any(s in file for s in fit_exts):
            shutil.copy(file, out_dir)
            new_name = re.sub(rf"^{db_dir}", out_dir, os.path.basename(file))
            os.rename(os.path.join(out_dir, file), os.path.join(out_dir, new_name))

    # compress fits
    if not options.no_compress:
        # compress refs
        tar_out_list = os.path.split(out_dir)
        tar_out = os.path.join(tar_out_list[0], tar_out_list[1] + ".tar.bz2")
        #print(tar_out)
        onlyfiles = [os.path.join(tar_out_list[1], f) for f in os.listdir(out_dir) if os.path.isfile(os.path.join(out_dir, f))]
        os.chdir(tar_out_list[0])
        with tarfile.open(tar_out, "w:bz2") as tar:
            for file in onlyfiles:
                tar.add(file)

        # compress full
        out_dir = out_full
        tar_out_list = os.path.split(out_dir)
        tar_out = os.path.join(tar_out_list[0], tar_out_list[1] + ".tar.bz2")
        #print(tar_out)
        onlyfiles = [os.path.join(tar_out_list[1], f) for f in os.listdir(out_dir) if os.path.isfile(os.path.join(out_dir, f))]
        os.chdir(tar_out_list[0])
        with tarfile.open(tar_out, "w:bz2") as tar:
            for file in onlyfiles:
                tar.add(file)


    sys.exit(0)
