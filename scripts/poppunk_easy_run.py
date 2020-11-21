#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018-2020 John Lees and Nick Croucher

import sys
import argparse
import subprocess

# command line parsing
def get_options():

    parser = argparse.ArgumentParser(description='Easy run mode (create + dbscan + refine)',
                                     prog='easy_run')

    # input options
    parser.add_argument('--r-files', help='List of sequence names and files (as for --r-files')
    parser.add_argument('--output', help='Prefix for output files')

    parser.add_argument('--other-args', help="Other arguments to pass to poppunk. e.g. "
                                             "'--min-k 13 --max-k 29'")
    parser.add_argument('--poppunk-exe', help="Location of poppunk executable. Use "
                                              "'python poppunk-runner.py' to run from source tree")

    return parser.parse_args()

# main code
if __name__ == "__main__":

    # Check input ok
    args = get_options()

    if args.poppunk_exe is None:
        poppunk = "poppunk"
    else:
        poppunk = args.poppunk_exe

    if args.other_args is None:
        pp_args = ""
    else:
        pp_args = args.other_args

    sys.stderr.write("Running --create-db\n")
    create_db_cmd = poppunk + " --create-db --r-files " + args.r_files + " --output " + args.output + " " + pp_args
    sys.stderr.write(create_db_cmd + "\n")
    subprocess.run(create_db_cmd, shell=True, check=True)

    sys.stderr.write("Running --fit-model dbscan\n")
    dbscan_cmd = poppunk + " --fit-model dbscan --ref-db " + args.output + " --output " + args.output + " " + pp_args
    sys.stderr.write(dbscan_cmd + "\n")
    subprocess.run(dbscan_cmd, shell=True, check=True)

    sys.stderr.write("Running --fit-model refine\n")
    refine_cmd = poppunk + " --fit-model refine --ref-db " + args.output + " --output " + args.output + " " + pp_args
    sys.stderr.write(refine_cmd + "\n")
    subprocess.run(refine_cmd, shell=True, check=True)

    sys.exit(0)
