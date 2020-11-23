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

    parser.add_argument('--analysis-args', help="Other arguments to pass to poppunk. e.g. "
                                             "'--min-k 13 --max-k 29'")
    parser.add_argument('--viz', help = "Run visualisation of output", default = False, action = "store_true")
    parser.add_argument('--viz-args', help = "Options to use for visualisation")
    parser.add_argument('--poppunk-exe', help="Location of poppunk executable. Use "
                                              "'python poppunk-runner.py' to run from source tree")
    parser.add_argument('--viz-exe', help = "Location of poppunk_visualisation executable")

    return parser.parse_args()

# main code
if __name__ == "__main__":

    # Check input ok
    args = get_options()

    if args.poppunk_exe is None:
        poppunk = "poppunk"
    else:
        poppunk = args.poppunk_exe

    if args.analysis_args is None:
        pp_args = ""
    else:
        pp_args = args.analysis_args

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

    if args.viz:

        if args.viz_exe is None:
            poppunk_viz = "poppunk_visualise"
        else:
            poppunk_viz = args.viz_exe

        if args.viz_args is None:
            viz_extra = ""
        else:
            viz_extra = args.viz_args

        viz_cmd = poppunk_viz + " --ref-db " + args.output + " --output " + args.output + " " + viz_extra
        sys.stderr.write(viz_cmd + "\n")
        subprocess.run(viz_cmd, shell=True, check=True)

    sys.exit(0)
