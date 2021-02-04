#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018-2021 John Lees and Nick Croucher

# universal
import os
import sys
import argparse
import subprocess
import shutil
import glob
import tempfile
from collections import defaultdict

rfile_names = "rlist.txt"

# command line parsing
def get_options():

    parser = argparse.ArgumentParser(description='Batch MST mode (create db + lineage model fit + assign + sparse_mst)',
                                     prog='poppunk_batch_mst')

    # input options
    ioGroup = parser.add_argument_group('Input and output file options')
    ioGroup.add_argument('--r-files', help="Sample names and locations (as for poppunk --r-files)",
                                      required=True)
    ioGroup.add_argument('--batch-file', help="Batches to process samples in --r-files in",
                                         required = True)
    ioGroup.add_argument('--output', help='Prefix for output files', required=True)
    ioGroup.add_argument('--previous-clustering', help='CSV file with previous clusters in MST drawing',
                                                  default=None)
    ioGroup.add_argument('--keep-intermediates', help='Retain the outputs of each batch',
                                                        default=False,
                                                        action='store_true')
    ioGroup.add_argument('--use-batch-names', help='Name the stored outputs of each batch',
                                                        default=False,
                                                        action='store_true')
    # analysis options
    aGroup = parser.add_argument_group('Analysis options')
    aGroup.add_argument('--rank', help='Comma separated ranks used to fit lineage model (list of ints)',
                                      type = str,
                                      default = "10")
    aGroup.add_argument('--threads', help='Number of threads for parallelisation (int)',
                                     type = int,
                                     default = 1)
    aGroup.add_argument('--use-gpu', help='Use GPU for analysis',
                                     default=False,
                                     action='store_true')
    aGroup.add_argument('--deviceid', help='GPU device ID (int)',
                                      type = int,
                                      default = 0)
    aGroup.add_argument('--db-args', help="Other arguments to pass to poppunk. e.g. "
                                             "'--min-k 13 --max-k 29'",
                                     default = "")
    aGroup.add_argument('--model-args', help="Other arguments to pass to lineage model fit",
                                        default = "")
    aGroup.add_argument('--assign-args', help="Other arguments to pass to poppunk_assign",
                                         default = "")

    # Executable options
    eGroup = parser.add_argument_group('Executable locations')
    eGroup.add_argument('--poppunk-exe', help="Location of poppunk executable. Use "
                                             "'python poppunk-runner.py' to run from source tree",
                                         default="poppunk")
    eGroup.add_argument('--assign-exe', help="Location of poppunk_assign executable. Use "
                                             "'python poppunk_assign-runner.py' to run from source tree",
                                        default="poppunk_assign")
    eGroup.add_argument('--mst-exe', help="Location of poppunk executable. Use "
                                           "'python poppunk_mst-runner.py' to run from source tree",
                                     default="poppunk_mst")

    return parser.parse_args()

def writeBatch(rlines, batches, batch_selected, use_names = False):
    tmpdir = ""
    if use_names:
        tmpdir = "./pp_mst_" + batch_selected
        os.mkdir(tmpdir)
    else:
        tmpdir = tempfile.mkdtemp(prefix="pp_mst", dir="./")
    with open(tmpdir + "/" + rfile_names, 'w') as outfile:
        for rline, batch in zip(rlines, batches):
            if batch == batch_selected:
                outfile.write(rline)

    return tmpdir

def runCmd(cmd_string):
    sys.stderr.write("Running command:\n")
    sys.stderr.write(cmd_string)
    subprocess.run(cmd_string, shell=True, check=True)

# main code
if __name__ == "__main__":

    # Check input ok
    args = get_options()
    if args.previous_clustering is not None and \
        not os.path.isfile(args.previous_clustering):
        sys.stderr.write("Provided --previous-clustering file cannot be found\n")
        sys.exit(1)

    # Extract ranks
    ranks = [int(rank) for rank in args.rank.split(',')]
    max_rank = max(ranks)

    # Check batching
    rlines = []
    batches = []
    with open(args.batch_file,'r') as batch_file:
        batches = [batch_line.rstrip() for batch_line in batch_file.readlines()]
    batch_names = sorted(set(batches))
    if len(batch_names) < 2:
        sys.stderr.write("You must supply multiple batches")
        sys.exit(1)
    first_batch = batch_names.pop(0)

    # Check input file
    with open(args.r_files,'r') as r_file:
        for r_line in r_file:
            rlines.append(r_line)


    # try/except block to clean up tmp files
    wd = writeBatch(rlines, batches, first_batch, args.use_batch_names)
    tmp_dirs = []
    try:
        # First batch is create DB + lineage
        create_db_cmd = args.poppunk_exe + " --create-db --r-files " + \
                                wd + "/" + rfile_names + \
                                " --output " + wd + " " + \
                                args.db_args + " --threads " + \
                                str(args.threads) + " " + \
                                args.db_args
        if args.use_gpu:
            create_db_cmd += " --gpu-sketch --gpu-dist --deviceid " + str(args.deviceid)
        runCmd(create_db_cmd)

        # Fit lineage model
        fit_model_cmd = args.poppunk_exe + " --fit-model lineage --ref-db " + \
                                wd + " --rank " + \
                                args.rank + " --threads " + \
                                str(args.threads) + " " + \
                                args.model_args
        runCmd(fit_model_cmd)

        for batch_idx, batch in enumerate(batch_names):
            batch_wd = writeBatch(rlines, batches, batch, args.use_batch_names)
            tmp_dirs.append(batch_wd)

            assign_cmd = args.assign_exe + " --db " + wd + \
                        " --query " + batch_wd + "/" + rfile_names + \
                        " --model-dir " + wd + " --output " + batch_wd + \
                        " --threads " + str(args.threads) + " --update-db " + \
                        args.assign_args
            if args.use_gpu:
                assign_cmd = assign_cmd + " --gpu-sketch --gpu-dist --deviceid " + str(args.deviceid)
            runCmd(assign_cmd)

            # Remove the previous batch
            if batch_idx > 0 and args.keep_intermediates == False:
                shutil.rmtree(tmp_dirs[batch_idx - 1])

        # Calculate MST
        output_dir = tmp_dirs[-1]
        mst_command = args.mst_exe + " --distance-pkl " + output_dir + \
                        "/" + os.path.basename(output_dir) + ".dists.pkl --rank-fit " + \
                        output_dir + "/" + os.path.basename(output_dir) + "_rank" + \
                        str(max_rank) +  "_fit.npz " + \
                        " --output " + args.output + \
                        " --threads " + str(args.threads)
        if args.previous_clustering is not None:
            mst_command = mst_command + " --previous-clustering " + args.previous_clustering
        else:
            mst_command = mst_command + " --previous-clustering " + \
                            os.path.join(output_dir,output_dir + "_lineages.csv")
        if args.use_gpu:
            mst_command = mst_command + " --gpu-graph"
        runCmd(mst_command)
        
        # Retrieve lineages from previous round
        os.rename(os.path.join(output_dir,os.path.basename(output_dir) + "_lineages.csv"),
                  os.path.join(args.output,os.path.basename(args.output) + "_lineages.csv"))
        
    except:
        if args.keep_intermediates == False:
            for tmpdir in tmp_dirs:
                shutil.rmtree(wd)
                shutil.rmtree(tmpdir)
        print("Unexpected error:", sys.exc_info()[0])
        raise

    if args.keep_intermediates == False:
        shutil.rmtree(wd)
        shutil.rmtree(output_dir)
