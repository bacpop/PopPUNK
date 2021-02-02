#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018-2020 John Lees and Nick Croucher

# universal
import os
import sys
import argparse
import subprocess
import shutil
import glob
from collections import defaultdict

def write_batch(batched_sequences, files, batch, output):
    out_fn = output + '.' + batch + '.list'
    with open(out_fn,'w') as out_file:
        for seq in batched_sequences[batch]:
            out_file.write(seq + "\t" + files[seq] + "\n")
    return out_fn

# command line parsing
def get_options():

    parser = argparse.ArgumentParser(description='Batch MST mode (create db + lineage model fit + assign)',
                                     prog='batch_mst')

    # input options
    ioGroup = parser.add_argument_group('Input and output file options')
    ioGroup.add_argument('--batch-file', help='Tab-separated list of sequence names, files '
                                              'and batch assignments',
                                              required = True)
    ioGroup.add_argument('--batch-order', help='File specifying order in which batches should '
                                              'be processed')
    ioGroup.add_argument('--keep-intermediates', help='Retain the outputs of each batch',
                                                        default=False,
                                                        action='store_true')
    ioGroup.add_argument('--output', help='Prefix for output files', required=True)
    
    # analysis options
    aGroup = parser.add_argument_group('Analysis options')
    aGroup.add_argument('--rank', help='Rank used to fit lineage model (int)',
                                               type = int,
                                               default = 1)
    aGroup.add_argument('--threads', help='Number of threads for parallelisation (int)',
                                              type = int,
                                              default = 1)
    aGroup.add_argument('--gpu', help='Use GPU for analysis',
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
                                            "'python poppunk-runner.py' to run from source tree")
    eGroup.add_argument('--assign-exe', help="Location of poppunk executable. Use "
                                            "'python poppunk-runner.py' to run from source tree")
    eGroup.add_argument('--mst-exe', help="Location of poppunk executable. Use "
                                            "'python poppunk-runner.py' to run from source tree")

    return parser.parse_args()

# main code
if __name__ == "__main__":

    # Check input ok
    args = get_options()

    # Get poppunk executable
    if args.poppunk_exe is None:
        poppunk = "poppunk"
    else:
        poppunk = args.poppunk_exe
    # Need to add poppunk_assign_exe

    # Check input file and batching
    batch_set = set()
    files = {}
    batched_sequences = defaultdict(list)
    with open(args.batch_file,'r') as input_file:
        for line in input_file.readlines():
            info = line.rstrip().split()
            files[info[0]] = info[1]
            batch_set.add(info[2])
            batched_sequences[info[2]].append(info[0])

    # Check on batch order
    batches = []
    if args.batch_order is not None:
        with open(args.batch_order,'r') as order_file:
            batches = [line for line in input_file.readlines().rstrip()]
        if set(batches) != batch_set:
            batch_discrepancies = set(batches).difference(batch_set) + \
                                    batch_set.difference(set(batches))
            sys.stderr.write('Discrepancies between input file and batch '
            'ordering: ' + str(batch_discrepancies) + '\n')
            sys.exit()
    else:
        batches = list(batch_set)

    # Iterate through batches
    first_batch = True
    current_dir = args.output
    for batch in batches:
        # Write batch file
        batch_fn = write_batch(batched_sequences, files, batch, args.output)
        if first_batch:
            # Initialise database
            create_db_cmd = poppunk + " --create-db --r-files " + batch_fn + " --output " + args.output + " " + args.db_args + " --threads " + str(args.threads) + " " + args.db_args
            if args.gpu:
                create_db_cmd = create_db_cmd + " --gpu-sketch --gpu-dist --deviceid " + str(args.deviceid)
            sys.stderr.write(create_db_cmd + "\n")
            subprocess.run(create_db_cmd, shell=True, check=True)
            # Fit lineage model
            fit_model_cmd = poppunk + " --fit-model lineage --ref-db " + args.output + " --rank " + str(args.rank) + " --threads " + str(args.threads) + " " + args.model_args
            sys.stderr.write(fit_model_cmd + "\n")
            subprocess.run(fit_model_cmd, shell=True, check=True)
            # Completed first batch
            first_batch = False
        else:
            # Define batch prefix
            batch_prefix = args.output + "_" + batch
            # Add to first batch through querying
            assign_cmd = "poppunk_assign --db " + current_dir + " --query " + batch_fn + " --model-dir " + args.output + " --output " + batch_prefix + " --threads " + str(args.threads) + " --update-db " + args.assign_args
            if args.gpu:
                assign_cmd = assign_cmd + " --gpu-dist --deviceid " + str(args.deviceid)
            sys.stderr.write(assign_cmd + "\n")
            subprocess.run(assign_cmd, shell=True, check=True)
            # Process output
            if args.keep_intermediates:
#                shutil.rmtree(batch_prefix)
                current_dir = batch_prefix
                print("Switch current dir to " + current_dir)
            else:
                for file in glob.glob(args.output + "_" + batch + "/*"):
                    file_basename = os.path.basename(file)
                    if file_basename.startswith(batch_prefix):
                        print("Moving file " + args.output + "_" + batch + '/' + file_basename + " to " + current_dir + '/' + file_basename.replace(batch_prefix,args.output))
                        os.rename(args.output + "_" + batch + '/' + file_basename,
                                  current_dir + '/' + file_basename.replace(batch_prefix,args.output))
                shutil.rmtree(args.output + "_" + batch)
                    
        # Remove npy dist file
#        os.remove(args.output + "/" + args.output + ".dists.npy")

    # Calculate MST
    mst_command = "poppunk_mst --distances " + args.output + "/" + args.output + ".dists --rank-fit " + args.output + "/" + args.output + "_rank" + str(args.rank) + "_fit.npz --previous-clustering " + args.output + "/" + args.output + "_lineages.csv --output " + args.output + " --threads " + str(args.threads)
    if args.gpu:
        mst_command = mst_command + " --gpu-network"
