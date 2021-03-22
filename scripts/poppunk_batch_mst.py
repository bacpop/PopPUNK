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
import pandas as pd

rfile_names = "rlist.txt"

# command line parsing
def get_options():

    parser = argparse.ArgumentParser(description='Batch MST mode (create db + lineage model fit + assign + sparse_mst)',
                                     prog='poppunk_batch_mst')

    # input options
    ioGroup = parser.add_argument_group('Input and output file options')
    ioGroup.add_argument('--r-files', help='Sample names and locations (as for poppunk --r-files)',
                                                required=True)
    ioGroup.add_argument('--batch-file', help='Single column list of batches to process samples in --r-files in')
    ioGroup.add_argument('--n-batches', help='Number of batches for process if --batch-file is not specified',
                                                type=int,
                                                default=10)
    ioGroup.add_argument('--info-csv', help='CSV containing information about sequences', default=None)
    ioGroup.add_argument('--output', help='Prefix for output files',
                                                required=True)
    ioGroup.add_argument('--previous-clustering', help='CSV file with previous clusters in MST drawing',
                                                default=None)
    ioGroup.add_argument('--iterative-mst', help='Re-calculate the MST for each batch',
                                                default=False,
                                                action='store_true')
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
    aGroup.add_argument('--gpu-dist', help='Use GPU for distance calculations',
                                                default=False,
                                                action='store_true')
    aGroup.add_argument('--gpu-graph', help='Use GPU for network analysis',
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

    # QC options
    qcGroup = parser.add_argument_group('Quality control options for distances')
    qcGroup.add_argument('--qc-filter', help='Behaviour following sequence QC step: "stop" [default], "prune"'
                                                ' (analyse data passing QC), or "continue" (analyse all data)',
                                                default='stop', type = str, choices=['stop', 'prune', 'continue'])
    qcGroup.add_argument('--retain-failures', help='Retain sketches of genomes that do not pass QC filters in '
                                                'separate database [default = False]', default=False, action='store_true')
    qcGroup.add_argument('--max-a-dist', help='Maximum accessory distance to permit [default = 0.5]',
                                                default = 0.5, type = float)
    qcGroup.add_argument('--length-sigma', help='Number of standard deviations of length distribution beyond '
                                                'which sequences will be excluded [default = 5]', default = None, type = int)
    qcGroup.add_argument('--length-range', help='Allowed length range, outside of which sequences will be excluded '
                                                '[two values needed - lower and upper bounds]', default=[None,None],
                                                type = int, nargs = 2)
    qcGroup.add_argument('--prop-n', help='Threshold ambiguous base proportion above which sequences will be excluded'
                                                ' [default = 0.1]', default = None,
                                                type = float)
    qcGroup.add_argument('--upper-n', help='Threshold ambiguous base count above which sequences will be excluded',
                                                default=None, type = int)

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
        tmpdir = "./pp_mst_" + str(batch_selected)
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
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
    sys.stderr.write(cmd_string + '\n')
    subprocess.run(cmd_string, shell=True, check=True)

def readLineages(clustCSV):
    clusters = defaultdict(dict)
    # read CSV
    clustersCsv = pd.read_csv(clustCSV, index_col = 0, quotechar='"')
    # select relevant columns
    type_columns = [n for n,col in enumerate(clustersCsv.columns) if ('Rank_' in col or 'overall' in col)]
    # read file
    for row in clustersCsv.itertuples():
        for cls_idx in type_columns:
            cluster_name = clustersCsv.columns[cls_idx]
            cluster_name = cluster_name.replace('__autocolour','')
            clusters[cluster_name][row.Index] = str(row[cls_idx + 1])
    # return data structure
    return clusters

def isolateNameToLabel(names):
    labels = [name.split('/')[-1].split('.')[0] for name in names]
    return labels

def writeClusterCsv(outfile, nodeNames, nodeLabels, clustering,
                epiCsv = None, suffix = '_Lineage'):
    # set order of column names
    colnames = ['ID']
    for cluster_type in clustering:
        col_name = cluster_type + suffix
        colnames.append(col_name)
    # process epidemiological data
    d = defaultdict(list)
    # process epidemiological data without duplicating names
    # used by PopPUNK
    columns_to_be_omitted = ['id', 'Id', 'ID', 'combined_Cluster__autocolour',
    'core_Cluster__autocolour', 'accessory_Cluster__autocolour',
    'overall_Lineage']
    if epiCsv is not None:
        epiData = pd.read_csv(epiCsv, index_col = False, quotechar='"')
        epiData.index = isolateNameToLabel(epiData.iloc[:,0])
        for e in epiData.columns.values:
            if e not in columns_to_be_omitted:
                colnames.append(str(e))
    # get example clustering name for validation
    example_cluster_title = list(clustering.keys())[0]
    for name, label in zip(nodeNames, isolateNameToLabel(nodeLabels)):
        if name in clustering[example_cluster_title]:
            d['ID'].append(label)
            for cluster_type in clustering:
                col_name = cluster_type + suffix
                d[col_name].append(clustering[cluster_type][name])
            if epiCsv is not None:
                if label in epiData.index:
                    for col, value in zip(epiData.columns.values, epiData.loc[label].values):
                        if col not in columns_to_be_omitted:
                            d[col].append(str(value))
                else:
                    for col in epiData.columns.values:
                        if col not in columns_to_be_omitted:
                            d[col].append('nan')
        else:
            sys.stderr.write("Cannot find " + name + " in clustering\n")
            sys.exit(1)
    # print CSV
    sys.stderr.write("Parsed data, now writing to CSV\n")
    try:
        pd.DataFrame(data=d).to_csv(outfile, columns = colnames, index = False)
    except subprocess.CalledProcessError as e:
        sys.stderr.write("Problem with epidemiological data CSV; returned code: " + str(e.returncode) + "\n")
        # check CSV
        prev_col_items = -1
        prev_col_name = "unknown"
        for col in d:
            this_col_items = len(d[col])
            if prev_col_items > -1 and prev_col_items != this_col_items:
                sys.stderr.write("Discrepant length between " + prev_col_name + \
                                 " (length of " + prev_col_items + ") and " + \
                                 col + "(length of " + this_col_items + ")\n")
        sys.exit(1)

# main code
if __name__ == "__main__":

    ###########
    # Prepare #
    ###########

    # Check input ok
    args = get_options()
    if args.previous_clustering is not None and \
        not os.path.isfile(args.previous_clustering):
        sys.stderr.write("Provided --previous-clustering file cannot be found\n")
        sys.exit(1)

    # Extract ranks
    ranks = [int(rank) for rank in args.rank.split(',')]
    max_rank = max(ranks)

    # Check input file
    rlines = []
    nodeNames = []
    nodeLabels = []
    with open(args.r_files,'r') as r_file:
        for r_line in r_file:
            rlines.append(r_line)
            node_info = r_line.rstrip().split()
            nodeNames.append(node_info[0])
            nodeLabels.append(node_info[1])

    # Check batching
    batches = []
    if args.batch_file:
        # Read specified batches
        with open(args.batch_file,'r') as batch_file:
            batches = [batch_line.rstrip() for batch_line in batch_file.readlines()]
    else:
        # Generate arbitrary batches
        x = 0
        n = 1
        while x < len(rlines):
            if n > args.n_batches:
                n = 1
            batches.append(n)
            n = n + 1
            x = x + 1
    # Validate batches
    batch_names = sorted(set(batches))
    if len(batch_names) < 2:
        sys.stderr.write("You must supply multiple batches\n")
        sys.exit(1)
    first_batch = batch_names.pop(0)

    # try/except block to clean up tmp files
    wd = writeBatch(rlines, batches, first_batch, args.use_batch_names)
    tmp_dirs = [wd]
    try:
    
        ###############
        # First batch #
        ###############
    
        # First batch is create DB + lineage
        create_db_cmd = args.poppunk_exe + " --create-db --r-files " + \
                                wd + "/" + rfile_names + \
                                " --output " + wd + " " + \
                                args.db_args + " --threads " + \
                                str(args.threads) + " " + \
                                args.db_args
        # QC options
        if None not in args.length_range:
            create_db_cmd += " --length-range " + str(args.length_range[0]) + " " + str(args.length_range[1])
        elif args.length_sigma is not None:
            create_db_cmd += " --length-sigma " + str(args.length_sigma)
        if args.upper_n is not None:
            create_db_cmd += " --upper-n " + str(args.upper_n)
        elif args.prop_n is not None:
            create_db_cmd += " --prop-n " + str(args.prop_n)
        create_db_cmd += " --qc-filter " + args.qc_filter
        # GPU options
        if args.gpu_dist:
            create_db_cmd += " --gpu-dist --deviceid " + str(args.deviceid)
        runCmd(create_db_cmd)

        # Fit lineage model
        fit_model_cmd = args.poppunk_exe + " --fit-model lineage --ref-db " + \
                                wd + " --rank " + \
                                args.rank + " --threads " + \
                                str(args.threads) + " " + \
                                args.model_args
        runCmd(fit_model_cmd)
        
        # Calculate MST if operating iteratively
        if args.iterative_mst:
        
            mst_command = args.mst_exe + " --distance-pkl " + wd + \
                            "/" + os.path.basename(wd) + ".dists.pkl --rank-fit " + \
                            wd + "/" + os.path.basename(wd) + "_rank" + \
                            str(max_rank) +  "_fit.npz " + \
                            " --output " + wd + \
                            " --threads " + str(args.threads) + \
                            " --previous-clustering " + wd + \
                            "/" + os.path.basename(wd) + "_lineages.csv"
            # GPU options
            if args.gpu_graph:
                mst_command = mst_command + " --gpu-graph"
            runCmd(mst_command)
            
        ###########
        # Iterate #
        ###########
        
        for batch_idx, batch in enumerate(batch_names):
            prev_wd = tmp_dirs[-1]
            batch_wd = writeBatch(rlines, batches, batch, args.use_batch_names)
            tmp_dirs.append(batch_wd)

            assign_cmd = args.assign_exe + " --db " + prev_wd + \
                        " --query " + batch_wd + "/" + rfile_names + \
                        " --model-dir " + prev_wd + " --output " + batch_wd + \
                        " --threads " + str(args.threads) + " --update-db " + \
                        args.assign_args
            # QC options
            if None not in args.length_range:
                assign_cmd += " --length-range " + str(args.length_range[0]) + " " + str(args.length_range[1])
            elif args.length_sigma is not None:
                assign_cmd += " --length-sigma " + str(args.length_sigma)
            else:
                assign_cmd += " --length-sigma 5" # default from __main__
            if args.upper_n is not None:
                create_db_cmd += " --upper-n " + str(args.upper_n)
            elif args.prop_n is not None:
                assign_cmd += " --prop-n " + str(args.prop_n)
            else:
                assign_cmd += " --prop-n 0.1" # default from __main__
            assign_cmd += " --qc-filter " + args.qc_filter
            # GPU options
            if args.gpu_dist:
                assign_cmd = assign_cmd + " --gpu-dist --deviceid " + str(args.deviceid)
            runCmd(assign_cmd)
            
            # Calculate MST if operating iteratively
            if args.iterative_mst:
            
                mst_command = args.mst_exe + " --distance-pkl " + batch_wd + \
                                "/" + os.path.basename(batch_wd) + ".dists.pkl --rank-fit " + \
                                batch_wd + "/" + os.path.basename(batch_wd) + "_rank" + \
                                str(max_rank) +  "_fit.npz " + \
                                " --output " + batch_wd + \
                                " --threads " + str(args.threads) + \
                                " --previous-mst " + \
                                prev_wd + "/" + os.path.basename(prev_wd) + ".graphml" + \
                                " --previous-clustering " + batch_wd + \
                                "/" + os.path.basename(batch_wd) + "_lineages.csv"
                if args.gpu_graph:
                    mst_command = mst_command + " --gpu-graph"
                runCmd(mst_command)

            # Remove the previous batch
            if batch_idx > 0 and args.keep_intermediates == False:
                shutil.rmtree(tmp_dirs[batch_idx - 1])

        ##########
        # Finish #
        ##########

        # Calculate MST
        output_dir = tmp_dirs[-1]
        if args.iterative_mst:
            # Create directory
            if os.path.exists(args.output):
                if os.path.isdir(args.output):
                    shutil.rmtree(args.output)
                else:
                    os.remove(args.output)
            os.mkdir(args.output)
            # Copy over final MST
            shutil.copy(os.path.join(output_dir,os.path.basename(output_dir) + ".graphml"),
                        os.path.join(args.output,os.path.basename(args.output) + ".graphml"))
            shutil.copy(os.path.join(output_dir,os.path.basename(output_dir) + "_MST.nwk"),
            os.path.join(args.output,os.path.basename(args.output) + "_MST.nwk"))
        else:
            # Calculate MST
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
                                os.path.join(output_dir,os.path.basename(output_dir) + "_lineages.csv")
            if args.gpu_graph:
                mst_command = mst_command + " --gpu-graph"
            runCmd(mst_command)
        
        # Retrieve isolate names and lineages from previous round
        os.rename(os.path.join(output_dir,os.path.basename(output_dir) + ".dists.pkl"),
                  os.path.join(args.output,os.path.basename(args.output) + ".dists.pkl"))
        os.rename(os.path.join(output_dir,os.path.basename(output_dir) + "_lineages.csv"),
                  os.path.join(args.output,os.path.basename(args.output) + "_lineages.csv"))
        for rank in ranks:
            os.rename(os.path.join(output_dir, os.path.basename(output_dir) + "_rank" + str(rank) + "_fit.npz"),
                      os.path.join(args.output, os.path.basename(args.output) + "_rank" + str(rank) + "_fit.npz"))

        # Merge with epidemiological data if requested
        if args.info_csv is not None:
            lineage_clustering = readLineages(os.path.join(args.output,
                                            os.path.basename(args.output) + "_lineages.csv"))
            writeClusterCsv(os.path.join(args.output,
                                            os.path.basename(args.output)  + "_info.csv"),
                            nodeNames,
                            nodeLabels,
                            lineage_clustering,
                            epiCsv = args.info_csv)

    except:
        if args.keep_intermediates == False:
            for tmpdir in tmp_dirs:
                try:
                    shutil.rmtree(tmpdir)
                except:
                    sys.stderr.write("Unable to remove " + tmpdir + "\n")
        print("Unexpected error:", sys.exc_info()[0])
        raise

    if args.keep_intermediates == False:
        shutil.rmtree(output_dir)
