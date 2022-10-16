#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018-2020 John Lees and Nick Croucher

import os
import sys
import argparse
import subprocess
import pickle
import pandas as pd
from collections import defaultdict

import pp_sketchlib
from .network import construct_network_from_edge_list, printClusters, save_network
from .models import LineageFit
from .plot import writeClusterCsv
from .sketchlib import readDBParams
from .qc import prune_distance_matrix
from .utils import createOverallLineage, readPickle

# command line parsing
def get_options():

    parser = argparse.ArgumentParser(description='Generate script and databases for lineage clustering across strains',
                                     prog='lineages_within_strains')

    # input options
    ioGroup = parser.add_argument_group('Input and output files')
    ioGroup.add_argument('--db',         help="PopPUNK strain database")
    ioGroup.add_argument('--distances',
                                        help="Distance file prefix (if not in database)")
    ioGroup.add_argument('--clustering-file',
                                        help="Clustering file (if not in database)")
    ioGroup.add_argument('--clustering-col-name',
                                        help="Clustering column name [default: 'Cluster']",
                                        default="Cluster")
    ioGroup.add_argument('--lineage-db-prefix',
                                        help="Prefix used for new database directories",
                                        default="strain")
    ioGroup.add_argument('--write-networks',
                                        help="Write networks for lineages",
                                        default=False,
                                        action='store_true')
    ioGroup.add_argument('--output',    help = "Output script file for analysing by strain, then lineage",
                                        required=True)
    
    aGroup = parser.add_argument_group('Analysis options')
    aGroup.add_argument('--poppunk_exe',help="Path to PopPUNK executable if not on path")
    aGroup.add_argument('--threads',    help="Number of CPUs to use in analysis",
                                        default=1,
                                        type=int)
    aGroup.add_argument('--use-gpu',    help="Use GPU for analyses",
                                        default=False,
                                        action='store_true')
    aGroup.add_argument('--deviceid',   help="Device ID of GPU",
                                        default=0,
                                        type=int)
    
    lGroup = parser.add_argument_group('Lineage model options')
    lGroup.add_argument('--ranks',      help="Comma separated list of ranks used in lineage clustering",
                                    default = "1,2,3",
                                    type = str)
    lGroup.add_argument('--max-search-depth',
                                    help="Number of kNN distances per sequence to filter when "
                                    "counting neighbours or using only reciprocal matches",
                                    type = int,
                                    default = None)
    lGroup.add_argument('--use-accessory',
                                    help="Use accessory distances for lineage clustering",
                                    action = 'store_true',
                                    default = False)
    lGroup.add_argument('--min-count',  help="Minimum number of isolates in a strain for lineages to be defined",
                                    default = 10,
                                    type = int)
    lGroup.add_argument('--count-unique-distances',
                                    help = "kNN enumerates number of unique distances rather than number of neighbours",
                                    action = 'store_true',
                                    default = False)
    lGroup.add_argument('--reciprocal-only',
                                    help="Only use reciprocal kNN matches for lineage definitions",
                                    action = 'store_true',
                                    default = False)

    return parser.parse_args()

# main code
def main():

    # Check input ok
    args = get_options()

    if args.poppunk_exe is None:
        poppunk = "poppunk"
    else:
        poppunk = args.poppunk_exe

    sys.stderr.write("Identifying strains in existing database\n")
    # Read in strain information
    strains = {}
    if args.clustering_file is None:
        clustering_file = os.path.join(args.db,os.path.basename(args.db) + '_clusters.csv')
    else:
        clustering_file = args.clustering_file
    strains = pd.read_csv(clustering_file).groupby(args.clustering_col_name)
    
    sys.stderr.write("Extracting properties of database\n")
    # Get rlist
    if args.distances is None:
        distances = os.path.join(args.db,os.path.basename(args.db) + ".dists")
    else:
        distances = args.distances
    # Get distances
    rlist, qlist, self, X = readPickle(distances, enforce_self=False, distances=True)
    # Get parameters
    kmers, sketch_sizes, codon_phased = readDBParams(args.db)
    # Ranks to use
    rank_list = [int(x) for x in args.ranks.split(',')]
    if args.max_search_depth is not None:
        if args.max_search_depth <= max(rank_list):
            sys.stderr.write("Max search depth must be greater than the highest lineage rank\n")
            sys.exit(1)
        else:
            max_search_depth = args.max_search_depth
    else:
        max_search_depth = max([int(0.1*len(rlist)),int(1.1*max(rank_list)),int(1+max(rank_list))])

    sys.stderr.write("Generating databases for individual strains\n")
    for strain,isolates in strains:
      # Make new database directory
      sys.stderr.write("Making database for strain " + str(strain) + "\n")
      strain_db_name = args.lineage_db_prefix + '_' + str(strain) + '_lineage_db'
      isolate_list = isolates[isolates.columns.values[0]].to_list()
      num_isolates = len(isolate_list)
      if num_isolates >= args.min_count:
        if not os.path.isdir(strain_db_name):
            try:
                os.makedirs(strain_db_name)
            except OSError:
                sys.stderr.write("Cannot create output directory\n")
                sys.exit(1)
        # Extract sparse distances
        prune_distance_matrix(rlist,
                        list(set(rlist) - set(isolate_list)),
                        X,
                        os.path.join(strain_db_name,strain_db_name + '.dists'))
        # Initialise model
        model = LineageFit(strain_db_name,
                  rank_list,
                  max_search_depth,
                  args.reciprocal_only,
                  args.count_unique_distances,
                  use_gpu = args.use_gpu)
        model.set_threads(args.threads)
        # Load pruned distance matrix
        strain_rlist, strain_qlist, strain_self, strain_X = \
                            readPickle(os.path.join(strain_db_name,strain_db_name + '.dists'),
                                        enforce_self=False,
                                        distances=True)
        # Fit model
        model.fit(strain_X,
                    args.use_accessory)
        # Lineage fit requires some iteration
        assignments = {}
        indivNetworks = {}
        lineage_clusters = defaultdict(dict)
        # Iterate over ranks
        for rank in rank_list:
            if rank <= num_isolates:
                assignments[rank] = \
                    model.assign(rank)
            # Generate networks
            indivNetworks[rank] = construct_network_from_edge_list(strain_rlist,
                                                        strain_rlist,
                                                        assignments[rank],
                                                        weights = None,
                                                        betweenness_sample = None,
                                                        use_gpu = args.use_gpu,
                                                        summarise = False
                                                       )
            # Write networks
            if args.write_networks:
                save_network(indivNetworks[rank],
                        prefix = output,
                        suffix = '_rank_' + str(rank) + '_graph',
                        use_gpu = args.use_gpu)
            # Identify clusters from output
            lineage_clusters[rank] = \
                printClusters(indivNetworks[rank],
                                                                                      strain_rlist,
                              printCSV = False,
                              use_gpu = args.use_gpu)
            n_clusters = max(lineage_clusters[rank].values())
            sys.stderr.write("Network for rank " + str(rank) + " has " +
                             str(n_clusters) + " lineages\n")
        # For each strain, print output of each rank as CSV
        overall_lineage = createOverallLineage(rank_list, lineage_clusters)
        writeClusterCsv(os.path.join(strain_db_name,os.path.basename(strain_db_name) + '_lineages.csv'),
            strain_rlist,
            strain_rlist,
            overall_lineage,
            output_format = 'phandango',
            epiCsv = None,
            suffix = '_Lineage')
        genomeNetwork = indivNetworks[min(rank_list)]
        # Save model
        model.save()

    return 0

if __name__ == '__main__':
    main()

    sys.exit(0)
