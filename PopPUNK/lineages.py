#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018-2023 John Lees and Nick Croucher

import os
import sys
import argparse
import subprocess
import pickle
import shutil
import pandas as pd
from collections import defaultdict

from .assign import assign_query_hdf5
from .network import construct_network_from_edge_list, printClusters, save_network
from .models import LineageFit
from .plot import writeClusterCsv
from .sketchlib import readDBParams
from .qc import prune_distance_matrix, sketchlibAssemblyQC
from .utils import createOverallLineage, get_match_search_depth, readPickle, setupDBFuncs

# command line parsing
def get_options():

    parser = argparse.ArgumentParser(description='Generate script and databases for lineage clustering across strains',
                                     prog='poppunk_lineages_within_strains')

    modeGroup = parser.add_argument_group('Mode of operation')
    mode = modeGroup.add_mutually_exclusive_group(required=True)
    mode.add_argument('--create-db',
            help='Strain database used to generate lineage databases',
            type=str,
            default=None)
    mode.add_argument('--query-db',
            help='File listing query input assemblies (required to query database)',
            type=str,
            default=None)

    # input/output options
    ioGroup = parser.add_argument_group('Input and output files')
    ioGroup.add_argument('--db-scheme', help = "Pickle file describing database scheme, written by --create-db"
                                        " and read by --query-db",
                                        required=True)
    ioGroup.add_argument('--output',    help = "Location of query output",
                                        required=True)
    ioGroup.add_argument('--model-dir',
                                        help="Directory containing model (if not in database)")
    ioGroup.add_argument('--distances',
                                        help="Distance file prefix (if not in database)")
    ioGroup.add_argument('--external-clustering',
                                        help="File with cluster definitions or other labels "
                                        "generated with any other method",
                                        default=None)
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
    ioGroup.add_argument('--overwrite',
                                        help="Overwrite existing analyses",
                                        default=False,
                                        action='store_true')

    aGroup = parser.add_argument_group('Analysis options')
    aGroup.add_argument('--threads',    help="Number of CPUs to use in analysis",
                                        default=1,
                                        type=int)
    aGroup.add_argument('--gpu-sketch', help="Use GPU for sketching",
                                        default=False,
                                        action='store_true')
    aGroup.add_argument('--gpu-dist', help="Use GPU for distance calculations",
                                        default=False,
                                        action='store_true')
    aGroup.add_argument('--gpu-graph', help="Use GPU for graph analysis",
                                        default=False,
                                        action='store_true')
    aGroup.add_argument('--deviceid',   help="Device ID of GPU",
                                        default=0,
                                        type=int)

    qGroup = parser.add_argument_group('Strain model querying options')
    strain_dist_type = qGroup.add_mutually_exclusive_group(required=False)
    strain_dist_type.add_argument('--core',
                                    help="Use core distances for strain definitions",
                                    action = 'store_true',
                                    default = False)
    strain_dist_type.add_argument('--accessory',
                                    help="Use accessory distances for strain definitions",
                                    action = 'store_true',
                                    default = False)
    qGroup.add_argument('--strand-preserved',
                                    help="Treat input as being on the same strand, and ignore reverse complement",
                                    action = 'store_true',
                                    default = False)
    qGroup.add_argument('--min-kmer-count',
                                    help="Minimum k-mer count when using reads as input  [default = 0]",
                                    type = int,
                                    default = 0)
    qGroup.add_argument('--exact-count',
                                    help="Use the exact k-mer counter with reads [default = use countmin counter]",
                                    action = 'store_true',
                                    default = False)

    lGroup = parser.add_argument_group('Lineage model options')
    lGroup.add_argument('--ranks',  help="Comma separated list of ranks used in lineage clustering",
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

    if args.create_db is not None:
        create_db(args)
    elif args.query_db is not None:
        query_db(args)


def create_db(args):

    # Check if output files exist
    if not args.overwrite:
        if os.path.exists(args.output + '.csv'):
            sys.stderr.write('Output file ' + args.output + '.csv exists; use --overwrite to replace it\n')
            sys.exit(1)
        if os.path.exists(args.db_scheme):
            sys.stderr.write('Output file ' + args.db_scheme + ' exists; use --overwrite to replace it\n')
            sys.exit(1)

    sys.stderr.write("Identifying strains in existing database\n")
    # Read in strain information
    if args.model_dir is None:
        args.model_dir = args.create_db
    if args.external_clustering is None:
        clustering_file = os.path.join(args.model_dir,os.path.basename(args.model_dir) + '_clusters.csv')
    else:
        clustering_file = args.external_clustering
    strains = pd.read_csv(clustering_file, dtype = str).groupby(args.clustering_col_name)

    sys.stderr.write("Extracting properties of database\n")
    # Get rlist
    if args.distances is None:
        distances = os.path.join(args.create_db,os.path.basename(args.create_db) + ".dists")
    else:
        distances = args.distances
    # Get distances
    rlist, qlist, self, X = readPickle(distances, enforce_self=False, distances=True)
    # Get parameters
    kmers, sketch_sizes, codon_phased = readDBParams(args.create_db)
    # Ranks to use
    rank_list = [int(x) for x in args.ranks.split(',')]
    if args.max_search_depth is not None:
        if args.max_search_depth <= max(rank_list):
            sys.stderr.write("Max search depth must be greater than the highest lineage rank\n")
            sys.exit(1)
        else:
            max_search_depth = args.max_search_depth
    else:
        max_search_depth = get_match_search_depth(rlist,rank_list)

    sys.stderr.write("Generating databases for individual strains\n")
    # Dicts for storing typing information
    lineage_dbs = {}
    overall_lineage = {}
    for strain,isolates in strains:
        # Make new database directory
        sys.stderr.write("Making database for strain " + str(strain) + "\n")
        strain_db_name = args.lineage_db_prefix + '_' + str(strain) + '_lineage_db'
        isolate_list = isolates[isolates.columns.values[0]].to_list()
        num_isolates = len(isolate_list)
        if num_isolates >= args.min_count:
            lineage_dbs[strain] = strain_db_name
            if os.path.isdir(strain_db_name) and args.overwrite:
                sys.stderr.write("--overwrite means {strain_db_name} will be deleted now\n")
                shutil.rmtree(strain_db_name)
            if not os.path.isdir(strain_db_name):
                try:
                    os.makedirs(strain_db_name)
                except OSError:
                    sys.stderr.write("Cannot create output directory " + strain_db_name + "\n")
                    sys.exit(1)
            # Make link to main database
            src_db = os.path.join(args.create_db,os.path.basename(args.create_db) + '.h5')
            dest_db = os.path.join(strain_db_name,os.path.basename(strain_db_name) + '.h5')
            rel_path = os.path.relpath(src_db, os.path.dirname(dest_db))
            if os.path.exists(dest_db) and args.overwrite:
                sys.stderr.write("--overwrite means {dest_db} will be deleted now\n")
                shutil.rmtree(dest_db)
            elif not os.path.exists(dest_db):
                os.symlink(rel_path,dest_db)
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
                      args.lineage_resolution,
                      dist_col = 1 if args.use_accessory else 0,
                      use_gpu = args.gpu_graph)
            model.set_threads(args.threads)
            # Load pruned distance matrix
            strain_rlist, strain_qlist, strain_self, strain_X = \
                                readPickle(os.path.join(strain_db_name,strain_db_name + '.dists'),
                                            enforce_self=False,
                                            distances=True)
            # Fit model
            model.fit(strain_X)
            # Lineage fit requires some iteration
            indivNetworks = {}
            lineage_clusters = defaultdict(dict)
            # Iterate over ranks
            for rank in rank_list:
                if rank <= num_isolates:
                    assignments = model.assign(rank)
                # Generate networks
                indivNetworks[rank] = construct_network_from_edge_list(strain_rlist,
                                                            strain_rlist,
                                                            assignments,
                                                            weights = None,
                                                            betweenness_sample = None,
                                                            use_gpu = args.gpu_graph,
                                                            summarise = False
                                                           )
                # Write networks
                save_network(indivNetworks[rank],
                        prefix = strain_db_name,
                        suffix = '_rank_' + str(rank) + '_graph',
                            use_gpu = args.gpu_graph)
                # Identify clusters from output
                lineage_clusters[rank] = \
                    printClusters(indivNetworks[rank],
                                  strain_rlist,
                                  printCSV = False,
                                  use_gpu = args.gpu_graph)
                n_clusters = max(lineage_clusters[rank].values())
                sys.stderr.write("Network for rank " + str(rank) + " has " +
                                 str(n_clusters) + " lineages\n")
            # For each strain, print output of each rank as CSV
            overall_lineage[strain] = createOverallLineage(rank_list, lineage_clusters)
            writeClusterCsv(os.path.join(strain_db_name,os.path.basename(strain_db_name) + '_lineages.csv'),
                strain_rlist,
                strain_rlist,
                overall_lineage[strain],
                output_format = 'phandango',
                epiCsv = None,
                suffix = '_Lineage')
            genomeNetwork = indivNetworks[min(rank_list)]
            # Save model
            model.save()

    # Print combined strain and lineage clustering
    print_overall_clustering(overall_lineage,args.output + '.csv',rlist)

    # Write scheme to file
    with open(args.db_scheme, 'wb') as pickle_file:
        pickle.dump([args.create_db,
                      rlist,
                      args.model_dir,
                      clustering_file,
                      args.clustering_col_name,
                      distances,
                      kmers,
                      sketch_sizes,
                      codon_phased,
                      max_search_depth,
                      rank_list,
                      args.use_accessory,
                      args.min_count,
                      args.count_unique_distances,
                      args.reciprocal_only,
                      args.strand_preserved,
                      args.core,
                      args.accessory,
                      lineage_dbs],
                    pickle_file)


def query_db(args):

    # Read querying scheme
    with open(args.db_scheme, 'rb') as pickle_file:
        ref_db, rlist, model_dir, clustering_file, args.clustering_col_name, distances, \
        kmers, sketch_sizes, codon_phased, max_search_depth, rank_list, use_accessory, min_count, \
        count_unique_distances, reciprocal_only, strand_preserved, core, accessory, lineage_dbs = \
          pickle.load(pickle_file)

    dbFuncs = setupDBFuncs(args)

    # Define clustering files
    previous_clustering_file = os.path.join(model_dir,os.path.basename(model_dir) + '_clusters.csv')
    external_clustering = None
    if clustering_file != previous_clustering_file:
        external_clustering = clustering_file

    # Ignore QC at the moment
    qc_dict = {'run_qc': False, 'type_isolate': None }

    # Check output file
    if args.output is None:
        sys.stderr.write("Need an output file name\n")
        sys.exit(1)

    # Set up database
    createDatabaseDir = dbFuncs['createDatabaseDir']
    constructDatabase = dbFuncs['constructDatabase']
    readDBParams = dbFuncs['readDBParams']

    if ref_db == args.output:
        sys.stderr.write("--output and --ref-db must be different to "
                         "prevent overwrite.\n")
        sys.exit(1)

    # construct database
    createDatabaseDir(args.output, kmers)
    qNames = constructDatabase(args.query_db,
                                kmers,
                                sketch_sizes,
                                args.output,
                                args.threads,
                                True, # overwrite - probably OK?
                                codon_phased = codon_phased,
                                calc_random = False,
                                use_gpu = args.gpu_sketch,
                                deviceid = args.deviceid)

    if qc_dict["run_qc"]:
        pass_assembly_qc, fail_assembly_qc = \
                sketchlibAssemblyQC(output,
                                    qNames,
                                    qc_dict)
        if len(fail_assembly_qc) > 0:
            sys.stderr.write(f"{len(fail_assembly_qc)} samples failed:\n"
                             f"{','.join(fail_assembly_qc.keys())}\n")
            qNames = [x for x in qNames if x in pass_assembly_qc]
            if len(qNames) == 0:
                raise RuntimeError("No query sequences remaining after QC filters")
                sys.exit(1)

    isolateClustering = \
        assign_query_hdf5(dbFuncs,
                    ref_db,
                    qNames,
                    args.output,
                    qc_dict,
                    False, # update DB - not yet
                    False, # write references - need to consider whether to support ref-only databases for assignment
                    distances,
                    False, # serial - needs to be supported for web version?
                    None, # stable - not supported here
                    args.threads,
                    True, # overwrite - probably OK?
                    False, # plot_fit - turn off for now
                    False, # graph weights - might be helpful for MSTs not for strains
                    model_dir,
                    strand_preserved,
                    model_dir,
                    external_clustering,
                    core,
                    accessory,
                    args.gpu_dist,
                    args.gpu_graph,
                    save_partial_query_graph = False,
                    use_full_network = True) # Use full network - does not make sense to use references for lineages

    # Process clustering
    query_strains = {}
    clustering_type = 'combined'
    if core:
        core = True
        clustering_type = 'core'
    if accessory:
        core = False
        clustering_type = 'accessory'
    for isolate in isolateClustering[clustering_type]:
        if isolate in qNames:
          strain = isolateClustering[clustering_type][isolate]
          if strain in query_strains:
              query_strains[strain].append(isolate)
          else:
              query_strains[strain] = [isolate]

    # Assign to lineage for each isolate
    overall_lineage = {}
    for strain in query_strains:
        if strain in lineage_dbs.keys():
            lineage_distances = os.path.join(lineage_dbs[strain],os.path.basename(lineage_dbs[strain]) + '.dists')
            lineageClustering = \
                assign_query_hdf5(dbFuncs,
                            lineage_dbs[strain],
                            query_strains[strain],
                            args.output,
                            qc_dict,
                            False, # update DB - not yet
                            False, # write references - need to consider whether to support ref-only databases for assignment
                            lineage_distances,
                            False, # serial - needs to be supported for web version?
                            None, # stable - not supported here
                            args.threads,
                            True, # overwrite - probably OK?
                            False, # plot_fit - turn off for now
                            False, # graph weights - might be helpful for MSTs not for strains
                            lineage_dbs[strain],
                            strand_preserved,
                            lineage_dbs[strain],
                            None, # No external clustering
                            core,
                            accessory,
                            args.gpu_dist,
                            args.gpu_graph,
                            save_partial_query_graph = False,
                            use_full_network = True)
            overall_lineage[strain] = createOverallLineage(rank_list, lineageClustering)

    # Print combined strain and lineage clustering
    print_overall_clustering(overall_lineage,args.output + '.csv',qNames)


def print_overall_clustering(overall_lineage,output,include_list):

    # Get clustering information
    first_strain = list(overall_lineage.keys())[0]
    isolate_info = {}
    ranks = list(overall_lineage[first_strain].keys())

    # Compile clustering
    for strain in overall_lineage:
        for rank in ranks:
            for isolate in overall_lineage[strain][rank]:
                if isolate in include_list:
                    if isolate in isolate_info:
                        isolate_info[isolate].append(str(overall_lineage[strain][rank][isolate]))
                    else:
                        isolate_info[isolate] = [str(strain),str(overall_lineage[strain][rank][isolate])]

    # Print output
    with open(output,'w') as out:
        out.write('id,Cluster,')
        out.write(','.join(ranks))
        out.write('\n')
        for isolate in isolate_info:
            out.write(isolate + ',' + ','.join(isolate_info[isolate]))
            out.write('\n')

if __name__ == '__main__':
    main()
    sys.exit(0)

