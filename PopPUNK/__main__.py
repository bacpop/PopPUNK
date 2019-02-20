#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018 John Lees and Nick Croucher

# universal
import os
import sys
# additional
import numpy as np
import networkx as nx
import subprocess

# import poppunk package
from .__init__ import __version__

from .mash import checkMashVersion
from .mash import createDatabaseDir
from .mash import joinDBs
from .mash import constructDatabase
from .mash import queryDatabase
from .mash import readMashDBParams

from .models import *

from .network import constructNetwork
from .network import extractReferences
from .network import writeReferences
from .network import addQueryToNetwork
from .network import printClusters

from .plot import outputsForMicroreact
from .plot import outputsForCytoscape
from .plot import outputsForPhandango
from .plot import outputsForGrapetree

from .prune_db import prune_distance_matrix

from .utils import storePickle
from .utils import readPickle
from .utils import writeTmpFile
from .utils import qcDistMat
from .utils import translate_distMat
from .utils import update_distance_matrices

#################
# run main code #
#################

# command line parsing
def get_options():

    import argparse

    parser = argparse.ArgumentParser(description='PopPUNK (POPulation Partitioning Using Nucleotide Kmers)',
                                     prog='PopPUNK')

    modeGroup = parser.add_argument_group('Mode of operation')
    mode = modeGroup.add_mutually_exclusive_group(required=True)
    mode.add_argument('--easy-run',
            help='Create clusters from assemblies with default settings',
            default=False,
            action='store_true')
    mode.add_argument('--create-db',
            help='Create pairwise distances database between reference sequences',
            default=False,
            action='store_true')
    mode.add_argument('--fit-model',
            help='Fit a mixture model to a reference database',
            default=False,
            action='store_true')
    mode.add_argument('--refine-model',
            help='Refine the accuracy of a fitted model',
            default=False,
            action='store_true')
    mode.add_argument('--assign-query',
            help='Assign the cluster of query sequences without re-running the whole mixture model',
            default=False,
            action='store_true')
    mode.add_argument('--use-model',
            help='Apply a fitted model to a reference database to restore database files',
            default=False,
            action='store_true')

    # input options
    iGroup = parser.add_argument_group('Input files')
    iGroup.add_argument('--ref-db',type = str, help='Location of built reference database')
    iGroup.add_argument('--r-files', help='File listing reference input assemblies')
    iGroup.add_argument('--q-files', help='File listing query input assemblies')
    iGroup.add_argument('--distances', help='Prefix of input pickle of pre-calculated distances')
    iGroup.add_argument('--external-clustering', help='File with cluster definitions or other labels '
                                                      'generated with any other method.', default=None)

    # output options
    oGroup = parser.add_argument_group('Output options')
    oGroup.add_argument('--output', required=True, help='Prefix for output files (required)')
    oGroup.add_argument('--plot-fit', help='Create this many plots of some fits relating k-mer to core/accessory distances '
                                            '[default = 0]', default=0, type=int)
    oGroup.add_argument('--full-db', help='Keep full reference database, not just representatives', default=False, action='store_true')
    oGroup.add_argument('--update-db', help='Update reference database with query sequences', default=False, action='store_true')
    oGroup.add_argument('--overwrite', help='Overwrite any existing database files', default=False, action='store_true')

    # comparison metrics
    kmerGroup = parser.add_argument_group('Kmer comparison options')
    kmerGroup.add_argument('--min-k', default = 9, type=int, help='Minimum kmer length [default = 9]')
    kmerGroup.add_argument('--max-k', default = 29, type=int, help='Maximum kmer length [default = 29]')
    kmerGroup.add_argument('--k-step', default = 4, type=int, help='K-mer step size [default = 4]')
    kmerGroup.add_argument('--sketch-size', default=10000, type=int, help='Kmer sketch size [default = 10000]')

    # qc options
    qcGroup = parser.add_argument_group('Quality control options')
    qcGroup.add_argument('--max-a-dist', default = 0.5, type=float, help='Maximum accessory distance to permit '
                                                                         '[default = 0.5]')
    qcGroup.add_argument('--ignore-length', help='Ignore outliers in terms of assembly length '
                                                 '[default = False]', default=False, action='store_true')

    # model fitting
    modelGroup = parser.add_argument_group('Model fit options')
    modelGroup.add_argument('--K', help='Maximum number of mixture components [default = 2]', type=int, default=2)
    modelGroup.add_argument('--dbscan', help='Use DBSCAN rather than mixture model', default=False, action='store_true')
    modelGroup.add_argument('--D', help='Maximum number of clusters in DBSCAN fitting [default = 100]', type=int, default=100)
    modelGroup.add_argument('--min-cluster-prop', help='Minimum proportion of points in a cluster '
                                                        'in DBSCAN fitting [default = 0.0001]', type=float, default=0.0001)

    # model refinement
    refinementGroup = parser.add_argument_group('Refine model options')
    refinementGroup.add_argument('--pos-shift', help='Maximum amount to move the boundary away from origin [default = 0.2]',
            type=float, default=0.2)
    refinementGroup.add_argument('--neg-shift', help='Maximum amount to move the boundary towards the origin [default = 0.4]',
            type=float, default=0.4)
    refinementGroup.add_argument('--manual-start', help='A file containing information for a start point. '
            'See documentation for help.', default=None)
    refinementGroup.add_argument('--indiv-refine', help='Also run refinement for core and accessory individually', default=False,
            action='store_true')
    refinementGroup.add_argument('--no-local', help='Do not perform the local optimization step (speed up on very large datasets)',
            default=False, action='store_true')

    # sequence querying
    queryingGroup = parser.add_argument_group('Database querying options')
    queryingGroup.add_argument('--model-dir', help='Directory containing model to use for assigning queries '
                                                   'to clusters [default = reference database directory]', type = str)
    queryingGroup.add_argument('--previous-clustering', help='Directory containing previous cluster definitions '
                                                             'and network [default = use that in the directory '
                                                             'containing the model]', type = str)
    queryingGroup.add_argument('--core-only', help='Use a core-distance only model for assigning queries '
                                              '[default = False]', default=False, action='store_true')
    queryingGroup.add_argument('--accessory-only', help='Use an accessory-distance only model for assigning queries '
                                              '[default = False]', default=False, action='store_true')

    # model output
    faGroup = parser.add_argument_group('Further analysis options')
    faGroup.add_argument('--microreact', help='Generate output files for microreact visualisation', default=False, action='store_true')
    faGroup.add_argument('--cytoscape', help='Generate network output files for Cytoscape', default=False, action='store_true')
    faGroup.add_argument('--phandango', help='Generate phylogeny and TSV for Phandango visualisation', default=False, action='store_true')
    faGroup.add_argument('--grapetree', help='Generate phylogeny and CSV for grapetree visualisation', default=False, action='store_true')
    faGroup.add_argument('--rapidnj', help='Path to rapidNJ binary to build NJ tree for Microreact', default=None)
    faGroup.add_argument('--perplexity', type=float, default = 20.0,
                         help='Perplexity used to calculate t-SNE projection (with --microreact) [default=20.0]')
    faGroup.add_argument('--info-csv',
                     help='Epidemiological information CSV formatted for microreact (can be used with other outputs)')

    # processing
    other = parser.add_argument_group('Other options')
    other.add_argument('--mash', default='mash', help='Location of mash executable')
    other.add_argument('--threads', default=1, type=int, help='Number of threads to use [default = 1]')
    other.add_argument('--no-stream', help='Use temporary files for mash dist interfacing. Reduce memory use/increase disk use for large datasets', default=False, action='store_true')

    other.add_argument('--version', action='version',
                       version='%(prog)s '+__version__)

    return parser.parse_args()

def main():

    if sys.version_info[0] < 3:
        sys.stderr.write('PopPUNK requires python version 3 or above\n')
        sys.exit(1)

    args = get_options()

    # check mash is installed
    checkMashVersion(args.mash)

    # check kmer properties
    if args.min_k >= args.max_k or args.min_k < 9 or args.max_k > 31 or args.k_step < 2:
        sys.stderr.write("Minimum kmer size " + str(args.min_k) + " must be smaller than maximum kmer size " +
                         str(args.max_k) + "; range must be between 9 and 31, step must be at least one\n")
        sys.exit(1)
    kmers = np.arange(args.min_k, args.max_k + 1, args.k_step)

    # define sketch sizes, store in hash in case on day
    # different kmers get different hash sizes
    sketch_sizes = {}
    if args.sketch_size >= 100 and args.sketch_size <= 10**6:
        for k in kmers:
            sketch_sizes[k] = args.sketch_size
    else:
        sys.stderr.write("Sketch size should be between 100 and 10^6\n")
        sys.exit(1)

    # check on file paths and whether files will be overwritten
    # confusing to overwrite command line parameter
    #if not args.full_db and not (args.create_db or args.easy_run or args.assign_query):
    #    args.overwrite = True
    if args.output is not None and args.output.endswith('/'):
        args.output = args.output[:-1]
    if args.ref_db is not None and args.ref_db.endswith('/'):
        args.ref_db = args.ref_db[:-1]

    # run according to mode
    sys.stderr.write("PopPUNK (POPulation Partitioning Using Nucleotide Kmers)\n")

    # database construction
    if args.create_db or args.easy_run:
        if args.create_db:
            sys.stderr.write("Mode: Building new database from input sequences\n")
        elif args.easy_run:
            sys.stderr.write("Mode: Creating clusters from assemblies (create_db & fit_model)\n")
        if args.r_files is not None:
            createDatabaseDir(args.output, kmers)
            constructDatabase(args.r_files, kmers, sketch_sizes, args.output, args.ignore_length,
                              args.threads, args.mash, args.overwrite)
            refList, queryList, distMat = queryDatabase(args.r_files, kmers, args.output, args.output, True,
                    args.plot_fit, args.no_stream, args.mash, args.threads)
            qcDistMat(distMat, refList, queryList, args.max_a_dist)

            dists_out = args.output + "/" + os.path.basename(args.output) + ".dists"
            storePickle(refList, queryList, True, distMat, dists_out)
        else:
            sys.stderr.write("Need to provide a list of reference files with --r-files")
            sys.exit(1)

    # model fit and network construction
    # refine model also needs to run all model steps
    if args.fit_model or args.use_model or args.refine_model or args.easy_run:
        # Set up saved data from first step, if easy_run mode
        if args.easy_run:
            distances = dists_out
            ref_db = args.output
        else:
            if args.fit_model:
                sys.stderr.write("Mode: Fitting model to reference database\n\n")
            elif args.use_model:
                sys.stderr.write("Mode: Using previous model with a reference database\n\n")
            else:
                sys.stderr.write("Mode: Refining model fit using network properties\n\n")

            if args.distances is not None and args.ref_db is not None:
                distances = args.distances
                ref_db = args.ref_db
            else:
                sys.stderr.write("Need to provide an input set of distances with --distances "
                                 "and reference database directory with --ref-db\n\n")
                sys.exit(1)

        # Set up variables for using previous models
        if args.refine_model or args.use_model:
            model_prefix = args.ref_db
            if args.model_dir is not None:
                model_prefix = args.model_dir
            model = loadClusterFit(model_prefix + "/" + os.path.basename(model_prefix) + '_fit.pkl',
                        model_prefix + "/" + os.path.basename(model_prefix) + '_fit.npz',
                        args.output)
            if args.refine_model and model.type == 'refine':
                sys.stderr.write("Model needs to be from --fit-model not --refine-model\n")
                sys.exit(1)

        # Load the distances
        refList, queryList, self, distMat = readPickle(distances)
        if not self:
            sys.stderr.write("Model fit should be to a reference db made with --create-db\n")
            sys.exit(1)
        if qcDistMat(distMat, refList, queryList, args.max_a_dist) == False:
            sys.stderr.write("Distances failed quality control (change QC options to run anyway)\n")
            sys.exit(1)

        # Run selected model here, or if easy run DBSCAN followed by refinement
        if args.fit_model or args.easy_run:
            # Run DBSCAN model
            if args.dbscan or args.easy_run:
                model = DBSCANFit(args.output)
                assignments = model.fit(distMat, args.D, args.min_cluster_prop)
                model.plot()
            # Run Gaussian model
            else:
                model = BGMMFit(args.output)
                assignments = model.fit(distMat, args.K)
                model.plot(distMat, assignments)

        # Run model refinement
        if args.refine_model or args.easy_run:
            new_model = RefineFit(args.output)
            assignments = new_model.fit(distMat, refList, model, args.pos_shift, args.neg_shift,
                    args.manual_start, args.indiv_refine, args.no_local, args.threads)
            new_model.plot(distMat)
            model = new_model

        # Load and apply a previous model of any type
        if args.use_model:
            assignments = model.assign(distMat)
            model.plot(distMat, assignments)

        fit_type = 'combined'
        model.save()
        genomeNetwork = constructNetwork(refList, queryList, assignments, model.within_label)

        # Ensure all in dists are in final network
        networkMissing = set(refList).difference(list(genomeNetwork.nodes()))
        if len(networkMissing) > 0:
            sys.stderr.write("WARNING: Samples " + ",".join(networkMissing) + " are missing from the final network\n")

        isolateClustering = {fit_type: printClusters(genomeNetwork,
                                                     args.output + "/" + os.path.basename(args.output),
                                                     externalClusterCSV = args.external_clustering)}

        # Write core and accessory based clusters, if they worked
        if model.indiv_fitted:
            indivNetworks = {}
            for dist_type, slope in zip(['core', 'accessory'], [0, 1]):
                indivAssignments = model.assign(distMat, slope)
                indivNetworks[dist_type] = constructNetwork(refList, queryList, indivAssignments, model.within_label)
                isolateClustering[dist_type] = printClusters(indivNetworks[dist_type],
                                                 args.output + "/" + os.path.basename(args.output) + "_" + dist_type,
                                                 externalClusterCSV = args.external_clustering)
                nx.write_gpickle(indivNetworks[dist_type], args.output + "/" + os.path.basename(args.output) +
                                                           "_" + dist_type + '_graph.gpickle')
            if args.core_only:
                fit_type = 'core'
                genomeNetwork = indivNetworks['core']
            elif args.accessory_only:
                fit_type = 'accessory'
                genomeNetwork = indivNetworks['accessory']

        # generate distance matrices for outputs if required
        if args.microreact or args.cytoscape or args.phandango or args.grapetree:
            combined_seq, core_distMat, acc_distMat = update_distance_matrices(refList, distMat)
            # generate outputs for microreact if asked
            if args.microreact:
                sys.stderr.write("Writing microreact output\n")
                outputsForMicroreact(refList, core_distMat, acc_distMat, isolateClustering, args.perplexity,
                                     args.output, args.info_csv, args.rapidnj, overwrite = args.overwrite)
            # generate outputs for phandango if asked
            if args.phandango:
                sys.stderr.write("Writing phandango output\n")
                outputsForPhandango(refList, core_distMat, isolateClustering, args.output, args.info_csv, args.rapidnj,
                                    overwrite = args.overwrite, microreact = args.microreact)
            # generate outputs for grapetree if asked
            if args.grapetree:
                sys.stderr.write("Writing grapetree output\n")
                outputsForGrapetree(refList, core_distMat, isolateClustering, args.output, args.info_csv, args.rapidnj,
                                    overwrite = args.overwrite, microreact = args.microreact)
            # generate outputs for cytoscape if asked
            if args.cytoscape:
                sys.stderr.write("Writing cytoscape output\n")
                outputsForCytoscape(genomeNetwork, isolateClustering, args.output, args.info_csv)
                if model.indiv_fitted:
                    sys.stderr.write("Writing individual cytoscape networks\n")
                    for dist_type in ['core', 'accessory']:
                        outputsForCytoscape(indivNetworks[dist_type], isolateClustering, args.output,
                                    args.info_csv, suffix = dist_type, writeCsv = False)

        # extract limited references from clique by default
        if not args.full_db:
            newReferencesNames, newReferencesFile = extractReferences(genomeNetwork, refList, args.output)
            nodes_to_remove = set(refList).difference(newReferencesNames)
            genomeNetwork.remove_nodes_from(nodes_to_remove)
            prune_distance_matrix(refList, nodes_to_remove, distMat,
                                  args.output + "/" + os.path.basename(args.output) + ".dists")
            # Read previous database
            kmers, sketch_sizes = readMashDBParams(ref_db, kmers, sketch_sizes)
            constructDatabase(newReferencesFile, kmers, sketch_sizes, args.output, True, args.threads,
                              args.mash, True) # overwrite old db

        nx.write_gpickle(genomeNetwork, args.output + "/" + os.path.basename(args.output) + '_graph.gpickle')

    elif args.assign_query:
        if args.ref_db is not None and args.q_files is not None:
            sys.stderr.write("Mode: Assigning clusters of query sequences\n\n")
            self = False
            if args.ref_db == args.output:
                sys.stderr.write("--output and --ref-db must be different to "
                                 "prevent overwrite.\n")
                sys.exit(1)
            if (args.update_db and not args.distances):
                sys.stderr.write("--update-db requires --distances to be provided\n")
                sys.exit(1)
            if (args.microreact or args.cytoscape) and (not args.update_db or not args.distances):
                sys.stderr.write("--microreact and/or --cytoscape output must be "
                        "run with --distances and --update-db to generate a full "
                        " distance matrix\n")
                sys.exit(1)

            # Find distances to reference db
            kmers, sketch_sizes = readMashDBParams(args.ref_db, kmers, sketch_sizes)

            createDatabaseDir(args.output, kmers)
            constructDatabase(args.q_files, kmers, sketch_sizes, args.output, args.ignore_length,
                              args.threads, args.mash, args.overwrite)
            refList, queryList, distMat = queryDatabase(args.q_files, kmers, args.ref_db, args.output, False, args.plot_fit,
                                                        args.no_stream, args.mash, args.threads)
            qcPass = qcDistMat(distMat, refList, queryList, args.max_a_dist)

            # Assign these distances as within or between
            model_prefix = args.ref_db
            if args.model_dir is not None:
                model_prefix = args.model_dir
            model = loadClusterFit(model_prefix + "/" + os.path.basename(model_prefix) + '_fit.pkl',
                                   model_prefix + "/" + os.path.basename(model_prefix) + '_fit.npz')
            queryAssignments = model.assign(distMat)

            # Set directories of previous fit
            if args.previous_clustering is not None:
                prev_clustering = args.previous_clustering
            else:
                prev_clustering = model_prefix

            # If a refined fit, may use just core or accessory distances
            if args.core_only and model.type == 'refine':
                model.slope = 0
                old_network_file = prev_clustering + "/" + os.path.basename(prev_clustering) + '_core_graph.gpickle'
                old_cluster_file = prev_clustering + "/" + os.path.basename(prev_clustering) + '_core_clusters.csv'
            elif args.accessory_only and model.type == 'refine':
                model.slope = 1
                old_network_file = prev_clustering + "/" + os.path.basename(prev_clustering) + '_accessory_graph.gpickle'
                old_cluster_file = prev_clustering + "/" + os.path.basename(prev_clustering) + '_accessory_clusters.csv'
            else:
                old_network_file = prev_clustering + "/" + os.path.basename(prev_clustering) + '_graph.gpickle'
                old_cluster_file = prev_clustering + "/" + os.path.basename(prev_clustering) + '_clusters.csv'
                if args.core_only or args.accessory_only:
                    sys.stderr.write("Can only do --core-only or --accessory-only fits from "
                                     "a refined fit. Using the combined distances.\n")

            genomeNetwork = nx.read_gpickle(old_network_file)
            sys.stderr.write("Network loaded: " + str(genomeNetwork.number_of_nodes()) + " samples\n")

            # Ensure all in dists are in final network
            networkMissing = set(refList).difference(list(genomeNetwork.nodes()))
            if len(networkMissing) > 0:
                sys.stderr.write("WARNING: Samples " + ",".join(networkMissing) + " are missing from the final network\n")

            # Assign clustering by adding to network
            ordered_queryList, query_distMat = addQueryToNetwork(refList, queryList, args.q_files,
                    genomeNetwork, kmers, queryAssignments, model, args.output, args.no_stream, args.update_db,
                    args.threads, args.mash)

            # if running simple query
            print_full_clustering = False
            if args.update_db:
                print_full_clustering = True
            isolateClustering = {'combined': printClusters(genomeNetwork, args.output + "/" + os.path.basename(args.output),
                                                           old_cluster_file, args.external_clustering, print_full_clustering)}

            # update_db like no full_db
            if args.update_db:
                if not qcPass:
                    sys.stderr.write("Queries contained outlier distances, not updating database\n")
                else:
                    sys.stderr.write("Updating reference database to " + args.output + "\n")

                # Update the network + ref list
                if args.full_db is False:
                    mashOrder = refList + ordered_queryList
                    newRepresentativesNames, newRepresentativesFile = extractReferences(genomeNetwork, mashOrder, args.output, refList)
                    genomeNetwork.remove_nodes_from(set(genomeNetwork.nodes).difference(newRepresentativesNames))
                    newQueries = [x for x in ordered_queryList if x in frozenset(newRepresentativesNames)] # intersection that maintains order
                else:
                    newQueries = ordered_queryList
                nx.write_gpickle(genomeNetwork, args.output + "/" + os.path.basename(args.output) + '_graph.gpickle')

                # Update the mash database
                if newQueries != queryList:
                    tmpRefFile = writeTmpFile(newQueries)
                    constructDatabase(tmpRefFile, kmers, sketch_sizes, args.output, True,
                                      args.threads, args.mash, True) # overwrite old db
                    os.remove(tmpRefFile)
                joinDBs(args.ref_db, args.output, args.output, kmers)

                # Update distance matrices with all calculated distances
                if args.distances == None:
                    distanceFiles = args.ref_db + "/" + os.path.basename(args.ref_db) + ".dists"
                else:
                    distanceFiles = args.distances
                refList, refList_copy, self, ref_distMat = readPickle(distanceFiles)
                combined_seq, core_distMat, acc_distMat = update_distance_matrices(refList, ref_distMat,
                                                                    ordered_queryList, distMat, query_distMat)
                complete_distMat = translate_distMat(combined_seq, core_distMat, acc_distMat)

                # Prune distances to references only, if not full db
                dists_out = args.output + "/" + os.path.basename(args.output) + ".dists"
                if args.full_db is False:
                    # could also have newRepresentativesNames in this diff (should be the same) - but want
                    # to ensure consistency with the network in case of bad input/bugs
                    nodes_to_remove = set(combined_seq).difference(genomeNetwork.nodes)
                    # This function also writes out the new distance matrix
                    postpruning_combined_seq, newDistMat = prune_distance_matrix(combined_seq, nodes_to_remove,
                                                                                 complete_distMat, dists_out)

                    # ensure mash sketch and distMat order match
                    assert postpruning_combined_seq == refList + newQueries

                else:
                    storePickle(combined_seq, combined_seq, True, complete_distMat, dists_out)

                    # ensure mash sketch and distMat order match
                    assert combined_seq == refList + newQueries

            # generate outputs for microreact if asked
            if args.microreact:
                sys.stderr.write("Writing microreact output\n")
                outputsForMicroreact(combined_seq, core_distMat, acc_distMat, isolateClustering, args.perplexity,
                                     args.output, args.info_csv, args.rapidnj, ordered_queryList, args.overwrite)
            # generate outputs for phandango if asked
            if args.phandango:
                sys.stderr.write("Writing phandango output\n")
                outputsForPhandango(combined_seq, core_distMat, isolateClustering, args.output, args.info_csv, args.rapidnj,
                                    queryList = ordered_queryList, overwrite = args.overwrite, microreact = args.microreact)
            # generate outputs for grapetree if asked
            if args.grapetree:
                sys.stderr.write("Writing grapetree output\n")
                outputsForGrapetree(combined_seq, core_distMat, isolateClustering, args.output, args.info_csv, args.rapidnj,
                                    queryList = ordered_queryList, overwrite = args.overwrite, microreact = args.microreact)
            # generate outputs for cytoscape if asked
            if args.cytoscape:
                sys.stderr.write("Writing cytoscape output\n")
                outputsForCytoscape(genomeNetwork, isolateClustering, args.output, args.info_csv, ordered_queryList)
                if model.indiv_fitted:
                    sys.stderr.write("Writing individual cytoscape networks\n")
                    for dist_type in ['core', 'accessory']:
                        outputsForCytoscape(indivNetworks[dist_type], isolateClustering, args.output,
                            args.info_csv, queryList = ordered_queryList, suffix = dist_type, writeCsv = False)


        else:
            sys.stderr.write("Need to provide both a reference database with --ref-db and "
                             "query list with --q-files\n")
            sys.exit(1)


    sys.stderr.write("\nDone\n")

if __name__ == '__main__':
    main()

    sys.exit(0)
