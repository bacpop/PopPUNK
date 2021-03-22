#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018-2020 John Lees and Nick Croucher

# universal
import os
import sys
# additional
import numpy as np
import subprocess
from collections import defaultdict

# Try to import sketchlib
try:
    import pp_sketchlib
except ImportError as e:
    sys.stderr.write("Sketchlib backend not available\n")
    sys.exit(1)

import poppunk_refine
import h5py

# import poppunk package
from .__init__ import __version__

# globals
accepted_weights_types = ["core", "accessory", "euclidean"]

#******************************#
#*                            *#
#* Command line parsing       *#
#*                            *#
#******************************#
def get_options():

    import argparse

    parser = argparse.ArgumentParser(description='PopPUNK (POPulation Partitioning Using Nucleotide Kmers)',
                                     prog='poppunk')

    modeGroup = parser.add_argument_group('Mode of operation')
    mode = modeGroup.add_mutually_exclusive_group(required=True)
    mode.add_argument('--create-db',
            help='Create pairwise distances database between reference sequences',
            default=False,
            action='store_true')
    mode.add_argument('--fit-model',
            help='Fit a mixture model to a reference database',
            choices=['bgmm', 'dbscan', 'refine', 'lineage', 'threshold'],
            default = False)
    mode.add_argument('--use-model',
            help='Apply a fitted model to a reference database to restore database files',
            default=False,
            action='store_true')

    # input options
    iGroup = parser.add_argument_group('Input files')
    iGroup.add_argument('--ref-db',type = str, help='Location of built reference database')
    iGroup.add_argument('--r-files', help='File listing reference input assemblies')
    iGroup.add_argument('--distances', help='Prefix of input pickle of pre-calculated distances')
    iGroup.add_argument('--external-clustering', help='File with cluster definitions or other labels '
                                                      'generated with any other method.', default=None)

    # output options
    oGroup = parser.add_argument_group('Output options')
    oGroup.add_argument('--output', help='Prefix for output files')
    oGroup.add_argument('--plot-fit', help='Create this many plots of some fits relating k-mer to core/accessory distances '
                                            '[default = 0]', default=0, type=int)
    oGroup.add_argument('--overwrite', help='Overwrite any existing database files', default=False, action='store_true')
    oGroup.add_argument('--graph-weights', help='Save within-strain Euclidean distances into the graph', default=False, action='store_true')

    # comparison metrics
    kmerGroup = parser.add_argument_group('Create DB options')
    kmerGroup.add_argument('--min-k', default = 13, type=int, help='Minimum kmer length [default = 13]')
    kmerGroup.add_argument('--max-k', default = 29, type=int, help='Maximum kmer length [default = 29]')
    kmerGroup.add_argument('--k-step', default = 4, type=int, help='K-mer step size [default = 4]')
    kmerGroup.add_argument('--sketch-size', default=10000, type=int, help='Kmer sketch size [default = 10000]')
    kmerGroup.add_argument('--codon-phased', default=False, action='store_true',
                            help='Used codon phased seeds X--X--X [default = False]')
    kmerGroup.add_argument('--min-kmer-count', default=0, type=int, help='Minimum k-mer count when using reads as input [default = 0]')
    kmerGroup.add_argument('--exact-count', default=False, action='store_true',
                           help='Use the exact k-mer counter with reads '
                                '[default = use countmin counter]')
    kmerGroup.add_argument('--strand-preserved', default=False, action='store_true',
                           help='Treat input as being on the same strand, and ignore reverse complement '
                                'k-mers [default = use canonical k-mers]')

    # qc options
    qcGroup = parser.add_argument_group('Quality control options')
    qcGroup.add_argument('--qc-filter', help='Behaviour following sequence QC step: "stop" [default], "prune"'
                                                ' (analyse data passing QC), or "continue" (analyse all data)',
                                                default='stop', type = str, choices=['stop', 'prune', 'continue'])
    qcGroup.add_argument('--retain-failures', help='Retain sketches of genomes that do not pass QC filters in '
                                                'separate database [default = False]', default=False, action='store_true')
    qcGroup.add_argument('--max-a-dist', help='Maximum accessory distance to permit [default = 0.5]',
                                                default = 0.5, type = float)
    qcGroup.add_argument('--max-pi-dist', help='Maximum core distance to permit [default = 0.5]',
                                                default = 0.5, type = float)
    qcGroup.add_argument('--type-isolate', help='Isolate from which distances will be calculated for pruning [default = None]',
                                                default = None, type = str)
    qcGroup.add_argument('--length-sigma', help='Number of standard deviations of length distribution beyond '
                                                'which sequences will be excluded [default = 5]', default = 5, type = int)
    qcGroup.add_argument('--length-range', help='Allowed length range, outside of which sequences will be excluded '
                                                '[two values needed - lower and upper bounds]', default=[None,None],
                                                type = int, nargs = 2)
    qcGroup.add_argument('--prop-n', help='Threshold ambiguous base proportion above which sequences will be excluded'
                                                ' [default = 0.1]', default = 0.1,
                                                type = float)
    qcGroup.add_argument('--upper-n', help='Threshold ambiguous base count above which sequences will be excluded',
                                                default=None, type = int)

    # model fitting
    modelGroup = parser.add_argument_group('Model fit options')
    modelGroup.add_argument('--K', help='Maximum number of mixture components [default = 2]', type=int, default=2)
    modelGroup.add_argument('--D', help='Maximum number of clusters in DBSCAN fitting [default = 100]', type=int, default=100)
    modelGroup.add_argument('--min-cluster-prop', help='Minimum proportion of points in a cluster '
                                                        'in DBSCAN fitting [default = 0.0001]', type=float, default=0.0001)
    modelGroup.add_argument('--threshold', help='Cutoff if using --fit-model threshold', type=float)

    # model refinement
    refinementGroup = parser.add_argument_group('Refine model options')
    refinementGroup.add_argument('--pos-shift', help='Maximum amount to move the boundary away from origin [default = to between-strain mean]',
            type=float, default = None)
    refinementGroup.add_argument('--neg-shift', help='Maximum amount to move the boundary towards the origin [default = to within-strain mean]',
            type=float, default = None)
    refinementGroup.add_argument('--manual-start', help='A file containing information for a start point. '
            'See documentation for help.', default=None)
    refinementGroup.add_argument('--model-dir', help='Directory containing model to use for assigning queries '
                                                   'to clusters [default = reference database directory]', type = str)
    refinementGroup.add_argument('--score-idx',
            help='Index of score to use [default = 0]',
            type=int, default = 0, choices=[0, 1, 2])
    refineMode = refinementGroup.add_mutually_exclusive_group()
    refineMode.add_argument('--unconstrained',
            help='Optimise both boundary gradient and intercept',
            default=False, action='store_true')
    refineMode.add_argument('--indiv-refine', help='Also run refinement for core and accessory individually',
            choices=['both', 'core', 'accessory'], default=False)

    # lineage clustering within strains
    lineagesGroup = parser.add_argument_group('Lineage analysis options')
    lineagesGroup.add_argument('--ranks',
                                help='Comma separated list of ranks used in lineage clustering [default = 1,2,3]',
                                type = str,
                                default = "1,2,3")
    lineagesGroup.add_argument('--use-accessory',
                                help='Use accessory distances for lineage definitions [default = use core distances]',
                                action = 'store_true',
                                default = False)

    other = parser.add_argument_group('Other options')
    other.add_argument('--threads', default=1, type=int, help='Number of threads to use [default = 1]')
    other.add_argument('--gpu-sketch', default=False, action='store_true', help='Use a GPU when calculating sketches (read data only) [default = False]')
    other.add_argument('--gpu-dist', default=False, action='store_true', help='Use a GPU when calculating distances [default = False]')
    other.add_argument('--gpu-graph', default=False, action='store_true', help='Use a GPU when calculating networks [default = False]')
    other.add_argument('--deviceid', default=0, type=int, help='CUDA device ID, if using GPU [default = 0]')
    other.add_argument('--no-plot', help='Switch off model plotting, which can be slow for large datasets',
                                                default=False, action='store_true')
    other.add_argument('--no-local', help='Do not perform the local optimization step in model refinement (speed up on very large datasets)',
                                                default=False, action='store_true')

    other.add_argument('--version', action='version',
                       version='%(prog)s '+__version__)
    other.add_argument('--citation',
                       action='store_true',
                       default=False,
                       help='Give a citation, and possible methods paragraph '
                            'based on the command line')


    # combine
    args = parser.parse_args()

    # ensure directories do not have trailing forward slash
    for arg in [args.ref_db, args.model_dir, args.output]:
        if arg is not None:
            arg = arg.rstrip('\\')

    return args

def main():
    """Main function. Parses cmd line args and runs in the specified mode.
    """

    #******************************#
    #*                            *#
    #* Check command options      *#
    #*                            *#
    #******************************#
    if sys.version_info[0] < 3:
        sys.stderr.write('PopPUNK requires python version 3 or above\n')
        sys.exit(1)

    args = get_options()

    # May just want to print the citation
    if args.citation:
        from .citation import print_citation
        print_citation(args)
        sys.exit(0)

    # Imports are here because graph tool is very slow to load
    from .models import loadClusterFit, ClusterFit, BGMMFit, DBSCANFit, RefineFit, LineageFit
    from .sketchlib import checkSketchlibLibrary
    from .sketchlib import removeFromDB

    from .network import constructNetwork
    from .network import extractReferences
    from .network import printClusters
    from .network import get_vertex_list
    from .network import save_network
    from .network import checkNetworkVertexCount

    from .plot import writeClusterCsv
    from .plot import plot_scatter

    from .prune_db import prune_distance_matrix

    from .utils import setGtThreads
    from .utils import setupDBFuncs
    from .utils import storePickle
    from .utils import readPickle
    from .utils import qcDistMat
    from .utils import createOverallLineage

    # check kmer properties
    if args.min_k >= args.max_k:
        sys.stderr.write("Minimum kmer size " + str(args.min_k) + " must be smaller than maximum kmer size\n")
        sys.exit(1)
    elif args.k_step < 1:
        sys.stderr.write("Kmer size step must be at least one\n")
        sys.exit(1)
    elif args.min_k < 3:
        sys.stderr.write("Min k-mer length must be 3 or higher\n")
        sys.exit(1)
    kmers = np.arange(args.min_k, args.max_k + 1, args.k_step)

    # Dict of QC options for passing to database construction and querying functions
    qc_dict = {
        'run_qc': args.create_db,
        'qc_filter': args.qc_filter,
        'retain_failures': args.retain_failures,
        'length_sigma': args.length_sigma,
        'length_range': args.length_range,
        'prop_n': args.prop_n,
        'upper_n': args.upper_n,
        'max_pi_dist': args.max_pi_dist,
        'max_a_dist': args.max_a_dist,
        'type_isolate': args.type_isolate
    }

    # Dict of DB access functions
    dbFuncs = setupDBFuncs(args, args.min_kmer_count, qc_dict)
    createDatabaseDir = dbFuncs['createDatabaseDir']
    constructDatabase = dbFuncs['constructDatabase']
    queryDatabase = dbFuncs['queryDatabase']

    # define sketch sizes, store in hash in case one day
    # different kmers get different hash sizes
    sketch_sizes = {}
    if args.sketch_size >= 100 and args.sketch_size <= 10**6:
        for k in kmers:
            sketch_sizes[k] = args.sketch_size
    else:
        sys.stderr.write("Sketch size should be between 100 and 10^6\n")
        sys.exit(1)
    # for sketchlib, only supporting a single sketch size
    sketch_sizes = int(round(max(sketch_sizes.values())/64))

    # if a length range is specified, check it makes sense
    if args.length_range[0] is not None:
        if args.length_range[0] >= args.length_range[1]:
            sys.stderr.write('Ensure the specified length range is space-separated argument of'
                             ' length 2, with the lower value first\n')
            sys.exit(1)

    # check if working with lineages
    if args.fit_model == 'lineage':
        rank_list = sorted([int(x) for x in args.ranks.split(',')])
        if int(min(rank_list)) == 0:
            sys.stderr.write("Ranks must be >= 1\n")
        if max(rank_list) > 100:
            sys.stderr.write("WARNING: Ranks should be small non-zero integers for sensible lineage results\n")

    # run according to mode
    sys.stderr.write("PopPUNK (POPulation Partitioning Using Nucleotide Kmers)\n")
    sys.stderr.write("\t(with backend: " + dbFuncs['backend'] + " v" + dbFuncs['backend_version'] + "\n")
    sys.stderr.write("\t sketchlib: " + checkSketchlibLibrary() + ")\n")

    # Check on parallelisation of graph-tools
    setGtThreads(args.threads)

    #******************************#
    #*                            *#
    #* Create database            *#
    #*                            *#
    #******************************#
    if args.create_db:
        sys.stderr.write("Mode: Building new database from input sequences\n")
        if args.r_files is None or args.output is None:
            sys.stderr.write("--create-db requires --r-files and --output")
            sys.exit(1)

        # generate sketches and QC sequences to identify sequences not matching specified criteria
        createDatabaseDir(args.output, kmers)
        seq_names_passing = \
            constructDatabase(
                args.r_files,
                kmers,
                sketch_sizes,
                args.output,
                args.threads,
                args.overwrite,
                codon_phased = args.codon_phased,
                calc_random = True)

        # calculate distances between sequences
        distMat = queryDatabase(rNames = seq_names_passing,
                                qNames = seq_names_passing,
                                dbPrefix = args.output,
                                queryPrefix = args.output,
                                klist = kmers,
                                self = True,
                                number_plot_fits = args.plot_fit,
                                threads = args.threads)

        # QC pairwise distances to identify long distances indicative of anomalous sequences in the collection
        seq_names_passing, distMat = qcDistMat(distMat,
                                                seq_names_passing,
                                                seq_names_passing,
                                                args.output,
                                                args.output,
                                                qc_dict)

        # Plot results
        if not args.no_plot:
            plot_scatter(distMat,
                         args.output + "/" + os.path.basename(args.output) + "_distanceDistribution",
                         args.output + " distances")

    #******************************#
    #*                            *#
    #* model fit and network      *#
    #* construction               *#
    #*                            *#
    #******************************#
    # refine model also needs to run all model steps
    if args.fit_model or args.use_model:
        if args.fit_model:
            sys.stderr.write("Mode: Fitting " + args.fit_model + \
                             " model to reference database\n\n")
        elif args.use_model:
            sys.stderr.write("Mode: Using previous model with a reference database\n\n")

        # Check and set required parameters
        if args.ref_db is None:
            sys.stderr.write("Need to provide --ref-db where .h5 and .dists from "
                             "--create-db mode were output")
        if args.distances is None:
            distances = args.ref_db + "/" + os.path.basename(args.ref_db) + ".dists"
        else:
            distances = args.distances
        if args.output is None:
            output = args.ref_db
        else:
            output = args.output

        # Set up variables for using previous models
        if args.fit_model == "refine" or args.use_model:
            model_prefix = args.ref_db
            if args.model_dir is not None:
                model_prefix = args.model_dir
            model = loadClusterFit(model_prefix + "/" + os.path.basename(model_prefix) + '_fit.pkl',
                                   model_prefix + "/" + os.path.basename(model_prefix) + '_fit.npz',
                                   output)
            model.set_threads(args.threads)
            sys.stderr.write("Loaded previous model of type: " + model.type + "\n")
            if args.fit_model == "refine" and args.manual_start == None \
                and model.type != 'bgmm' and model.type != 'dbscan':
                sys.stderr.write("Model needs to be from BGMM or DBSCAN to refine\n")
                sys.exit(1)

        # Load the distances
        refList, queryList, self, distMat = readPickle(distances, enforce_self=True)
        seq_names = set(set(refList) | set(queryList))
        seq_names_passing, distMat = qcDistMat(distMat, refList, queryList, args.ref_db, output, qc_dict)
        if len(set(seq_names_passing).difference(seq_names)) > 0 and args.qc_filter == "stop":
            sys.stderr.write("Distances failed quality control (change QC options to run anyway)\n")
            sys.exit(1)

        #******************************#
        #*                            *#
        #* model fit                  *#
        #*                            *#
        #******************************#
        # Run selected model here, or if easy run DBSCAN followed by refinement
        if args.fit_model:
            # Run DBSCAN model
            if args.fit_model == "dbscan":
                model = DBSCANFit(output)
                model.set_threads(args.threads)
                assignments = model.fit(distMat, args.D, args.min_cluster_prop)
            # Run Gaussian model
            elif args.fit_model == "bgmm":
                model = BGMMFit(output)
                model.set_threads(args.threads)
                assignments = model.fit(distMat, args.K)
            elif args.fit_model == "refine":
                new_model = RefineFit(output)
                new_model.set_threads(args.threads)
                assignments = new_model.fit(distMat, refList, model,
                                            args.pos_shift, args.neg_shift,
                                            args.manual_start,
                                            args.indiv_refine,
                                            args.unconstrained,
                                            args.score_idx,
                                            args.no_local,
                                            args.gpu_graph)
                model = new_model
            elif args.fit_model == "threshold":
                new_model = RefineFit(output)
                new_model.set_threads(args.threads)
                assignments = new_model.apply_threshold(distMat,
                                                        args.threshold)
                model = new_model
            elif args.fit_model == "lineage":
                # run lineage clustering. Sparsity & low rank should keep memory
                # usage of dict reasonable
                model = LineageFit(output, rank_list)
                model.set_threads(args.threads)
                model.fit(distMat, args.use_accessory)

                assignments = {}
                for rank in rank_list:
                    assignments[rank] = \
                        model.assign(int(rank))

            # save model
            model.save()
            
            # plot model
            if not args.no_plot:
                model.plot(distMat, assignments)

        # use model
        else:
            assignments = model.assign(distMat)

        #******************************#
        #*                            *#
        #* network construction       *#
        #*                            *#
        #******************************#
        if model.type != "lineage":
            if args.graph_weights:
                weights = distMat
            else:
                weights = None
            genomeNetwork = \
                constructNetwork(refList,
                                 queryList,
                                 assignments,
                                 model.within_label,
                                 weights = weights,
                                 use_gpu = args.gpu_graph)
        else:
            # Lineage fit requires some iteration
            indivNetworks = {}
            lineage_clusters = defaultdict(dict)
            for rank in sorted(rank_list):
                sys.stderr.write("Network for rank " + str(rank) + "\n")
                if args.graph_weights:
                    weights = model.edge_weights(rank)
                else:
                    weights = None
                indivNetworks[rank] = constructNetwork(
                                        refList,
                                        refList,
                                        assignments[rank],
                                        0,
                                        edge_list = True,
                                        weights = weights,
                                        use_gpu = args.gpu_graph
                                       )
                lineage_clusters[rank] = \
                    printClusters(indivNetworks[rank],
                                  refList,
                                  printCSV = False,
                                  use_gpu = args.gpu_graph)

            # print output of each rank as CSV
            overall_lineage = createOverallLineage(rank_list, lineage_clusters)
            writeClusterCsv(output + "/" + \
                os.path.basename(output) + '_lineages.csv',
                refList,
                refList,
                overall_lineage,
                output_format = 'phandango',
                epiCsv = None,
                suffix = '_Lineage')
            genomeNetwork = indivNetworks[min(rank_list)]

        # Ensure all in dists are in final network
        checkNetworkVertexCount(refList, genomeNetwork, use_gpu = args.gpu_graph)

        fit_type = model.type
        isolateClustering = {fit_type: printClusters(genomeNetwork,
                                                     refList,
                                                     output + "/" + os.path.basename(output),
                                                     externalClusterCSV = args.external_clustering,
                                                     use_gpu = args.gpu_graph)}

        # Write core and accessory based clusters, if they worked
        if model.indiv_fitted:
            indivNetworks = {}
            for dist_type, slope in zip(['core', 'accessory'], [0, 1]):
                indivAssignments = model.assign(distMat, slope)
                indivNetworks[dist_type] = \
                    constructNetwork(refList,
                                     queryList,
                                     indivAssignments,
                                     model.within_label)
                isolateClustering[dist_type] = \
                    printClusters(indivNetworks[dist_type],
                                  refList,
                                  output + "/" + os.path.basename(output) + "_" + dist_type,
                                  externalClusterCSV = args.external_clustering)
                indivNetworks[dist_type].save(
                    output + "/" + os.path.basename(output) + \
                    "_" + dist_type + '_graph.gt', fmt = 'gt')

            if args.indiv_refine == 'core':
                fit_type = 'core'
                genomeNetwork = indivNetworks['core']
            elif args.indiv_refine == 'accessory':
                fit_type = 'accessory'
                genomeNetwork = indivNetworks['accessory']

        save_network(genomeNetwork, prefix = output, suffix = "_graph", use_gpu = args.gpu_graph)

        #******************************#
        #*                            *#
        #* clique pruning             *#
        #*                            *#
        #******************************#
        # extract limited references from clique by default
        # (this no longer loses information and should generally be kept on)
        if model.type != "lineage":
            newReferencesIndices, newReferencesNames, newReferencesFile, genomeNetwork = \
                extractReferences(genomeNetwork,
                                    refList,
                                    output,
                                    type_isolate = qc_dict['type_isolate'],
                                    threads = args.threads,
                                    use_gpu = args.gpu_graph)
            nodes_to_remove = set(range(len(refList))).difference(newReferencesIndices)
            names_to_remove = [refList[n] for n in nodes_to_remove]

            if (len(names_to_remove) > 0):
                # Save reference distances
                prune_distance_matrix(refList, names_to_remove, distMat,
                                      output + "/" + os.path.basename(output) + ".refs.dists")
                # Save reference network
                save_network(genomeNetwork, prefix = output, suffix = ".refs_graph",
                            use_gpu = args.gpu_graph)
                removeFromDB(args.ref_db, output, names_to_remove)
                os.rename(output + "/" + os.path.basename(output) + ".tmp.h5",
                          output + "/" + os.path.basename(output) + ".refs.h5")

    sys.stderr.write("\nDone\n")

if __name__ == '__main__':
    main()

    sys.exit(0)
