#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018-2023 John Lees and Nick Croucher

# universal
import os
import sys
# additional
import numpy as np
from collections import defaultdict

# import poppunk package
from .__init__ import __version__

# globals
accepted_weights_types = ["core", "accessory", "euclidean"]
betweenness_sample_default = 100
default_length_sigma = 5
default_prop_n = 0.1
default_max_zero = 0.05
default_max_a_dist = 0.5
default_max_pi_dist = 0.1
default_max_merge = -1 # off

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
    mode.add_argument('--qc-db',
            help='Run quality control on a reference database',
            default=False,
            action='store_true')
    mode.add_argument('--fit-model',
            help='Fit a model to a (QCed) reference database',
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
    qcGroup.add_argument('--qc-keep', help="Only write failing sequences to a file, don't remove them from the database file",
                                      default=False, action='store_true')
    qcGroup.add_argument('--remove-samples', help='A list of names to remove from the database (regardless of any other QC)',
                                     default=None, type=str)
    qcGroup.add_argument('--retain-failures', help='Retain sketches of genomes that do not pass QC filters in '
                                                   'separate database [default = False]', default=False, action='store_true')
    qcGroup.add_argument('--max-a-dist', help=f"Maximum accessory distance to permit [default = {default_max_a_dist}]",
                                                default = default_max_a_dist, type = float)
    qcGroup.add_argument('--max-pi-dist', help=f"Maximum core distance to permit [default = {default_max_pi_dist}]",
                                                default = default_max_pi_dist, type = float)
    qcGroup.add_argument('--max-zero-dist', help=f"Maximum proportion of zero distances to permit [default = {default_max_zero}]",
                                                default = default_max_zero, type = float)
    qcGroup.add_argument('--type-isolate', help='Isolate from which distances will be calculated for pruning [default = None]',
                                                default = None, type = str)
    qcGroup.add_argument('--length-sigma', help=f"Number of standard deviations of length distribution beyond "
                                                f"which sequences will be excluded [default = {default_length_sigma}]", default = default_length_sigma, type = int)
    qcGroup.add_argument('--length-range', help='Allowed length range, outside of which sequences will be excluded '
                                                '[two values needed - lower and upper bounds]', default=[None,None],
                                                type = int, nargs = 2)
    qcGroup.add_argument('--prop-n', help=f"Threshold ambiguous base proportion above which sequences will be excluded"
                                          f" [default = {default_prop_n}]", default = default_prop_n,
                                                type = float)
    qcGroup.add_argument('--upper-n', help='Threshold ambiguous base count above which sequences will be excluded',
                                                default=None, type = int)

    # model fitting
    modelGroup = parser.add_argument_group('Model fit options')
    modelGroup.add_argument('--model-subsample',
                            help='Number of pairwise distances used to fit model [default = 100000]',
                            type=int,
                            default=100000)
    modelGroup.add_argument('--assign-subsample',
                            help='Number of pairwise distances in each assignment batch [default = 5000]',
                            type=int,
                            default=5000)
    modelGroup.add_argument('--for-refine',
                            help='Fit a BGMM or DBSCAN model without assigning all points to initialise a refined model',
                            default=False,
                            action='store_true')
    modelGroup.add_argument('--K',
                            help='Maximum number of mixture components [default = 2]',
                            type=int,
                            default=2)
    modelGroup.add_argument('--D',
                            help='Maximum number of clusters in DBSCAN fitting [default = 100]',
                            type=int,
                            default=100)
    modelGroup.add_argument('--min-cluster-prop',
                            help='Minimum proportion of points in a cluster '
                                 'in DBSCAN fitting [default = 0.0001]',
                            type=float,
                            default=0.0001)
    modelGroup.add_argument('--threshold',
                            help='Cutoff if using --fit-model threshold',
                            type=float)

    # model refinement
    refinementGroup = parser.add_argument_group('Network analysis and model refinement options')
    refinementGroup.add_argument('--pos-shift', help='Maximum amount to move the boundary right past between-strain mean',
            type=float, default = 0)
    refinementGroup.add_argument('--neg-shift', help='Maximum amount to move the boundary left past within-strain mean]',
            type=float, default = 0)
    refinementGroup.add_argument('--manual-start', help='A file containing information for a start point. '
            'See documentation for help.', default=None)
    refinementGroup.add_argument('--model-dir', help='Directory containing model to use for assigning queries '
                                                     'to clusters [default = reference database directory]', type = str)
    refinementGroup.add_argument('--score-idx',
            help='Index of score to use [default = 0]',
            type=int, default = 0, choices=[0, 1, 2])
    refinementGroup.add_argument('--summary-sample',
            help='Number of sequences used to estimate graph properties [default = all]',
            type=int, default = None)
    refinementGroup.add_argument('--betweenness-sample',
            help='Number of sequences used to estimate betweeness with a GPU [default = 100]',
            type = int, default = betweenness_sample_default)
    refineMode = refinementGroup.add_mutually_exclusive_group()
    refineMode.add_argument('--unconstrained',
            help='Optimise both boundary gradient and intercept',
            default=False, action='store_true')
    refineMode.add_argument('--multi-boundary',
            help='Produce multiple sets of clusters at different boundary positions. This argument sets the'
                 'number of boundary positions between n-1 clusters and the refine optimum.',
            type=int, default=0)
    refineMode.add_argument('--indiv-refine', help='Also run refinement for core and accessory individually',
            choices=['both', 'core', 'accessory'], default=None)

    # lineage clustering within strains
    lineagesGroup = parser.add_argument_group('Lineage analysis options')
    lineagesGroup.add_argument('--ranks',
                                help='Comma separated list of ranks used in lineage clustering [default = 1,2,3]',
                                type = str,
                                default = "1,2,3")
    lineagesGroup.add_argument('--count-unique-distances',
                                help='kNN enumerates number of unique distances rather than number of '
                                'neighbours',
                                action = 'store_true',
                                default = False)
    lineagesGroup.add_argument('--reciprocal-only',
                                help='Only use reciprocal kNN matches for lineage definitions',
                                action = 'store_true',
                                default = False)
    lineagesGroup.add_argument('--max-search-depth',
                                help='Number of kNN distances per sequence to filter when '
                                      'counting neighbours or using only reciprocal matches',
                                type = int,
                                default = None)
    lineagesGroup.add_argument('--write-lineage-networks',
                                help='Save all lineage networks',
                                action = 'store_true',
                                default = False)
    lineagesGroup.add_argument('--use-accessory',
                                help='Use accessory distances for lineage definitions [default = use core distances]',
                                action = 'store_true',
                                default = False)

    other = parser.add_argument_group('Other options')
    other.add_argument('--threads', default=1, type=int, help='Number of threads to use [default = 1]')
    other.add_argument('--gpu-sketch', default=False, action='store_true', help='Use a GPU when calculating sketches (read data only) [default = False]')
    other.add_argument('--gpu-dist', default=False, action='store_true', help='Use a GPU when calculating distances [default = False]')
    other.add_argument('--gpu-model', default=False, action='store_true', help='Use a GPU when fitting a model [default = False]')
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
    from .models import loadClusterFit, BGMMFit, DBSCANFit, RefineFit, LineageFit
    from .sketchlib import checkSketchlibLibrary, removeFromDB

    from .network import construct_network_from_edge_list
    from .network import construct_network_from_assignments
    from .network import extractReferences
    from .network import printClusters
    from .network import save_network
    from .network import checkNetworkVertexCount

    from .plot import writeClusterCsv
    from .plot import plot_scatter
    from .plot import plot_database_evaluations

    from .qc import prune_distance_matrix, qcDistMat, sketchlibAssemblyQC, remove_qc_fail

    from .utils import setGtThreads
    from .utils import setupDBFuncs
    from .utils import readPickle, storePickle
    from .utils import createOverallLineage
    from .utils import get_match_search_depth
    from .utils import check_and_set_gpu

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

    # Dict of DB access functions
    dbFuncs = setupDBFuncs(args)
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

    # check if working with lineages
    if args.fit_model == 'lineage':
        rank_list = sorted([int(x) for x in args.ranks.split(',')])
        if int(min(rank_list)) == 0:
            sys.stderr.write("Ranks must be greater than 1\n")
        if args.max_search_depth is not None and args.max_search_depth < max(rank_list):
            sys.stderr.write("The maximum search depth must be greater than the highest lineage rank\n")
            sys.exit(1)

    if args.create_db == False:
        # Check and set required parameters for other modes
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

    # run according to mode
    sys.stderr.write("PopPUNK (POPulation Partitioning Using Nucleotide Kmers)\n")
    sys.stderr.write("\t(with backend: " + dbFuncs['backend'] + " v" + dbFuncs['backend_version'] + "\n")
    sys.stderr.write("\t sketchlib: " + checkSketchlibLibrary() + ")\n")

    # Check on parallelisation of graph-tools
    setGtThreads(args.threads)

    # Check on initialisation of GPU libraries and memory
    try:
        import cupyx
        import cugraph
        import cudf
        import cupy as cp
        from numba import cuda
        import rmm
        gpu_lib = True
    except ImportError as e:
        gpu_lib = False
    args.gpu_graph = check_and_set_gpu(args.gpu_graph,
                                        gpu_lib,
                                        quit_on_fail = True)

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
        seq_names = constructDatabase(
            args.r_files,
            kmers,
            sketch_sizes,
            args.output,
            args.threads,
            args.overwrite,
            codon_phased = args.codon_phased,
            calc_random = True)

        # calculate distances between sequences
        distMat = queryDatabase(rNames = seq_names,
                                qNames = seq_names,
                                dbPrefix = args.output,
                                queryPrefix = args.output,
                                klist = kmers,
                                self = True,
                                number_plot_fits = args.plot_fit,
                                threads = args.threads)
        storePickle(seq_names, seq_names, True, distMat, f"{args.output}/{os.path.basename(args.output)}.dists")

        # Plot results
        if not args.no_plot:
            plot_scatter(distMat,
                         args.output,
                         args.output + " distances")
            plot_database_evaluations(args.output)

    #******************************#
    #*                            *#
    #* DB QC                      *#
    #*                            *#
    #******************************#
    if args.qc_db:
        # if a length range is specified, check it makes sense
        if args.length_range[0] is not None:
            if args.length_range[0] >= args.length_range[1]:
                sys.stderr.write('Ensure the specified length range is space-separated argument of'
                                ' length 2, with the lower value first\n')
                sys.exit(1)

        # Dict of QC options for passing to database construction and querying functions
        qc_dict = {
            'no_remove': args.qc_keep,
            'retain_failures': args.retain_failures,
            'length_sigma': args.length_sigma,
            'length_range': args.length_range,
            'prop_n': args.prop_n,
            'upper_n': args.upper_n,
            'max_pi_dist': args.max_pi_dist,
            'max_a_dist': args.max_a_dist,
            'prop_zero': args.max_zero_dist,
            'type_isolate': args.type_isolate
        }

        refList, queryList, self, distMat = readPickle(distances, enforce_self=True)

        fail_unconditionally = {}
        # Unconditional removal
        if args.remove_samples:
            with open(args.remove_samples, 'r') as f:
                for line in f:
                    sample_to_remove = line.rstrip()
                    if sample_to_remove in refList:
                        fail_unconditionally[sample_to_remove] = ["removed"]

        # assembly qc
        pass_assembly_qc, fail_assembly_qc = \
            sketchlibAssemblyQC(args.ref_db,
                                refList,
                                qc_dict)
        sys.stderr.write(f"{len(fail_assembly_qc)} samples failed\n")

        # QC pairwise distances to identify long distances indicative of anomalous sequences in the collection
        pass_dist_qc, fail_dist_qc = \
            qcDistMat(distMat,
                      refList,
                      queryList,
                      args.ref_db,
                      qc_dict)
        sys.stderr.write(f"{len(fail_dist_qc)} samples failed\n")

        # Get list of passing samples
        pass_list = set(refList) - fail_unconditionally.keys() - fail_assembly_qc.keys() - fail_dist_qc.keys()
        assert(pass_list == (set(refList) - fail_unconditionally.keys()).intersection(set(pass_assembly_qc)).intersection(set(pass_dist_qc)))
        passed = [x for x in refList if x in pass_list]
        if qc_dict['type_isolate'] is not None and qc_dict['type_isolate'] not in pass_list:
            raise RuntimeError('Type isolate ' + qc_dict['type_isolate'] + \
                               ' not found in isolates after QC; check '
                               'name of type isolate and QC options\n')
        
        sys.stderr.write(f"{len(passed)} samples passed QC\n")
        if len(passed) < len(refList):
            remove_qc_fail(qc_dict, refList, passed,
                           [fail_unconditionally, fail_assembly_qc, fail_dist_qc],
                           args.ref_db, distMat, output,
                           args.strand_preserved, args.threads,
                           args.gpu_graph)

        # Plot results
        if not args.no_plot:
            plot_scatter(distMat,
                         output,
                         output + " distances")
            plot_database_evaluations(output)

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
                                   output,
                                   use_gpu = args.gpu_graph)
            model.set_threads(args.threads)
            sys.stderr.write("Loaded previous model of type: " + model.type + "\n")
            if args.fit_model == "refine" and args.manual_start == None \
                and model.type != 'bgmm' and model.type != 'dbscan':
                sys.stderr.write("Model needs to be from BGMM or DBSCAN to refine\n")
                sys.exit(1)

        # Load the distances
        refList, queryList, self, distMat = readPickle(distances, enforce_self=True)

        #******************************#
        #*                            *#
        #* model fit                  *#
        #*                            *#
        #******************************#
        # Run selected model here, or if easy run DBSCAN followed by refinement
        if args.fit_model:
            # Run DBSCAN model
            if args.fit_model == "dbscan":
                model = DBSCANFit(output,
                                  max_samples = args.model_subsample,
                                  max_batch_size = args.assign_subsample,
                                  assign_points = not args.for_refine)
                model.set_threads(args.threads)
                assignments = model.fit(distMat,
                                        args.D,
                                        args.min_cluster_prop,
                                        args.gpu_model)
            # Run Gaussian model
            elif args.fit_model == "bgmm":
                model = BGMMFit(output,
                                max_samples = args.model_subsample,
                                max_batch_size = args.assign_subsample,
                                assign_points = not args.for_refine)
                model.set_threads(args.threads)
                assignments = model.fit(distMat,
                                        args.K)
            elif args.fit_model == "refine":
                new_model = RefineFit(output)
                new_model.set_threads(args.threads)
                assignments = new_model.fit(distMat, refList, model,
                                            args.pos_shift, args.neg_shift,
                                            args.manual_start,
                                            args.indiv_refine,
                                            args.unconstrained,
                                            args.multi_boundary,
                                            args.score_idx,
                                            args.no_local,
                                            args.betweenness_sample,
                                            args.summary_sample,
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
                # Memory usage determined by maximum search depth
                if args.max_search_depth is not None:
                    max_search_depth = int(args.max_search_depth)
                elif args.max_search_depth is None and (args.reciprocal_only or args.count_unique_distances):
                    max_search_depth = get_match_search_depth(refList,rank_list)
                else:
                    max_search_depth = max(rank_list)

                model = LineageFit(output,
                                    rank_list,
                                    max_search_depth,
                                    args.reciprocal_only,
                                    args.count_unique_distances,
                                    1 if args.use_accessory else 0,
                                    use_gpu = args.gpu_graph)
                model.set_threads(args.threads)
                model.fit(distMat,
                            args.use_accessory)

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

        # end here if not assigning data
        if args.for_refine:
            sys.stderr.write('Initial model fit complete; points will be assigned when this model is refined\nusing "--fit-model refine"\n')
            sys.exit(0)

        #******************************#
        #*                            *#
        #* network construction       *#
        #*                            *#
        #******************************#
        if model.type != "lineage":
            if args.graph_weights:
                weights_type = 'euclidean'
            else:
                weights_type = None
            genomeNetwork = \
                construct_network_from_assignments(refList,
                                                     queryList,
                                                     assignments,
                                                     model.within_label,
                                                     distMat = distMat,
                                                     weights_type = weights_type,
                                                     sample_size = args.summary_sample,
                                                     betweenness_sample = args.betweenness_sample,
                                                     use_gpu = args.gpu_graph)
        else:
            # Lineage fit requires some iteration
            indivNetworks = {}
            lineage_clusters = defaultdict(dict)
            for rank in sorted(rank_list):
                if args.graph_weights:
                    weights = model.edge_weights(rank)
                else:
                    weights = None
                indivNetworks[rank] = construct_network_from_edge_list(refList,
                                                                        refList,
                                                                        assignments[rank],
                                                                        weights = weights,
                                                                        sample_size = args.summary_sample,
                                                                        betweenness_sample = args.betweenness_sample,
                                                                        use_gpu = args.gpu_graph,
                                                                        summarise = False
                                                                       )
                # Print individual networks if requested
                if args.write_lineage_networks:
                    save_network(indivNetworks[rank],
                                    prefix = output,
                                    suffix = '_rank_' + str(rank) + '_graph',
                                    use_gpu = args.gpu_graph)

                # Identify clusters from output
                lineage_clusters[rank] = \
                    printClusters(indivNetworks[rank],
                                  refList,
                                  printCSV = False,
                                  use_gpu = args.gpu_graph)
                n_clusters = max(lineage_clusters[rank].values())
                sys.stderr.write("Network for rank " + str(rank) + " has " +
                                 str(n_clusters) + " lineages\n")

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

        # Save network
        save_network(genomeNetwork, prefix = output, suffix = "_graph", use_gpu = args.gpu_graph)

        # Write core and accessory based clusters, if they worked
        if model.indiv_fitted:
            indivNetworks = {}
            for dist_type, slope in zip(['core', 'accessory'], [0, 1]):
                if args.indiv_refine == 'both' or args.indiv_refine == dist_type:
                    indivAssignments = model.assign(distMat, slope = slope)
                    indivNetworks[dist_type] = \
                        construct_network_from_assignments(refList,
                                                             queryList,
                                                             indivAssignments,
                                                             model.within_label,
                                                             sample_size = args.summary_sample,
                                                             betweenness_sample = args.betweenness_sample,
                                                             use_gpu = args.gpu_graph)
                    isolateClustering[dist_type] = \
                        printClusters(indivNetworks[dist_type],
                                      refList,
                                      output + "/" + os.path.basename(output) + "_" + dist_type,
                                      externalClusterCSV = args.external_clustering,
                                      use_gpu = args.gpu_graph)
                    save_network(indivNetworks[dist_type],
                                    prefix = output,
                                    suffix = '_' + dist_type + '_graph',
                                    use_gpu = args.gpu_graph)

        #******************************#
        #*                            *#
        #* clique pruning             *#
        #*                            *#
        #******************************#
        # extract limited references from clique by default
        # (this no longer loses information and should generally be kept on)
        if model.type != "lineage":
            dist_type_list = ['original']
            dist_string_list = ['']
            if model.indiv_fitted:
                if args.indiv_refine == 'both' or args.indiv_refine == 'core':
                    dist_type_list.append('core')
                    dist_string_list.append('_core')
                if args.indiv_refine == 'both' or args.indiv_refine == 'accessory':
                    dist_type_list.append('accessory')
                    dist_string_list.append('_accessory')
            # Iterate through different network types
            for dist_type, dist_string in zip(dist_type_list, dist_string_list):
                if dist_type == 'original':
                    network_for_refs = genomeNetwork
                elif dist_type == 'core':
                    network_for_refs = indivNetworks[dist_type]
                elif dist_type == 'accessory':
                    network_for_refs = indivNetworks[dist_type]
                newReferencesIndices, newReferencesNames, newReferencesFile, genomeNetwork = \
                    extractReferences(network_for_refs,
                                        refList,
                                        output,
                                        outSuffix = dist_string,
                                        type_isolate = args.type_isolate,
                                        threads = args.threads,
                                        use_gpu = args.gpu_graph)
                nodes_to_remove = set(range(len(refList))).difference(newReferencesIndices)
                names_to_remove = [refList[n] for n in nodes_to_remove]

                if (len(names_to_remove) > 0):
                    # Save reference distances
                    dists_suffix = dist_string + '.refs.dists'
                    prune_distance_matrix(refList, names_to_remove, distMat,
                                          output + "/" + os.path.basename(output) + dists_suffix)
                    # Save reference network
                    graphs_suffix = dist_string + '.refs_graph'
                    save_network(genomeNetwork,
                                    prefix = output,
                                    suffix = graphs_suffix,
                                    use_gpu = args.gpu_graph)
                    db_suffix = dist_string + '.refs.h5'
                    removeFromDB(args.ref_db, output, names_to_remove)
                    os.rename(output + "/" + os.path.basename(output) + '.tmp.h5',
                              output + "/" + os.path.basename(output) + db_suffix)

    sys.stderr.write("\nDone\n")

if __name__ == '__main__':
    main()

    sys.exit(0)
