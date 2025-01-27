#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018-2023 John Lees and Nick Croucher

# universal
from operator import itemgetter
import os
import sys
import warnings
# additional
import numpy as np
from collections import defaultdict
import h5py

# required from v2.1.1 onwards (no mash support)
import pp_sketchlib

# import poppunk package
from .__init__ import __version__
from .__main__ import default_prop_n, default_length_sigma, default_max_a_dist, \
    default_max_pi_dist, default_max_zero, default_max_merge

#******************************#
#*                            *#
#* Command line parsing       *#
#*                            *#
#******************************#
def get_options():

    import argparse

    parser = argparse.ArgumentParser(description='Assign isolates to strains (by POPulation Partitioning Using Nucleotide Kmers)',
                                     prog='poppunk_assign')

    # input options
    iGroup = parser.add_argument_group('Input files')
    iGroup.add_argument('--db', required=True, type = str, help='Location of built reference database')
    iGroup.add_argument('--query', required=True, help='File listing query input assemblies')
    iGroup.add_argument('--distances', help='Prefix of input pickle of pre-calculated distances (if not in --db)')
    iGroup.add_argument('--external-clustering', help='File with cluster definitions or other labels '
                                                      'generated with any other method.', default=None)

    # output options
    oGroup = parser.add_argument_group('Output options')
    oGroup.add_argument('--output', required=True, help='Prefix for output files (required)')
    oGroup.add_argument('--plot-fit', help='Create this many plots of some fits relating k-mer to core/accessory distances '
                                            '[default = 0]', default=0, type=int)
    oGroup.add_argument('--write-references', help='Write reference database isolates\' cluster assignments out too',
                                              default=False, action='store_true')
    oGroup.add_argument('--update-db', help='Update reference database with query sequences. Full: pick new references from cliques. Fast: pick random new references.', default=False, choices=['full', 'fast', False])
    oGroup.add_argument('--overwrite', help='Overwrite any existing database files', default=False, action='store_true')
    oGroup.add_argument('--graph-weights', help='Save within-strain Euclidean distances into the graph', default=False, action='store_true')
    oGroup.add_argument('--save-partial-query-graph', help='Save only the network components to which queries are assigned', default=False, action='store_true')

    # comparison metrics
    kmerGroup = parser.add_argument_group('Kmer comparison options')
    kmerGroup.add_argument('--min-kmer-count', default=0, type=int, help='Minimum k-mer count when using reads as input [default = 0]')
    kmerGroup.add_argument('--exact-count', default=False, action='store_true',
                           help='Use the exact k-mer counter with reads '
                                '[default = use countmin counter]')
    kmerGroup.add_argument('--strand-preserved', default=False, action='store_true',
                           help='Treat input as being on the same strand, and ignore reverse complement '
                                'k-mers [default = use canonical k-mers]')

    # qc options
    qcGroup = parser.add_argument_group('Quality control options for distances')
    qcGroup.add_argument('--run-qc', help="Run the QC steps", default=False, action="store_true")
    qcGroup.add_argument('--retain-failures', help='Retain sketches of genomes that do not pass QC filters in '
                                                'separate database [default = False]', default=False, action='store_true')
    qcGroup.add_argument('--max-a-dist', help=f"Maximum accessory distance to permit [default = {default_max_a_dist}]",
                                                default = default_max_a_dist, type = float)
    qcGroup.add_argument('--max-pi-dist', help=f"Maximum core distance to permit [default = {default_max_pi_dist}]",
                                                default = default_max_pi_dist, type = float)
    qcGroup.add_argument('--max-zero-dist', help=f"Maximum proportion of zero distances to permit [default = {default_max_zero}]",
                                                default = default_max_zero, type = float)
    qcGroup.add_argument('--max-merge', help=f"Maximum number of cluster merges a sample can cause [default = {default_max_merge}]",
                                                default = default_max_merge, type = int)
    qcGroup.add_argument('--betweenness', default=False, action='store_true',
                               help='Report the betweenness of all the query nodes [default = False]')
    qcGroup.add_argument('--type-isolate', help='Isolate from which distances can be calculated for pruning [default = None]',
                                                default = None, type = str)
    qcGroup.add_argument('--length-sigma', help='Number of standard deviations of length distribution beyond '
                                                'which sequences will be excluded [default = 5]', default = None, type = int)
    qcGroup.add_argument('--length-range', help='Allowed length range, outside of which sequences will be excluded '
                                                '[two values needed - lower and upper bounds]', default=[None,None],
                                                type = int, nargs = 2)
    qcGroup.add_argument('--prop-n', help='Threshold ambiguous base proportion above which sequences will be excluded'
                                                ' [default = None]', default = None,
                                                type = float)
    qcGroup.add_argument('--upper-n', help='Threshold ambiguous base count above which sequences will be excluded',
                                                default=None, type = int)

    # sequence querying
    queryingGroup = parser.add_argument_group('Database querying options')
    queryingGroup.add_argument('--serial', default=False, action='store_true',
                               help='Do assignment one-by-one, not in batches (see docs) [default = False]')
    queryingGroup.add_argument('--stable', default=None, choices=['core', 'accessory', False],
                               help='\'Stable nomenclature\': do assignment one-by-one, to nearest neighbours (see docs) [default = False]')
    queryingGroup.add_argument('--model-dir', help='Directory containing model to use for assigning queries '
                                                   'to clusters [default = reference database directory]', type = str)
    queryingGroup.add_argument('--previous-clustering', help='Directory containing previous cluster definitions '
                                                             'and network [default = use that in the directory '
                                                             'containing the model]', type = str)
    queryingGroup.add_argument('--core', help='(with a \'refine\' model) '
                                                   'Use a core-distance only model for assigning queries '
                                                   '[default = False]', default=False, action='store_true')
    queryingGroup.add_argument('--accessory', help='(with a \'refine\' or \'lineage\' model) '
                                                        'Use an accessory-distance only model for assigning queries '
                                                        '[default = False]', default=False, action='store_true')
    queryingGroup.add_argument('--use-full-network', help='Use full network rather than reference network for querying [default = False]',
                                                    default = False,
                                                    action = 'store_true')

    # processing
    other = parser.add_argument_group('Other options')
    other.add_argument('--threads', default=1, type=int, help='Number of threads to use [default = 1]')
    other.add_argument('--gpu-sketch', default=False, action='store_true', help='Use a GPU when calculating sketches (read data only) [default = False]')
    other.add_argument('--gpu-dist', default=False, action='store_true', help='Use a GPU when calculating distances [default = False]')
    other.add_argument('--gpu-graph', default=False, action='store_true', help='Use a GPU when constructing networks [default = False]')
    other.add_argument('--deviceid', default=0, type=int, help='CUDA device ID, if using GPU [default = 0]')
    other.add_argument('--version', action='version',
                       version='%(prog)s '+__version__)
    other.add_argument('--citation',
                       action='store_true',
                       default=False,
                       help='Give a citation, and possible methods paragraph'
                            ' based on the command line')


    # combine
    args = parser.parse_args()

    # ensure directories do not have trailing forward slash
    for arg in [args.db, args.model_dir, args.output, args.previous_clustering]:
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
    args = get_options()

    # May just want to print the citation
    if args.citation:
        from .citation import print_citation
        print_citation(args, assign=True)
        sys.exit(0)

    from .sketchlib import checkSketchlibLibrary
    from .utils import setGtThreads
    from .utils import setupDBFuncs

    # Dict of QC options for passing to database construction and querying functions
    if args.run_qc:
        # define defaults if one QC parameter given
        # length_sigma
        if args.length_sigma is not None:
            length_sigma = args.length_sigma
        elif None in args.length_range:
            length_sigma = default_length_sigma
        else:
            length_sigma = None
        # prop_n
        if args.prop_n is not None:
            prop_n = args.prop_n
        elif args.upper_n is None:
            prop_n = default_prop_n
        else:
            prop_n = None

        qc_dict = {
            'run_qc': True,
            'retain_failures': args.retain_failures,
            'length_sigma': length_sigma,
            'length_range': args.length_range,
            'prop_n': prop_n,
            'upper_n': args.upper_n,
            'max_pi_dist': args.max_pi_dist,
            'max_a_dist': args.max_a_dist,
            'prop_zero': args.max_zero_dist,
            'max_merge': args.max_merge,
            'betweenness': args.betweenness,
            'type_isolate': args.type_isolate
        }
    else:
        qc_dict = {'run_qc': False, 'type_isolate': None }

    # Dict of DB access functions for assign_query (which is out of scope)
    dbFuncs = setupDBFuncs(args)

    sys.stderr.write("PopPUNK: assign\n")
    sys.stderr.write("\t(with backend: " + dbFuncs['backend'] + " v" + dbFuncs['backend_version'] + "\n")
    sys.stderr.write('\t sketchlib: ' + checkSketchlibLibrary() + ')\n')
    sys.stderr.write("Mode: Assigning clusters of query sequences\n\n")

    # Check on parallelisation of graph-tools
    setGtThreads(args.threads)

    if args.distances is None:
        distances = args.db + "/" + os.path.basename(args.db) + ".dists"
    else:
        distances = args.distances

    #*******************************#
    #*                             *#
    #* query assignment            *#
    #*                             *#
    #*******************************#
    assign_query(dbFuncs,
                 args.db,
                 args.query,
                 args.output,
                 qc_dict,
                 args.update_db,
                 args.write_references,
                 distances,
                 args.serial,
                 args.stable,
                 args.threads,
                 args.overwrite,
                 args.plot_fit,
                 args.graph_weights,
                 args.model_dir,
                 args.strand_preserved,
                 args.previous_clustering,
                 args.external_clustering,
                 args.core,
                 args.accessory,
                 args.gpu_sketch,
                 args.gpu_dist,
                 args.gpu_graph,
                 args.deviceid,
                 args.save_partial_query_graph,
                 args.use_full_network)

    sys.stderr.write("\nDone\n")

#*******************************#
#*                             *#
#* query assignment            *#
#*                             *#
#*******************************#
def assign_query(dbFuncs,
                 ref_db,
                 q_files,
                 output,
                 qc_dict,
                 update_db,
                 write_references,
                 distances,
                 serial,
                 stable,
                 threads,
                 overwrite,
                 plot_fit,
                 graph_weights,
                 model_dir,
                 strand_preserved,
                 previous_clustering,
                 external_clustering,
                 core,
                 accessory,
                 gpu_sketch,
                 gpu_dist,
                 gpu_graph,
                 deviceid,
                 save_partial_query_graph,
                 use_full_network):
    """Code for assign query mode for CLI"""
    createDatabaseDir = dbFuncs['createDatabaseDir']
    constructDatabase = dbFuncs['constructDatabase']
    readDBParams = dbFuncs['readDBParams']

    if ref_db == output and overwrite == False:
        sys.stderr.write("--output and --db must be different to "
                         "prevent overwrite.\n")
        sys.exit(1)

    # Find distances to reference db
    kmers, sketch_sizes, codon_phased = readDBParams(ref_db)

    # construct database
    createDatabaseDir(output, kmers)
    qNames = constructDatabase(q_files,
                                kmers,
                                sketch_sizes,
                                output,
                                threads,
                                overwrite,
                                codon_phased = codon_phased,
                                calc_random = False,
                                use_gpu = gpu_sketch,
                                deviceid = deviceid)

    isolateClustering = assign_query_hdf5(dbFuncs,
                    ref_db,
                    qNames,
                    output,
                    qc_dict,
                    update_db,
                    write_references,
                    distances,
                    serial,
                    stable,
                    threads,
                    overwrite,
                    plot_fit,
                    graph_weights,
                    model_dir,
                    strand_preserved,
                    previous_clustering,
                    external_clustering,
                    core,
                    accessory,
                    gpu_dist,
                    gpu_graph,
                    save_partial_query_graph,
                    use_full_network)
    return(isolateClustering)

def assign_query_hdf5(dbFuncs,
                 ref_db,
                 qNames,
                 output,
                 qc_dict,
                 update_db,
                 write_references,
                 distances,
                 serial,
                 stable,
                 threads,
                 overwrite,
                 plot_fit,
                 graph_weights,
                 model_dir,
                 strand_preserved,
                 previous_clustering,
                 external_clustering,
                 core,
                 accessory,
                 gpu_dist,
                 gpu_graph,
                 save_partial_query_graph,
                 use_full_network):
    """Code for assign query mode taking hdf5 as input. Written as a separate function so it can be called
    by web APIs"""
    # Modules imported here as graph tool is very slow to load (it pulls in all of GTK?)
    from tqdm import tqdm
    from .models import loadClusterFit

    from .sketchlib import removeFromDB

    from .network import fetchNetwork
    from .network import construct_network_from_edge_list
    from .network import extractReferences
    from .network import addQueryToNetwork
    from .network import printClusters
    from .network import save_network
    from .network import get_vertex_list
    from .network import printExternalClusters
    from .network import vertex_betweenness
    from .network import remove_non_query_components
    from .qc import sketchlibAssemblyQC

    from .plot import writeClusterCsv

    from .qc import qcDistMat, qcQueryAssignments, \
        prune_query_distance_matrix, write_qc_failure_report

    from .sketchlib import addRandom

    from .utils import storePickle
    from .utils import readPickle
    from .utils import createOverallLineage

    failed_assembly_qc = {}
    failed_assembly_samples = frozenset()
    if qc_dict["run_qc"]:
        pass_assembly_qc, failed_assembly_qc = \
                sketchlibAssemblyQC(output,
                                    qNames,
                                    qc_dict)
        failed_assembly_samples = frozenset(failed_assembly_qc.keys())
    if len(failed_assembly_qc) > 0:
        sys.stderr.write(f"{len(failed_assembly_qc)} samples failed:\n"
                            f"{','.join(failed_assembly_samples)}\n")

        qNames = [x for x in qNames if x in pass_assembly_qc]
        if len(qNames) == 0:
            write_qc_failure_report(failed_assembly_samples, [failed_assembly_qc], output)
            sys.exit(1)

    joinDBs = dbFuncs['joinDBs']
    queryDatabase = dbFuncs['queryDatabase']
    readDBParams = dbFuncs['readDBParams']
    getSeqsInDb = dbFuncs['getSeqsInDb']

    if ref_db == output and overwrite == False:
        sys.stderr.write("--output and --db must be different to "
                         "prevent overwrite.\n")
        sys.exit(1)
    if (update_db and not distances):
        sys.stderr.write("--update-db requires --distances to be provided\n")
        sys.exit(1)
    if stable is not None:
        serial = True
    if serial and update_db:
        raise RuntimeError("--update-db cannot be used with --serial or --stable")

    # Load the previous model
    model_prefix = ref_db
    if model_dir is not None:
        model_prefix = model_dir
    model_file = model_prefix + "/" + os.path.basename(model_prefix) + "_fit"

    model = loadClusterFit(model_file + '.pkl',
                           model_file + '.npz')
    if model.type == "lineage" and serial:
        raise RuntimeError("lineage models cannot be used with --serial or --stable")
    model.set_threads(threads)

    # Only proceed with a fully-fitted model
    if not model.fitted or (hasattr(model,'assign_points') and model.assign_points == False):
        sys.stderr.write('Cannot assign points with an incompletely-fitted model\n'
                         'Please refine this initial fit with "--fit-model refine"\n')
        sys.exit(1)

    # Set directories of previous fit
    if previous_clustering is not None:
        prev_clustering = previous_clustering
    else:
        prev_clustering = model_prefix

    # Find distances to reference db
    kmers = readDBParams(ref_db)[0]

    # Iterate through different types of model fit with a refined model when specified
    # Core and accessory assignments use the same model and same overall set of distances
    # but have different networks, references, reference distances and assignments
    fit_type_list = ['default']
    if model.type == 'refine' and model.indiv_fitted:
        if core:
            fit_type_list.append('core_refined')
        if accessory:
            fit_type_list.append('accessory_refined')

    for fit_type in fit_type_list:
        # Define file name extension
        file_extension_string = ''
        if fit_type != 'default':
            file_extension_string = '_' + fit_type

        if os.path.isfile(distances + ".pkl"):
            rNames = readPickle(distances, enforce_self = True, distances=False)[0]
        elif update_db:
            sys.stderr.write("Distance order .pkl missing, cannot use --update-db\n")
            sys.exit(1)
        else:
            rNames = getSeqsInDb(os.path.join(ref_db, os.path.basename(ref_db) + ".h5"))

        # Take just the ref seqs...
        ref_file_name = os.path.join(model_prefix,
                        os.path.basename(model_prefix) + file_extension_string + ".refs")
        if not os.path.isfile(ref_file_name):
            warnings.warn(f"Could not find refs for {fit_type}: {ref_file_name}\n", stacklevel=2)
            warnings.warn("Supply a model directory with .refs file, or assignment will be slower\n", stacklevel=2)
        # ...unless a lineage model, they don't exist or doing a full update-db run
        use_ref_graph = \
            os.path.isfile(ref_file_name) and update_db != 'full' and model.type != 'lineage' and not use_full_network
        if use_ref_graph:
            refNames = list()
            with open(ref_file_name) as refFile:
                for reference in refFile:
                    refNames.append(reference.rstrip())
            rNames = [ref for ref in rNames if ref in frozenset(refNames)]

        # Deal with name clash
        same_names = set(rNames).intersection(qNames)
        if same_names:
            warnings.warn("Names of queries match names in reference database\n", stacklevel=2)
            if not write_references:
                sys.stderr.write("Not running -- change names or add --write-references to override this behaviour\n")
                sys.exit(1)
            else:
                query_db = h5py.File(output + "/" + os.path.basename(output) + ".h5", 'r+')
                print(output + "/" + os.path.basename(output))
                sketch_grp = query_db['sketches']
                for idx, query in enumerate(qNames):
                    if query in same_names:
                        new_name = query + "_query"
                        qNames[idx] = new_name
                        sketch_grp.move(query, new_name)
                query_db.close()

        if (fit_type == 'default' or (fit_type != 'default' and use_ref_graph)):
            # run query
            qrDistMat = queryDatabase(rNames = rNames,
                                      qNames = qNames,
                                      dbPrefix = ref_db,
                                      queryPrefix = output,
                                      klist = kmers,
                                      self = False,
                                      number_plot_fits = plot_fit,
                                      threads = threads,
                                      use_gpu = gpu_dist)

        # QC distance matrix
        if qc_dict['run_qc']:
            sys.stderr.write("Running QC on distance matrix\n")
            seq_names_passing, failed_distmatrix_qc = qcDistMat(qrDistMat, rNames, qNames, ref_db, qc_dict)
            failed_distmatrix_samples = frozenset(qNames) - frozenset(seq_names_passing)
            if len(failed_distmatrix_samples) > 0:
                sys.stderr.write(f"{len(failed_distmatrix_samples)} samples failed:\n"
                                 f"{','.join(failed_distmatrix_samples)}\n")
                write_qc_failure_report(failed_distmatrix_samples | failed_assembly_samples, [failed_distmatrix_qc, failed_assembly_qc], output)

                if len(failed_distmatrix_samples) == len(qNames):
                    sys.exit(1)
                else:
                    qNames, qrDistMat = \
                        prune_query_distance_matrix(rNames, qNames, failed_distmatrix_samples, qrDistMat)[0:2]

        if model.type == 'lineage':
            # Assign lineages by calculating query-query information
            addRandom(output, qNames, kmers, strand_preserved, overwrite, threads)
            qqDistMat = queryDatabase(rNames = qNames,
                                      qNames = qNames,
                                      dbPrefix = output,
                                      queryPrefix = output,
                                      klist = kmers,
                                      self = True,
                                      number_plot_fits = 0,
                                      threads = threads,
                                      use_gpu = gpu_dist)
            model.extend(qqDistMat, qrDistMat)

            genomeNetwork = {}
            isolateClustering = defaultdict(dict)
            for rank in model.ranks:
                assignment = model.assign(rank)
                # Overwrite the network loaded above
                if graph_weights:
                    weights = model.edge_weights(rank)
                else:
                    weights = None
                genomeNetwork[rank] = construct_network_from_edge_list(rNames + qNames,
                                                                       rNames + qNames,
                                                                       edge_list = assignment,
                                                                       weights = weights,
                                                                       use_gpu = gpu_graph,
                                                                       summarise = False)

                isolateClustering[rank] = \
                    printClusters(genomeNetwork[rank],
                                  rNames + qNames,
                                  printCSV = False,
                                  use_gpu = gpu_graph)[0]

            overall_lineage = createOverallLineage(model.ranks, isolateClustering)
            writeClusterCsv(
                output + "/" + os.path.basename(output) + '_lineages.csv',
                rNames + qNames,
                rNames + qNames,
                overall_lineage,
                output_format = 'phandango',
                epiCsv = None,
                queryNames = qNames,
                suffix = '_Lineage')

        else:
            genomeNetwork, old_cluster_file = \
                fetchNetwork(prev_clustering,
                             model,
                             rNames,
                             ref_graph = use_ref_graph,
                             core_only = (fit_type == 'core_refined'),
                             accessory_only = (fit_type == 'accessory_refined'),
                             use_gpu = gpu_graph)
            sys.stderr.write(f"Loading previous cluster assignments from {old_cluster_file}\n")

            n_vertices = len(get_vertex_list(genomeNetwork, use_gpu = gpu_graph))
            if n_vertices != len(rNames):
                sys.stderr.write(f"ERROR: There are {n_vertices} vertices in the network but {len(rNames)} reference names supplied; " + \
                                 "please check the '--model-dir' variable is pointing to the correct directory\n")
                sys.exit(1)

            # Assign these distances as within or between strain
            if fit_type == 'core_refined' or (model.type == 'refine' and model.threshold):
                queryAssignments = model.assign(qrDistMat, slope = 0)
                dist_type = 'core'
            elif fit_type == 'accessory_refined':
                queryAssignments = model.assign(qrDistMat, slope = 1)
                dist_type = 'accessory'
            else:
                queryAssignments = model.assign(qrDistMat)
                dist_type = 'euclidean'

            # QC assignments to check for multi-links
            if qc_dict['run_qc'] and qc_dict['max_merge'] > 1:
                sys.stderr.write("Running QC on model assignments\n")
                seq_names_passing = \
                    frozenset(qcQueryAssignments(rNames,
                                                 qNames,
                                                 queryAssignments,
                                                 qc_dict['max_merge'],
                                                 old_cluster_file)[0])
                failed_samples = frozenset(qNames) - seq_names_passing
                if len(failed_samples) > 0:
                    sys.stderr.write(f"{len(failed_samples)} samples failed:\n"
                                     f"{','.join(failed_samples)}\n")
                    if len(failed_samples) == len(qNames):
                        sys.exit(1)
                    else:
                        qNames, qrDistMat, queryAssignments = \
                            prune_query_distance_matrix(rNames, qNames,
                                failed_samples, qrDistMat, queryAssignments)

            # Assign clustering by adding to network
            if graph_weights:
                weights = qrDistMat
            else:
                weights = None

            output_fn = os.path.join(output, os.path.basename(output) + file_extension_string)
            if not serial:
                genomeNetwork, qqDistMat = \
                    addQueryToNetwork(dbFuncs,
                                        rNames,
                                        qNames,
                                        genomeNetwork,
                                        queryAssignments,
                                        model,
                                        output,
                                        kmers = kmers,
                                        distance_type = dist_type,
                                        queryQuery = update_db and fit_type == 'default',
                                        strand_preserved = strand_preserved,
                                        weights = weights,
                                        threads = threads,
                                        use_gpu = gpu_graph)
                if qc_dict['run_qc'] and qc_dict['betweenness']:
                    betweenness = vertex_betweenness(genomeNetwork)[len(rNames):len(rNames) + len(qNames)]
                    query_betweenness = {query: b for query, b in zip(qNames, betweenness)}
                    print("query\tbetweenness")
                    for query, q_betweenness in sorted(query_betweenness.items(), key=itemgetter(1), reverse=True):
                        print(f"{query}\t{q_betweenness}")

                isolateClustering, merged_queries = \
                                  printClusters(genomeNetwork,
                                                rNames + qNames,
                                                output_fn,
                                                old_cluster_file,
                                                external_clustering,
                                                write_references or update_db,
                                                use_gpu = gpu_graph)
                isolateClustering = {'combined': isolateClustering }
            else:
                if stable is not None:
                    # Some of this could be moved out higher up, e.g. we don't really need
                    # network load etc. But perhaps not such a bad thing to check the
                    # model being assigned to is valid
                    sys.stderr.write("Assigning stably\n")

                    # Load reference cluster assignments
                    from .utils import readIsolateTypeFromCsv
                    refClustering = readIsolateTypeFromCsv(old_cluster_file, mode = 'clusters', return_dict = True)['Cluster']
                    isolateClustering = {}

                    # Find neighbours
                    import poppunk_refine
                    if stable == "core":
                        dist_col = 0
                    else:
                        dist_col = 1
                    query_idxs, ref_idxs, _distance = \
                        poppunk_refine.get_kNN_distances(
                            distMat=qrDistMat[:, dist_col].reshape(len(qNames), len(rNames)),
                            kNN=1,
                            dist_col=dist_col,
                            num_threads=threads
                        )
                    # Assign queries same cluster as their NN, if the distance was
                    # within the same cluster
                    for (query, ref) in zip(query_idxs, ref_idxs):
                        if queryAssignments[query * len(rNames) + ref] == -1:
                            isolateClustering[qNames[query]] = refClustering[rNames[ref]]
                        else:
                            isolateClustering[qNames[query]] = "NA"
                else:
                    sys.stderr.write("Assigning serially\n")
                    G_copy = genomeNetwork.copy()
                    isolateClustering = {}
                    for idx, sample in tqdm(enumerate(qNames), total=len(qNames)):
                        genomeNetwork = \
                            addQueryToNetwork(dbFuncs,
                                            rNames,
                                            [sample],
                                            genomeNetwork,
                                            queryAssignments[(idx * len(rNames)):((idx + 1) * len(rNames))],
                                            model,
                                            output)[0]
                        isolate_cluster = \
                            printClusters(genomeNetwork,
                                                    rNames + [sample],
                                                    output_fn,
                                                    old_cluster_file,
                                                    external_clustering,
                                                    printRef=False,
                                                    printCSV=False,
                                                    write_unwords=False,
                                                    use_gpu = gpu_graph)[0]
                        cluster = int(isolate_cluster[sample])
                        if cluster > len(rNames):
                            cluster = "novel"
                        isolateClustering[sample] = cluster
                        # Reset for next sample
                        genomeNetwork = G_copy

                # Write out the results
                cluster_f = open(f"{output}/{os.path.basename(output)}_clusters.csv", 'w')
                cluster_f.write("Taxon,Cluster\n")
                for sample, cluster in isolateClustering.items():
                    cluster_f.write(",".join((sample, str(cluster))) + "\n")
                cluster_f.close()

                if external_clustering is not None:
                    printExternalClusters(isolateClustering, external_clustering,
                                            output, rNames, printRef=False)

        # Update DB as requested
        dists_out = output + "/" + os.path.basename(output) + ".dists"
        if update_db:
            sys.stderr.write("Updating reference database to " + output + "\n")
            # Update the network + ref list (everything) - no need to duplicate for core/accessory
            if fit_type == 'default':
                joinDBs(ref_db, output, output,
                        {"threads": threads, "strand_preserved": strand_preserved})
            sys.stderr.write("Saving model and network\n")
            if model.type == 'lineage':
                save_network(genomeNetwork[min(model.ranks)],
                                prefix = output,
                                suffix = '_graph',
                                use_gpu = gpu_graph)
                # Save sparse distance matrices and updated model
                model.outPrefix = os.path.basename(output)
                model.save()
            elif update_db == 'full':
                # Don't write the full graph with fast-update, as it's not a true
                # full graph
                graph_suffix = file_extension_string + '_graph'
                save_network(genomeNetwork,
                                prefix = output,
                                suffix = graph_suffix,
                                use_gpu = gpu_graph)

            # Copy model if needed
            if output != model.outPrefix and fit_type == 'default':
                model.copy(output)

            combined_seq = rNames + qNames
            storePickle(combined_seq, combined_seq, True, None, dists_out)

            # Clique pruning
            if model.type != 'lineage' and os.path.isfile(ref_file_name):
                sys.stderr.write(f"Finding references ({update_db})\n")
                existing_ref_list = []
                with open(ref_file_name) as refFile:
                    for reference in refFile:
                        existing_ref_list.append(reference.rstrip())

                # Extract references from graph
                newRepresentativesIndices, newRepresentativesNames, \
                    newRepresentativesFile, genomeNetwork = \
                        extractReferences(genomeNetwork,
                                            combined_seq,
                                            output,
                                            merged_queries,
                                            outSuffix = file_extension_string,
                                            existingRefs = existing_ref_list,
                                            type_isolate = qc_dict['type_isolate'],
                                            threads = threads,
                                            use_gpu = gpu_graph,
                                            fast_mode = update_db == "fast")

                # could also have newRepresentativesNames in this diff (should be the same) - but want
                # to ensure consistency with the network in case of bad input/bugs
                nodes_to_remove = set(range(len(combined_seq))).difference(newRepresentativesIndices)
                names_to_remove = [combined_seq[n] for n in nodes_to_remove]

                if (len(names_to_remove) > 0):
                    graph_suffix = file_extension_string + '.refs_graph'
                    save_network(genomeNetwork,
                                    prefix = output,
                                    suffix = graph_suffix,
                                    use_gpu = gpu_graph)
                    removeFromDB(output, output, names_to_remove)
                    db_suffix = file_extension_string + '.refs.h5'
                    os.rename(output + "/" + os.path.basename(output) + ".tmp.h5",
                            output + "/" + os.path.basename(output) + db_suffix)
        else:
            storePickle(rNames, qNames, False, qrDistMat, dists_out)
            if save_partial_query_graph:
                genomeNetwork, pruned_isolate_lists = remove_non_query_components(genomeNetwork, rNames, qNames, use_gpu = gpu_graph)
                if model.type == 'lineage' and not serial:
                    save_network(genomeNetwork[min(model.ranks)], prefix = output, suffix = '_graph', use_gpu = gpu_graph)
                else:
                    graph_suffix = file_extension_string + '_graph'
                    save_network(genomeNetwork, prefix = output, suffix = graph_suffix, use_gpu = gpu_graph)
                with open(f"{output}/{os.path.basename(output)}_query.subset",'w') as pruned_isolate_csv:
                    for isolate in pruned_isolate_lists:
                        pruned_isolate_csv.write(isolate + '\n')

    return(isolateClustering)


if __name__ == '__main__':
    main()

    sys.exit(0)
