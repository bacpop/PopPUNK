#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018-2020 John Lees and Nick Croucher

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
    oGroup.add_argument('--update-db', help='Update reference database with query sequences', default=False, action='store_true')
    oGroup.add_argument('--overwrite', help='Overwrite any existing database files', default=False, action='store_true')
    oGroup.add_argument('--graph-weights', help='Save within-strain Euclidean distances into the graph', default=False, action='store_true')

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
                 save_partial_query_graph=False)

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
                 save_partial_query_graph):
    """Code for assign query mode for CLI"""
    from .qc import sketchlibAssemblyQC

    createDatabaseDir = dbFuncs['createDatabaseDir']
    constructDatabase = dbFuncs['constructDatabase']
    readDBParams = dbFuncs['readDBParams']

    if ref_db == output:
        sys.stderr.write("--output and --ref-db must be different to "
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
                sys.exit(1)

    isolateClustering = assign_query_hdf5(dbFuncs,
                    ref_db,
                    qNames,
                    output,
                    qc_dict,
                    update_db,
                    write_references,
                    distances,
                    serial,
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
                    save_partial_query_graph)
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
                 save_partial_query_graph):
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


    from .plot import writeClusterCsv

    from .qc import qcDistMat, qcQueryAssignments, prune_distance_matrix, \
        prune_query_distance_matrix

    from .sketchlib import addRandom

    from .utils import storePickle
    from .utils import readPickle
    from .utils import update_distance_matrices
    from .utils import createOverallLineage

    joinDBs = dbFuncs['joinDBs']
    queryDatabase = dbFuncs['queryDatabase']
    readDBParams = dbFuncs['readDBParams']
    getSeqsInDb = dbFuncs['getSeqsInDb']

    if ref_db == output:
        sys.stderr.write("--output and --ref-db must be different to "
                         "prevent overwrite.\n")
        sys.exit(1)
    if (update_db and not distances):
        sys.stderr.write("--update-db requires --distances to be provided\n")
        sys.exit(1)
    if serial and update_db:
        raise RuntimeError("--update-db cannot be used with --serial")

    # Load the previous model
    model_prefix = ref_db
    if model_dir is not None:
        model_prefix = model_dir
    model_file = model_prefix + "/" + os.path.basename(model_prefix) + "_fit"

    model = loadClusterFit(model_file + '.pkl',
                           model_file + '.npz')
    if model.type == "lineage" and serial:
        raise RuntimeError("lineage models cannot be used with --serial")
    model.set_threads(threads)

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
        # Find distances vs ref seqs
        rNames = []
        ref_file_name = os.path.join(model_prefix,
                        os.path.basename(model_prefix) + file_extension_string + ".refs")
        use_ref_graph = \
            os.path.isfile(ref_file_name) and not update_db and model.type != 'lineage'
        if use_ref_graph:
            with open(ref_file_name) as refFile:
                for reference in refFile:
                    rNames.append(reference.rstrip())
        else:
            if os.path.isfile(distances + ".pkl"):
                rNames = readPickle(distances, enforce_self = True, distances=False)[0]
            elif update_db:
                sys.stderr.write("Reference distances missing, cannot use --update-db\n")
                sys.exit(1)
            else:
                rNames = getSeqsInDb(os.path.join(ref_db, os.path.basename(ref_db) + ".h5"))

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
            seq_names_passing = \
                frozenset(qcDistMat(qrDistMat, rNames, qNames, ref_db, qc_dict)[0])
            failed_samples = frozenset(qNames) - seq_names_passing
            if len(failed_samples) > 0:
                sys.stderr.write(f"{len(failed_samples)} samples failed:\n"
                                 f"{','.join(failed_samples)}\n")
                if len(failed_samples) == len(qNames):
                    sys.exit(1)
                else:
                    qNames, qrDistMat = \
                        prune_query_distance_matrix(rNames, qNames, failed_samples, qrDistMat)[0:2]
                    if update_db:
                        sys.stderr.write("Queries contained outlier distances, "
                                         "not updating database\n")
                        update_db = False

        # Load the network based on supplied options
        genomeNetwork, old_cluster_file = \
            fetchNetwork(prev_clustering,
                         model,
                         rNames,
                         ref_graph = use_ref_graph,
                         core_only = (fit_type == 'core_refined'),
                         accessory_only = (fit_type == 'accessory_refined'),
                         use_gpu = gpu_graph)

        if max(get_vertex_list(genomeNetwork, use_gpu = gpu_graph)) != (len(rNames) - 1):
            sys.stderr.write("There are " + str(max(get_vertex_list(genomeNetwork, use_gpu = gpu_graph)) + 1) + \
                             " vertices in the network but " + str(len(rNames)) + " reference names supplied; " + \
                             "please check the '--model-dir' variable is pointing to the correct directory\n")

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
                                  use_gpu = gpu_graph)

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
            # Assign these distances as within or between strain
            if fit_type == 'core_refined' or model.threshold:
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
                        if update_db:
                            sys.stderr.write("Queries contained too many links, "
                                             "not updating database\n")
                            update_db = False

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
                                        queryQuery = (update_db and
                                                        (fit_type == 'default' or
                                                        (fit_type != 'default' and use_ref_graph)
                                                        )
                                                    ),
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

                isolateClustering = \
                    {'combined': printClusters(genomeNetwork,
                                                rNames + qNames,
                                                output_fn,
                                                old_cluster_file,
                                                external_clustering,
                                                write_references or update_db,
                                                use_gpu = gpu_graph)}
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
                    isolate_cluster = printClusters(genomeNetwork,
                                                rNames + [sample],
                                                output_fn,
                                                old_cluster_file,
                                                external_clustering,
                                                printRef=False,
                                                printCSV=False,
                                                write_unwords=False,
                                                use_gpu = gpu_graph)
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
            if model.type == 'lineage':
                save_network(genomeNetwork[min(model.ranks)],
                                prefix = output,
                                suffix = '_graph',
                                use_gpu = gpu_graph)
                # Save sparse distance matrices and updated model
                model.outPrefix = os.path.basename(output)
                model.save()
            else:
                graph_suffix = file_extension_string + '_graph'
                save_network(genomeNetwork,
                                prefix = output,
                                suffix = graph_suffix,
                                use_gpu = gpu_graph)
            # Load the previous distances
            refList_loaded, refList_copy, self, rrDistMat = \
                readPickle(distances,
                           enforce_self = True)
            # This should now always be true, otherwise both qrDistMat and sparse matrix
            # may need reordering
            assert(refList_loaded == rNames)
            combined_seq, core_distMat, acc_distMat = \
                update_distance_matrices(rNames, rrDistMat,
                                         qNames, qrDistMat,
                                         qqDistMat, threads = threads)
            assert combined_seq == rNames + qNames

            # Get full distance matrix and save
            complete_distMat = \
                np.hstack((pp_sketchlib.squareToLong(distMat=core_distMat, num_threads=threads).reshape(-1, 1),
                           pp_sketchlib.squareToLong(distMat=acc_distMat, num_threads=threads).reshape(-1, 1)))
            storePickle(combined_seq, combined_seq, True, complete_distMat, dists_out)

            # Copy model if needed
            if output != model.outPrefix and fit_type == 'default':
                model.copy(output)

            # Clique pruning
            if model.type != 'lineage':

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
                                            outSuffix = file_extension_string,
                                            existingRefs = existing_ref_list,
                                            type_isolate = qc_dict['type_isolate'],
                                            threads = threads,
                                            use_gpu = gpu_graph)

                # intersection that maintains order
                newQueries = [x for x in qNames if x in frozenset(newRepresentativesNames)]

                # could also have newRepresentativesNames in this diff (should be the same) - but want
                # to ensure consistency with the network in case of bad input/bugs
                nodes_to_remove = set(range(len(combined_seq))).difference(newRepresentativesIndices)
                names_to_remove = [combined_seq[n] for n in nodes_to_remove]

                if (len(names_to_remove) > 0):
                    # This function also writes out the new ref distance matrix
                    dists_suffix = file_extension_string + '.refs.dists'
                    postpruning_combined_seq, newDistMat = \
                        prune_distance_matrix(combined_seq, names_to_remove, complete_distMat,
                                              output + "/" + os.path.basename(output) + dists_suffix)
                    graph_suffix = file_extension_string + '_refs_graph'
                    save_network(genomeNetwork,
                                    prefix = output,
                                    suffix = graph_suffix,
                                    use_gpu = gpu_graph)
                    removeFromDB(output, output, names_to_remove)
                    db_suffix = file_extension_string + '.refs.h5'
                    os.rename(output + "/" + os.path.basename(output) + ".tmp.h5",
                              output + "/" + os.path.basename(output) + db_suffix)

                    # Check that the updated set of references includes all old references, and references added from
                    # queries; there may be further new references, even from the original database, where paths are
                    # added between reference isolates in the same component, or new cliques formed
                    added_references = set(existing_ref_list).union(set(newQueries))
                    assert set(postpruning_combined_seq).issuperset(added_references), "Error identifying references"
        else:
            storePickle(rNames, qNames, False, qrDistMat, dists_out)
            if save_partial_query_graph and not serial:
                if model.type == 'lineage':
                    save_network(genomeNetwork[min(model.ranks)], prefix = output, suffix = '_graph', use_gpu = gpu_graph)
                else:
                    graph_suffix = file_extension_string + '_graph'
                    save_network(genomeNetwork, prefix = output, suffix = graph_suffix, use_gpu = gpu_graph)

    return(isolateClustering)


if __name__ == '__main__':
    main()

    sys.exit(0)
