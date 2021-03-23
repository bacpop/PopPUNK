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
import scipy.optimize
from scipy.sparse import coo_matrix, bmat, find

# required from v2.1.1 onwards (no mash support)
import pp_sketchlib

# import poppunk package
from .__init__ import __version__
from .models import rankFile

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
                 threads,
                 overwrite,
                 plot_fit,
                 graph_weights,
                 max_a_dist,
                 max_pi_dist,
                 type_isolate,
                 model_dir,
                 strand_preserved,
                 previous_clustering,
                 external_clustering,
                 core_only,
                 accessory_only,
                 gpu_sketch,
                 gpu_dist,
                 gpu_graph,
                 deviceid,
                 web,
                 json_sketch,
                 save_partial_query_graph):
    """Code for assign query mode. Written as a separate function so it can be called
    by web APIs"""

    # Modules imported here as graph tool is very slow to load (it pulls in all of GTK?)
    from .models import loadClusterFit, ClusterFit, BGMMFit, DBSCANFit, RefineFit, LineageFit

    from .sketchlib import removeFromDB

    from .network import fetchNetwork
    from .network import constructNetwork
    from .network import extractReferences
    from .network import addQueryToNetwork
    from .network import printClusters
    from .network import save_network

    from .plot import writeClusterCsv

    from .prune_db import prune_distance_matrix

    from .sketchlib import addRandom

    from .utils import storePickle
    from .utils import readPickle
    from .utils import qcDistMat
    from .utils import update_distance_matrices
    from .utils import createOverallLineage

    from .web import sketch_to_hdf5

    createDatabaseDir = dbFuncs['createDatabaseDir']
    constructDatabase = dbFuncs['constructDatabase']
    joinDBs = dbFuncs['joinDBs']
    queryDatabase = dbFuncs['queryDatabase']
    readDBParams = dbFuncs['readDBParams']
    getSeqsInDb = dbFuncs['getSeqsInDb']

    sys.stderr.write("Mode: Assigning clusters of query sequences\n\n")
    if ref_db == output:
        sys.stderr.write("--output and --ref-db must be different to "
                         "prevent overwrite.\n")
        sys.exit(1)
    if (update_db and not distances):
        sys.stderr.write("--update-db requires --distances to be provided\n")
        sys.exit(1)

    # Load the previous model
    model_prefix = ref_db
    if model_dir is not None:
        model_prefix = model_dir
    model_file = model_prefix + "/" + os.path.basename(model_prefix) + "_fit"

    model = loadClusterFit(model_file + '.pkl',
                           model_file + '.npz')
    model.set_threads(threads)

    # Set directories of previous fit
    if previous_clustering is not None:
        prev_clustering = previous_clustering
    else:
        prev_clustering = model_prefix

    # Find distances to reference db
    kmers, sketch_sizes, codon_phased = readDBParams(ref_db)

    # Find distances vs ref seqs
    rNames = []
    use_ref_graph = \
        os.path.isfile(ref_db + "/" + os.path.basename(ref_db) + ".refs") \
        and not update_db and model.type != 'lineage'
    if use_ref_graph:
        with open(ref_db + "/" + os.path.basename(ref_db) + ".refs") as refFile:
            for reference in refFile:
                rNames.append(reference.rstrip())
    else:
        if os.path.isfile(distances + ".pkl"):
            rNames = readPickle(distances, enforce_self = True, distances=False)[0]
        elif update_db:
            sys.stderr.write("Reference distances missing, cannot use --update-db\n")
            sys.exit(1)
        else:
            rNames = getSeqsInDb(ref_db + "/" + os.path.basename(ref_db) + ".h5")
    # construct database
    if (web and json_sketch):
        qNames = sketch_to_hdf5(json_sketch, output)
    else:
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
        seq_names_passing = qcDistMat(qrDistMat, rNames, qNames, ref_db, output, qc_dict)[0]
    else:
        seq_names_passing = rNames + qNames

    # Load the network based on supplied options
    genomeNetwork, old_cluster_file = \
        fetchNetwork(prev_clustering,
                     model,
                     rNames,
                     ref_graph = use_ref_graph,
                     core_only = core_only,
                     accessory_only = accessory_only,
                     use_gpu = gpu_graph)

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
            genomeNetwork[rank] = constructNetwork(rNames + qNames,
                                                   rNames + qNames,
                                                   assignment,
                                                   0,
                                                   edge_list = True,
                                                   weights=weights,
                                                   use_gpu = gpu_graph)

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
        queryAssignments = model.assign(qrDistMat)

        # Assign clustering by adding to network
        if graph_weights:
            weights = qrDistMat
        else:
            weights = None

        genomeNetwork, qqDistMat = \
            addQueryToNetwork(dbFuncs, rNames, qNames,
                                genomeNetwork, kmers,
                                queryAssignments, model, output, update_db,
                                strand_preserved,
                                weights = weights, threads = threads, use_gpu = gpu_graph)

        isolateClustering = \
            {'combined': printClusters(genomeNetwork, rNames + qNames,
                                        output + "/" + os.path.basename(output),
                                        old_cluster_file,
                                        external_clustering,
                                        write_references or update_db,
                                        use_gpu = gpu_graph)}

    # Update DB as requested
    dists_out = output + "/" + os.path.basename(output) + ".dists"
    if update_db:
        # Check new sequences pass QC before adding them
        if len(set(seq_names_passing).difference(rNames + qNames)) > 0:
            sys.stderr.write("Queries contained outlier distances, "
                             "not updating database\n")
        else:
            sys.stderr.write("Updating reference database to " + output + "\n")

        # Update the network + ref list (everything)
        joinDBs(ref_db, output, output,
                {"threads": threads, "strand_preserved": strand_preserved})
        if model.type == 'lineage':
            save_network(genomeNetwork[min(model.ranks)], prefix = output, suffix = '_graph', use_gpu = gpu_graph)
            # Save sparse distance matrices and updated model
            model.outPrefix = os.path.basename(output)
            model.save()
        else:
            save_network(genomeNetwork, prefix = output, suffix = '_graph', use_gpu = gpu_graph)

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
            np.hstack((pp_sketchlib.squareToLong(core_distMat, threads).reshape(-1, 1),
                       pp_sketchlib.squareToLong(acc_distMat, threads).reshape(-1, 1)))
        storePickle(combined_seq, combined_seq, True, complete_distMat, dists_out)

        # Copy model if needed
        if output != model.outPrefix:
            model.copy(output)

        # Clique pruning
        if model.type != 'lineage':
            dbOrder = rNames + qNames
            newRepresentativesIndices, newRepresentativesNames, \
                newRepresentativesFile, genomeNetwork = \
                    extractReferences(genomeNetwork,
                                        dbOrder,
                                        output,
                                        existingRefs = rNames,
                                        type_isolate = qc_dict['type_isolate'],
                                        threads = threads,
                                        use_gpu = gpu_graph)
            # intersection that maintains order
            newQueries = [x for x in qNames if x in frozenset(newRepresentativesNames)]

            # could also have newRepresentativesNames in this diff (should be the same) - but want
            # to ensure consistency with the network in case of bad input/bugs
            nodes_to_remove = set(range(len(dbOrder))).difference(newRepresentativesIndices)
            names_to_remove = [dbOrder[n] for n in nodes_to_remove]

            if (len(names_to_remove) > 0):
                # This function also writes out the new ref distance matrix
                postpruning_combined_seq, newDistMat = \
                    prune_distance_matrix(combined_seq, names_to_remove, complete_distMat,
                                          output + "/" + os.path.basename(output) + ".refs.dists")
                save_network(genomeNetwork, prefix = output, suffix = 'refs_graph', use_gpu = gpu_graph)
                removeFromDB(output, output, names_to_remove)
                os.rename(output + "/" + os.path.basename(output) + ".tmp.h5",
                          output + "/" + os.path.basename(output) + ".refs.h5")

                # ensure sketch and distMat order match
                assert postpruning_combined_seq == rNames + newQueries
    else:
        storePickle(rNames, qNames, False, qrDistMat, dists_out)
        if save_partial_query_graph:
            if model.type == 'lineage':
                save_network(genomeNetwork[min(model.ranks)], prefix = output, suffix = '_graph', use_gpu = gpu_graph)
            else:
                save_network(genomeNetwork, prefix = output, suffix = '_graph', use_gpu = gpu_graph)

    return(isolateClustering)

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
    qcGroup.add_argument('--qc-filter', help='Behaviour following sequence QC step: "stop" [default], "prune"'
                                                ' (analyse data passing QC), or "continue" (analyse all data)',
                                                default='stop', type = str, choices=['stop', 'prune', 'continue'])
    qcGroup.add_argument('--retain-failures', help='Retain sketches of genomes that do not pass QC filters in '
                                                'separate database [default = False]', default=False, action='store_true')
    qcGroup.add_argument('--max-a-dist', help='Maximum accessory distance to permit [default = 0.5]',
                                                default = 0.5, type = float)
    qcGroup.add_argument('--max-pi-dist', help='Maximum core distance to permit [default = 0.5]',
                                                default = 0.5, type = float)
    qcGroup.add_argument('--type-isolate', help='Isolate from which distances can be calculated for pruning [default = None]',
                                                default = None, type = str)
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

    # sequence querying
    queryingGroup = parser.add_argument_group('Database querying options')
    queryingGroup.add_argument('--model-dir', help='Directory containing model to use for assigning queries '
                                                   'to clusters [default = reference database directory]', type = str)
    queryingGroup.add_argument('--previous-clustering', help='Directory containing previous cluster definitions '
                                                             'and network [default = use that in the directory '
                                                             'containing the model]', type = str)
    queryingGroup.add_argument('--core-only', help='(with a \'refine\' model) '
                                                   'Use a core-distance only model for assigning queries '
                                                   '[default = False]', default=False, action='store_true')
    queryingGroup.add_argument('--accessory-only', help='(with a \'refine\' or \'lineage\' model) '
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
    if args.length_sigma is None and None in args.length_range and args.prop_n is None \
        and args.upper_n is None and args.max_a_dist is None and args.max_pi_dist is None:
        qc_dict = {'run_qc': False, 'type_isolate': None }
    else:
        # define defaults if one QC parameter given
        # length_sigma
        if args.length_sigma is not None:
            length_sigma = args.length_sigma
        elif None in args.length_range:
            length_sigma = 5 # default used in __main__
        else:
            length_sigma = None
        # prop_n
        if args.prop_n is not None:
            prop_n = args.prop_n
        elif args.upper_n is None:
            prop_n = 0.1 # default used in __main__
        else:
            prop_n = None
        qc_dict = {
            'run_qc': True,
            'qc_filter': args.qc_filter,
            'retain_failures': args.retain_failures,
            'length_sigma': length_sigma,
            'length_range': args.length_range,
            'prop_n': prop_n,
            'upper_n': args.upper_n,
            'max_pi_dist': args.max_pi_dist,
            'max_a_dist': args.max_a_dist,
            'type_isolate': args.type_isolate
        }

    # Dict of DB access functions for assign_query (which is out of scope)
    dbFuncs = setupDBFuncs(args, args.min_kmer_count, qc_dict)

    # run according to mode
    sys.stderr.write("PopPUNK: assign\n")
    sys.stderr.write("\t(with backend: " + dbFuncs['backend'] + " v" + dbFuncs['backend_version'] + "\n")
    sys.stderr.write('\t sketchlib: ' + checkSketchlibLibrary() + ')\n')

    # Check on parallelisation of graph-tools
    setGtThreads(args.threads)

    if args.distances is None:
        distances = args.db + "/" + os.path.basename(args.db) + ".dists"
    else:
        distances = args.distances

    #*******************************#
    #*                             *#
    #* query assignment (function  *#
    #* at top)                     *#
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
                 args.threads,
                 args.overwrite,
                 args.plot_fit,
                 args.graph_weights,
                 args.max_a_dist,
                 args.max_pi_dist,
                 args.type_isolate,
                 args.model_dir,
                 args.strand_preserved,
                 args.previous_clustering,
                 args.external_clustering,
                 args.core_only,
                 args.accessory_only,
                 args.gpu_sketch,
                 args.gpu_dist,
                 args.gpu_graph,
                 args.deviceid,
                 web=False,
                 json_sketch=None,
                 save_partial_query_graph=False)

    sys.stderr.write("\nDone\n")


if __name__ == '__main__':
    main()

    sys.exit(0)
