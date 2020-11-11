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

# required from v2.1.1 onwards (no mash support)
import pp_sketchlib

# import poppunk package
from .__init__ import __version__



# Minimum sketchlib version
SKETCHLIB_MAJOR = 1
SKETCHLIB_MINOR = 5

#*******************************#
#*                             *#
#* query assignment            *#
#*                             *#
#*******************************#
def assign_query(dbFuncs,
                 ref_db,
                 q_files,
                 output,
                 update_db,
                 full_db,
                 distances,
                 threads,
                 overwrite,
                 plot_fit,
                 max_a_dist,
                 model_dir,
                 previous_clustering,
                 external_clustering,
                 core_only,
                 accessory_only,
                 assign_lineage,
                 rank,
                 lineage_accessory):
    """Code for assign query mode. Written as a separate function so it can be called
    by web APIs
    """
    # Modules imported here as graph tools very slow
    from .models import loadClusterFit, ClusterFit, BGMMFit, DBSCANFit, RefineFit, LineageFit

    from .sketchlib import removeFromDB

    from .network import fetchNetwork
    from .network import constructNetwork
    from .network import extractReferences
    from .network import addQueryToNetwork
    from .network import printClusters

    from .plot import writeClusterCsv

    from .prune_db import prune_distance_matrix

    from .sketchlib import calculateQueryQueryDistances

    from .utils import storePickle
    from .utils import readPickle
    from .utils import qcDistMat
    from .utils import update_distance_matrices

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

    # Find distances to reference db
    kmers, sketch_sizes, codon_phased = readDBParams(ref_db)

    # Sketch query sequences
    createDatabaseDir(output, kmers)

    # Find distances vs ref seqs
    rNames = []
    if os.path.isfile(ref_db + "/" + os.path.basename(ref_db) + ".refs"):
        with open(ref_db + "/" + os.path.basename(ref_db) + ".refs") as refFile:
            for reference in refFile:
                rNames.append(reference.rstrip())
    else:
        rNames = getSeqsInDb(ref_db + "/" + os.path.basename(ref_db) + ".h5")

    # construct database
    qNames = constructDatabase(q_files,
                                kmers,
                                sketch_sizes,
                                output,
                                threads,
                                overwrite,
                                codon_phased = codon_phased)

    # run query
    refList, queryList, distMat = queryDatabase(rNames = rNames,
                                                qNames = qNames,
                                                dbPrefix = ref_db,
                                                queryPrefix = output,
                                                klist = kmers,
                                                self = False,
                                                number_plot_fits = plot_fit,
                                                threads = threads)

    # QC distance matrix
    qcPass = qcDistMat(distMat, refList, queryList, max_a_dist)

    # Assign to strains or lineages, as requested.
    # Both need the previous model loaded
    model_prefix = ref_db
    if model_dir is not None:
        model_prefix = model_dir
    model_file = model_prefix + "/" + os.path.basename(model_prefix) + "_fit"

    model = loadClusterFit(model_file + '.pkl',
                            model_file + '.npz')

    # Set directories of previous fit
    if previous_clustering is not None:
        prev_clustering = previous_clustering
    else:
        prev_clustering = model_prefix

    # Load the network based on supplied options
    genomeNetwork, old_cluster_file = \
        fetchNetwork(prev_clustering, model, refList,
                        core_only, accessory_only)

    if assign_lineage:
        # Assign lineages by calculating query-query information
        qlist1, qlist2, query_distMat = queryDatabase(rNames = qNames,
                                                        qNames = qNames,
                                                        dbPrefix = output,
                                                        queryPrefix = output,
                                                        klist = kmers,
                                                        self = True,
                                                        number_plot_fits = 0,
                                                        threads = threads)
        model.extend(query_distMat, distMat)

        genomeNetwork = {}
        isolateClustering = defaultdict(dict)
        for rank in rank_list:
            assignment = model.assign(rank)
            # Overwrite the network loaded above
            genomeNetwork[rank] = constructNetwork(rNames + qNames,
                                                    rNames + qNames,
                                                    assignment,
                                                    0,
                                                    edge_list = True)

            isolateClustering[rank] = \
                printClusters(genomeNetwork[rank],
                                refList + queryList,
                                printCSV = False)

        overall_lineage = createOverallLineage(rank_list, isolateClustering)
        writeClusterCsv(output + "/" + \
            os.path.basename(output) + '_lineages.csv',
            refList + queryList,
            refList + queryList,
            overall_lineage,
            output_format = 'phandango',
            epiCsv = None,
            queryNames = refList,
            suffix = '_Lineage')

    else:
        # Assign these distances as within or between strain
        queryAssignments = model.assign(distMat)

        # Assign clustering by adding to network
        query_distMat = \
            addQueryToNetwork(dbFuncs, refList, queryList,
                                genomeNetwork, kmers,
                                queryAssignments, model, output, update_db,
                                threads)

        isolateClustering = \
            {'combined': printClusters(genomeNetwork, refList + queryList,
                                        output + "/" + os.path.basename(output),
                                        old_cluster_file,
                                        external_clustering,
                                        print_full_clustering)}

    # Update DB as requested
    if update_db:
        # Check new sequences pass QC before adding them
        if not qcPass:
            sys.stderr.write("Queries contained outlier distances, "
                            "not updating database\n")
        else:
            sys.stderr.write("Updating reference database to " + output + "\n")

        # Update the network + ref list (everything)
        joinDBs(ref_db, output, output)
        if assign_lineage:
            genomeNetwork[min(rank_list)].save(output + "/" + os.path.basename(output) + '_graph.gt', fmt = 'gt')
        else:
            genomeNetwork.save(output + "/" + os.path.basename(output) + '_graph.gt', fmt = 'gt')

        # Update distance matrices with all calculated distances
        dists_out = output + "/" + os.path.basename(output) + ".dists"
        if distances == None:
            distanceFiles = ref_db + "/" + os.path.basename(ref_db) + ".dists"
        else:
            distanceFiles = distances
        refList, refList_copy, self, ref_distMat = readPickle(distanceFiles)
        combined_seq, core_distMat, acc_distMat = \
            update_distance_matrices(refList, ref_distMat,
                                    queryList, distMat,
                                    query_distMat, threads = threads)
        complete_distMat = \
            np.hstack((pp_sketchlib.squareToLong(core_distMat, threads).reshape(-1, 1),
                        pp_sketchlib.squareToLong(acc_distMat, threads).reshape(-1, 1)))

        if not full_db and not assign_lineage:
            dbOrder = refList + queryList
            newRepresentativesIndices, newRepresentativesNames, \
                newRepresentativesFile, genomeNetwork = \
                    extractReferences(genomeNetwork, dbOrder, output, refList, threads = threads)
            # intersection that maintains order
            newQueries = [x for x in queryList if x in frozenset(newRepresentativesNames)]
            genomeNetwork.save(output + "/" + os.path.basename(output) + '.refs_graph.gt', fmt = 'gt')

            # could also have newRepresentativesNames in this diff (should be the same) - but want
            # to ensure consistency with the network in case of bad input/bugs
            nodes_to_remove = set(combined_seq).difference(newRepresentativesNames)
            # This function also writes out the new distance matrix
            postpruning_combined_seq, newDistMat = \
                prune_distance_matrix(combined_seq, nodes_to_remove,
                                        complete_distMat, dists_out)
            # Create and save a prune ref db
            if len(nodes_to_remove) > 0:
                removeFromDB(output, output, nodes_to_remove)
                os.rename(output + "/" + os.path.basename(output) + ".tmp.h5",
                            output + "/" + os.path.basename(output) + ".refs.h5")

            # ensure sketch and distMat order match
            assert postpruning_combined_seq == refList + newQueries
        else:
            storePickle(combined_seq, combined_seq, True,
                        complete_distMat, dists_out)
            # ensure sketch and distMat order match
            assert combined_seq == refList + queryList

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
    iGroup.add_argument('--ref-db', required=True, type = str, help='Location of built reference database')
    iGroup.add_argument('--q-files', required=True, help='File listing query input assemblies')
    iGroup.add_argument('--distances', required=True, help='Prefix of input pickle of pre-calculated distances')
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
    kmerGroup.add_argument('--min-kmer-count', default=0, type=int, help='Minimum k-mer count when using reads as input [default = 0]')
    kmerGroup.add_argument('--exact-count', default=False, action='store_true',
                           help='Use the exact k-mer counter with reads '
                                '[default = use countmin counter]')
    kmerGroup.add_argument('--strand-preserved', default=False, action='store_true',
                           help='Treat input as being on the same strand, and ignore reverse complement '
                                'k-mers [default = use canonical k-mers]')

    # qc options
    qcGroup = parser.add_argument_group('Quality control options')
    qcGroup.add_argument('--max-a-dist', help='Maximum accessory distance to permit [default = 0.5]',
                                                default = 0.5, type = float)

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

    # lineage clustering within strains
    lineagesGroup = parser.add_argument_group('Lineage analysis options')
    lineagesGroup.add_argument('--assign-lineages',
                                help='Assign isolates to a lineages scheme fitted with --lineage-clustering',
                                default=False,
                                action='store_true')
    lineagesGroup.add_argument('--rank',
                                help='Rank to assign to',
                                type = int,
                                default = 1)
    lineagesGroup.add_argument('--use-accessory',
                                help='Use accessory distances for lineage definitions [default = use core distances]',
                                action = 'store_true',
                                default = False)

    # processing
    other = parser.add_argument_group('Other options')
    other.add_argument('--threads', default=1, type=int, help='Number of threads to use [default = 1]')
    other.add_argument('--gpu-sketch', default=False, action='store_true', help='Use a GPU when calculating sketches (read data only) [default = False]')
    other.add_argument('--gpu-dist', default=False, action='store_true', help='Use a GPU when calculating distances [default = False]')
    other.add_argument('--deviceid', default=0, type=int, help='CUDA device ID, if using GPU [default = 0]')

    other.add_argument('--version', action='version',
                       version='%(prog)s '+__version__)


    # combine
    args = parser.parse_args()

    # ensure directories do not have trailing forward slash
    for arg in [args.ref_db, args.model_dir, args.output, args.previous_clustering]:
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

    from .sketchlib import checkSketchlibLibrary
    from .utils import setGtThreads
    from .utils import setupDBFuncs

    # Dict of QC options for passing to database construction and querying functions
    qc_dict = {'run_qc': False }

    # Dict of DB access functions for assign_query (which is out of scope)
    dbFuncs = setupDBFuncs(args, args.min_kmer_count, qc_dict)

    # run according to mode
    sys.stderr.write("PopPUNK: assign (POPulation Partitioning Using Nucleotide Kmers)\n")
    sys.stderr.write("\t(with backend: " + dbFuncs['backend'] + " v" + dbFuncs['backend_version'] + "\n")
    if (dbFuncs['backend'] == 'sketchlib'):
        sketchlib_version = [int(x) for x in dbFuncs['backend_version'].split(".")]
        if sketchlib_version[0] < SKETCHLIB_MAJOR or sketchlib_version[1] < SKETCHLIB_MINOR:
            sys.stderr.write("This version of PopPUNK requires sketchlib "
                             "v" + str(SKETCHLIB_MAJOR) + "." + str(SKETCHLIB_MINOR) + ".0 or higher\n")
            sys.exit(1)
        else:
            sys.stderr.write('\t sketchlib: ' + checkSketchlibLibrary() + ')\n')

    # Check on parallelisation of graph-tools
    setGtThreads(args.threads)

    #*******************************#
    #*                             *#
    #* query assignment (function  *#
    #* at top)                     *#
    #*                             *#
    #*******************************#
    assign_query(dbFuncs,
                 args.ref_db,
                 args.q_files,
                 args.output,
                 args.update_db,
                 args.full_db,
                 args.distances,
                 args.threads,
                 args.overwrite,
                 args.plot_fit,
                 args.max_a_dist,
                 args.model_dir,
                 args.previous_clustering,
                 args.external_clustering,
                 args.core_only,
                 args.accessory_only,
                 args.assign_lineages,
                 args.rank,
                 args.use_accessory)

    sys.stderr.write("\nDone\n")


if __name__ == '__main__':
    main()

    sys.exit(0)
