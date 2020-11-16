#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018-2020 John Lees and Nick Croucher

# universal
import os
import sys
# additional
import numpy as np

# required from v2.1.1 onwards (no mash support)
import pp_sketchlib

# import poppunk package
from .__init__ import __version__

from .models import loadClusterFit

from .network import fetchNetwork

from .plot import outputsForMicroreact
from .plot import outputsForCytoscape
from .plot import outputsForPhandango
from .plot import outputsForGrapetree
from .plot import writeClusterCsv

from .prune_db import prune_distance_matrix

from .utils import readPickle
from .utils import setGtThreads
from .utils import update_distance_matrices
from .utils import readIsolateTypeFromCsv

#******************************#
#*                            *#
#* Command line parsing       *#
#*                            *#
#******************************#
def get_options():

    import argparse

    parser = argparse.ArgumentParser(description='Create visualisations from PopPUNK results',
                                     prog='poppunk_visualise')

    # input options
    iGroup = parser.add_argument_group('Input files')
    iGroup.add_argument('--subset',
                         help='File with list of sequences to include in visualisation',
                         default=None)
    iGroup.add_argument('--ref-db',
                        type = str,
                        help='Location of built reference database',
                        required=True)
    iGroup.add_argument('--distances',
                        help='Prefix of input pickle of pre-calculated distances',
                        required=True)
    iGroup.add_argument('--external-clustering',
                        help='File with cluster definitions or other labels '
                             'generated with any other method.',
                        default=None)
    iGroup.add_argument('--model-dir',
                        help='Directory containing model to use for assigning queries '
                             'to clusters [default = reference database directory]',
                        type = str)
    iGroup.add_argument('--previous-clustering',
                        help='Directory containing previous cluster definitions '
                             'and network [default = use that in the directory '
                             'containing the model]',
                        type = str)

    # output options
    oGroup = parser.add_argument_group('Output options')
    oGroup.add_argument('--output',
                        required=True,
                        help='Prefix for output files (required)')
    oGroup.add_argument('--overwrite',
                        help='Overwrite any existing visualisation files',
                        default=False,
                        action='store_true')

    # plot output
    faGroup = parser.add_argument_group('Visualisation options')
    faGroup.add_argument('--microreact', help='Generate output files for microreact visualisation', default=False, action='store_true')
    faGroup.add_argument('--cytoscape', help='Generate network output files for Cytoscape', default=False, action='store_true')
    faGroup.add_argument('--phandango', help='Generate phylogeny and TSV for Phandango visualisation', default=False, action='store_true')
    faGroup.add_argument('--grapetree', help='Generate phylogeny and CSV for grapetree visualisation', default=False, action='store_true')
    faGroup.add_argument('--rapidnj', help='Path to rapidNJ binary to build NJ tree for Microreact', default=None)
    faGroup.add_argument('--perplexity',
                         type=float, default = 20.0,
                         help='Perplexity used to calculate t-SNE projection (with --microreact) [default=20.0]')
    faGroup.add_argument('--info-csv',
                         help='Epidemiological information CSV formatted for microreact (can be used with other outputs)')

    # processing
    other = parser.add_argument_group('Other options')
    other.add_argument('--threads', default=1, type=int, help='Number of threads to use [default = 1]')

    other.add_argument('--version', action='version',
                       version='%(prog)s '+__version__)


    # combine
    args = parser.parse_args()

    # ensure directories do not have trailing forward slash
    for arg in [args.ref_db, args.model_dir, args.output, args.external_clustering, args.previous_clustering]:
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

    # Check on parallelisation of graph-tools
    setGtThreads(args.threads)

    sys.stderr.write("PopPUNK: visualise\n")
    if not (args.microreact or args.phandango or args.grapetree or args.cytoscape):
        sys.stderr.write("Must specify at least one type of visualisation to output\n")
        sys.exit(1)

    # Initial processing
    # Load original distances
    rlist, qlist, self, complete_distMat = readPickle(args.distances)
    # TODO deal with self = False

    # make directory for new output files
    if not os.path.isdir(args.output):
        try:
            os.makedirs(args.output)
        except OSError:
            sys.stderr.write("Cannot create output directory\n")
            sys.exit(1)

    # Define set/subset to be visualised
    # extract subset of distances if requested
    if args.subset is not None:
        viz_subset = []
        with open(args.subset, 'r') as assemblyFiles:
            for assembly in assemblyFiles:
                viz_subset.append(assembly.rstrip())
        if len(set(viz_subset).difference(rlist)) > 0:
            sys.stderr.write("--subset contains names not in --distances")

        isolates_to_remove = set(rlist).difference(viz_subset)
        if len(isolates_to_remove) > 0:
            dists_out = args.output + "/" + os.path.basename(args.output) + ".dists"
            postpruning_combined_seq, newDistMat = \
                prune_distance_matrix(rlist, isolates_to_remove,
                                      complete_distMat, dists_out)
            combined_seq, core_distMat, acc_distMat = \
                update_distance_matrices(viz_subset, newDistMat,
                                         threads = args.threads)
    else:
        combined_seq, core_distMat, acc_distMat = \
            update_distance_matrices(rlist, complete_distMat,
                                     threads = args.threads)

    # Either use strain definitions, lineage assignments or external clustering
    isolateClustering = {}
    # Use external clustering if specified
    if args.external_clustering:
        cluster_file = args.external_clustering
        isolateClustering = readIsolateTypeFromCsv(cluster_file,
                                                   mode = 'external',
                                                   return_dict = True)

    # identify existing analysis files
    if args.model_dir is not None:
        model_prefix = args.model_dir
    else:
        model_prefix = args.ref_db
    model_file = model_prefix + "/" + os.path.basename(model_prefix)

    try:
        model = loadClusterFit(model_file + '_fit.pkl',
                               model_file + '_fit.npz')
    except:
        sys.stderr.write('Unable to locate previous model fit in ' + model_prefix + '\n')
        sys.exit(1)

    # Set directories of previous fit
    if args.previous_clustering is not None:
        prev_clustering = args.previous_clustering
    else:
        prev_clustering = os.path.dirname(args.distances + ".pkl")

    # load clustering
    if model.type == "lineage":
        cluster_file = prev_clustering + '/' + os.path.basename(prev_clustering) + '_lineages.csv'
        isolateClustering = readIsolateTypeFromCsv(cluster_file,
                                                   mode = 'lineages',
                                                   return_dict = True)
    elif model.indiv_fitted:
        cluster_file = args.ref_db + '/' + os.path.basename(args.ref_db) + '_clusters.csv'
        isolateClustering['refine'] = readIsolateTypeFromCsv(cluster_file, mode = 'clusters', return_dict = True)
        isolateClustering['refine'] = isolateClustering['refine']['Cluster']
        for type in ['accessory','core']:
            cluster_file = args.ref_db + '/' + os.path.basename(args.ref_db) + '_' + type + '_clusters.csv'
            isolateClustering[type] = readIsolateTypeFromCsv(cluster_file, mode = 'clusters', return_dict = True)
            isolateClustering[type] = isolateClustering[type]['Cluster']
    else:
        cluster_file = args.ref_db + '/' + os.path.basename(args.ref_db) + '_clusters.csv'
        isolateClustering = readIsolateTypeFromCsv(cluster_file, mode = 'clusters', return_dict = True)

    # generate selected visualisations
    if args.microreact:
        sys.stderr.write("Writing microreact output\n")
        outputsForMicroreact(combined_seq, core_distMat, acc_distMat, isolateClustering, args.perplexity,
                             args.output, args.info_csv, args.rapidnj, overwrite = args.overwrite)
    if args.phandango:
        sys.stderr.write("Writing phandango output\n")
        outputsForPhandango(combined_seq, core_distMat, isolateClustering, args.output, args.info_csv, args.rapidnj,
                            overwrite = args.overwrite, microreact = args.microreact)
    if args.grapetree:
        sys.stderr.write("Writing grapetree output\n")
        outputsForGrapetree(combined_seq, core_distMat, isolateClustering, args.output, args.info_csv, args.rapidnj,
                            overwrite = args.overwrite, microreact = args.microreact)
    if args.cytoscape:
        sys.stderr.write("Writing cytoscape output\n")
        genomeNetwork, cluster_file = fetchNetwork(prev_clustering, model, rlist, False, args.core_only, args.accessory_only)
        outputsForCytoscape(genomeNetwork, isolateClustering, args.output, args.info_csv, viz_subset = viz_subset)
        if model.indiv_fitted:
            sys.stderr.write("Writing individual cytoscape networks\n")
            import graph_tool.all as gt
            for dist_type in ['core', 'accessory']:
                indiv_network = gt.load_graph(args.ref_db + "/" + os.path.basename(args.ref_db) +
                "_" + dist_type + '_graph.gt')
                outputsForCytoscape(indiv_network, isolateClustering, args.output,
                            args.info_csv, suffix = dist_type, viz_subset = viz_subset)


if __name__ == '__main__':
    main()

    sys.exit(0)
