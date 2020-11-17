#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018-2020 John Lees and Nick Croucher

# universal
import os
import sys
# additional
from shutil import copyfile

# import poppunk package
from .__init__ import __version__

from .sketchlib import removeFromDB

from .network import extractReferences

from .prune_db import prune_distance_matrix

from .utils import setGtThreads
from .utils import readPickle

# command line parsing
def get_options():

    import argparse

    parser = argparse.ArgumentParser(description='Select references from a PopPUNK database',
                                     prog='poppunk_references')

    # input options
    iGroup = parser.add_argument_group('Input files')
    iGroup.add_argument('--network', required=True, help='gpickle file of network (required)')
    iGroup.add_argument('--distances', required=True, help='Prefix of input pickle of pre-calculated distances (required)')
    iGroup.add_argument('--ref-db', help='Location of reference db, if resketching')
    iGroup.add_argument('--model', help='Directory containing model fit. Will be copied to new directory')
    iGroup.add_argument('--clusters', default=None, help='Specify a different clustering (e.g. core/accessory) to '
                                                         'copy to new directory')

    # output options
    oGroup = parser.add_argument_group('Output options')
    oGroup.add_argument('--output', required=True, help='Prefix for output files (required)')

    # processing
    other = parser.add_argument_group('Other options')
    other.add_argument('--threads', default=1, type=int, help='Number of threads to use [default = 1]')

    other.add_argument('--version', action='version',
                       version='%(prog)s '+__version__)

    return parser.parse_args()

def main():

    # Check input args ok
    args = get_options()

    import graph_tool.all as gt

    # Check output path ok
    if not os.path.isdir(args.output):
        try:
            os.makedirs(args.output)
        except OSError:
            sys.stderr.write("Cannot create output directory\n")
            sys.exit(1)
    setGtThreads(args.threads)

    # Read in all distances
    refList, queryList, self, distMat = readPickle(args.distances, enforce_self=True)

    # Read in full network
    genomeNetwork = gt.load_graph(args.network)
    sys.stderr.write("Network loaded: " + str(len(list(genomeNetwork.vertices()))) + " samples\n")

    # This is the same set of function calls for --fit-model when no --full-db in __main__.py
    # Find refs and prune network
    reference_indices, reference_names, refFileName, G_ref = \
        extractReferences(genomeNetwork, refList, args.output, threads = args.threads)
    G_ref.save(args.output + "/" + os.path.basename(args.output) + '_graph.gt', fmt = 'gt')

    # Prune distances
    nodes_to_remove = set(range(len(refList))).difference(reference_indices)
    names_to_remove = [refList[n] for n in nodes_to_remove]

    # 'Resketch'
    if len(nodes_to_remove) > 0:
        prune_distance_matrix(refList, names_to_remove, distMat,
                          args.output + "/" + os.path.basename(args.output) + ".dists")

        removeFromDB(args.ref_db, args.output, set(refList) - set(reference_names))

        db_outfile = args.output + "/" + os.path.basename(args.output) + ".tmp.h5"
        db_infile = args.output + "/" + os.path.basename(args.output) + ".h5"
        if os.path.exists(db_infile):
            sys.stderr.write("Sketch DB exists in " + args.output + "\n"
                             "Not overwriting. Output DB is: " +
                             db_outfile + "\n")
        else:
            os.rename(db_outfile, db_infile)
    else:
        sys.stderr.write("No sequences to remove\n")

    # Copy model fit into new directory
    if args.model is not None and os.path.isdir(args.model):
        sys.stderr.write("Copying model fit into " + args.output + "\n")
        copyfile(args.model + "/" + os.path.basename(args.model) + "_fit.pkl",
                 args.output + "/" + os.path.basename(args.output) + "_fit.pkl")
        copyfile(args.model + "/" + os.path.basename(args.model) + "_fit.npz",
                 args.output + "/" + os.path.basename(args.output) + "_fit.npz")
        if args.clusters is not None:
            cluster_file = args.clusters
        else:
            cluster_file = args.model + "/" + os.path.basename(args.model) + "_clusters.csv"
        copyfile(cluster_file, args.output + "/" + os.path.basename(args.output) + "_clusters.csv")

    sys.exit(0)

