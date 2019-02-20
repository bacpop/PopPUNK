#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018 John Lees and Nick Croucher

# universal
import os
import sys
# additional
from shutil import copyfile
import networkx as nx

# import poppunk package
from .__init__ import __version__

from .mash import checkMashVersion
from .mash import createDatabaseDir
from .mash import constructDatabase
from .mash import getKmersFromReferenceDatabase
from .mash import getSketchSize

from .network import extractReferences

from .prune_db import prune_distance_matrix

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
    oGroup.add_argument('--no-resketch', default=False, action='store_true', help='Do not resketch the references '
                                                                                 '[default = False]')

    # processing
    other = parser.add_argument_group('Other options')
    other.add_argument('--mash', default='mash', help='Location of mash executable')
    other.add_argument('--threads', default=1, type=int, help='Number of threads to use [default = 1]')

    other.add_argument('--version', action='version',
                       version='%(prog)s '+__version__)

    return parser.parse_args()

def main():

    # Check input args ok
    args = get_options()
    checkMashVersion(args.mash)
    if not args.no_resketch and (args.ref_db is None or not os.path.isdir(args.ref_db)):
        sys.stderr.write("Must provide original --ref-db if using --resketch\n")
        sys.exit(1)

    # Check output path ok
    if not os.path.isdir(args.output):
        try:
            os.makedirs(args.output)
        except OSError:
            sys.stderr.write("Cannot create output directory\n")
            sys.exit(1)

    # Read in all distances
    refList, queryList, self, distMat = readPickle(args.distances)
    if not self:
        raise RuntimeError("Distance DB should be self-self distances")

    # Read in full network
    genomeNetwork = nx.read_gpickle(args.network)
    sys.stderr.write("Network loaded: " + str(genomeNetwork.number_of_nodes()) + " samples\n")

    # This is the same set of function calls for --fit-model when no --full-db in __main__.py
    # Find refs and prune network
    newReferencesNames, newReferencesFile = extractReferences(genomeNetwork, refList, args.output)
    nodes_to_remove = set(refList).difference(newReferencesNames)
    genomeNetwork.remove_nodes_from(nodes_to_remove)
    nx.write_gpickle(genomeNetwork, args.output + "/" + os.path.basename(args.output) + '_graph.gpickle')

    # Prune distances
    prune_distance_matrix(refList, nodes_to_remove, distMat,
                          args.output + "/" + os.path.basename(args.output) + ".dists")

    # Resketch
    if len(nodes_to_remove) > 0:
        if not args.no_resketch:
            sys.stderr.write("Resketching " + str(len(newReferencesNames)) + " sequences\n")

            # Find db properties
            kmers = getKmersFromReferenceDatabase(args.ref_db)
            sketch_sizes = getSketchSize(args.ref_db, kmers, args.mash)

            # Resketch all
            createDatabaseDir(args.output, kmers)
            constructDatabase(newReferencesFile, kmers, sketch_sizes, args.output, True, args.threads, args.mash, True)

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

