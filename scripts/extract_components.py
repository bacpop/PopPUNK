#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018 John Lees and Nick Croucher

import sys
import networkx as nx
import argparse

# command line parsing
def get_options():

    parser = argparse.ArgumentParser(description='Extract graphml files of each component for '
                                                 'individual visualisation',
                                     prog='extract_components')

    # input options
    parser.add_argument('graph', help='Input graph pickle (.gpickle)')
    parser.add_argument('output', help='Prefix for output files')

    return parser.parse_args()

# main code
if __name__ == "__main__":

    # Check input ok
    args = get_options()

    # open stored distances
    G = nx.read_gpickle(args.graph)
    sys.stderr.write("Writing " + str(nx.number_connected_components(G)) + " components "
                     "in reverse order of size\n")

    components = sorted(nx.connected_components(G), key=len, reverse=True)
    for component_idx, component in enumerate(components):
        nx.write_graphml(G.subgraph(component), args.output + ".component_" + str(component_idx + 1) + ".graphml")

    sys.exit(0)
