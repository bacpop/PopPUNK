#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018 John Lees and Nick Croucher

import sys
from scipy.stats import rankdata
import argparse

# command line parsing
def get_options():

    parser = argparse.ArgumentParser(description='Extract graphml files of each component for '
                                                 'individual visualisation',
                                     prog='extract_components')

    # input options
    parser.add_argument('--graph', help='Input graph pickle (.gt)')
    parser.add_argument('--output', help='Prefix for output files')

    return parser.parse_args()

# main code
if __name__ == "__main__":

    # Check input ok
    args = get_options()

    # open stored graph
    import graph_tool.all as gt
    G = gt.load_graph(args.graph)

    # extract individual components
    component_assignments, component_frequencies = gt.label_components(G)
    component_frequency_ranks = len(component_frequencies) - rankdata(component_frequencies, method = 'ordinal').astype(int)
    sys.stderr.write("Writing " + str(len(component_frequencies)) + " components "
                     "in reverse order of size\n")

    # extract as GraphView objects and print
    for component_index in range(len(component_frequency_ranks)):
        component_gv = gt.GraphView(G, vfilt = component_assignments.a == component_index)
        component_G = gt.Graph(component_gv, prune = True)
        component_fn = args.output + ".component_" + str(component_frequency_ranks[component_index]) + ".graphml"
        component_G.save(component_fn, fmt = 'graphml')

    sys.exit(0)
