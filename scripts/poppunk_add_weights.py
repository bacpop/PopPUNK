#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018-2020 John Lees and Nick Croucher

import sys
import argparse
import pickle
import numpy as np

# command line parsing
def get_options():

    parser = argparse.ArgumentParser(description='Add edge weights to a PopPUNK graph',
                                     prog='add_weights')

    # input options
    parser.add_argument('graph', help='Input graph (.gt)')
    parser.add_argument('distances', help='Prefix for distances (<name>.dists)')
    parser.add_argument('output', help='Prefix for output graph')

    parser.add_argument('--graphml', help='Save output as graphml file',
                        default=False, action='store_true')

    return parser.parse_args()

def quit_msg(message):
    sys.stderr.write(str(message) + "\n")
    sys.exit(1)

# Convert an i, j square index to long form (see sketchlib for info)
def square_to_condensed(i, j, n):
    assert (j > i)
    return n*i - ((i*(i+1)) >> 1) + j - 1 - i

# main code
if __name__ == "__main__":

    # Get command line options
    args = get_options()

    # Load network
    import graph_tool.all as gt
    G = gt.load_graph(args.graph)
    if "weight" in G.edge_properties:
        quit_msg("Graph already contains weights")

    # Load dists
    with open(args.distances + ".pkl", 'rb') as pickle_file:
        rlist, qlist, self = pickle.load(pickle_file)
        if not self:
            quit_msg("Distances are from query mode")
    dist_mat = np.load(args.distances + ".npy")

    # Check network and dists are compatible
    network_labels = G.vertex_properties["id"]
    if set(network_labels) != set(rlist):
        quit_msg("Names in distances do not match those in graph")
    n = G.num_vertices()
    assert(0.5 * n * (n-1) == dist_mat.shape[0])

    # Match dist row order with network order
    network_order = list(network_labels)
    if rlist != network_order:
        v_idx = [network_order.index(rname) for rname in rlist]
    else:
        v_idx = range(n)

    eprop = G.new_edge_property("float")
    for edge in G.edges():
        v1, v2 = sorted(tuple(edge))
        row_idx = square_to_condensed(v_idx[int(v1)], v_idx[int(v2)], n)
        dist = np.linalg.norm(dist_mat[row_idx, :])
        eprop[edge] = dist

    # Add as edge attribute
    G.edge_properties["weight"] = eprop
    if args.graphml:
        G.save(args.output + ".graphml", fmt="graphml")
    else:
        G.save(args.output + ".gt")

    sys.exit(0)
