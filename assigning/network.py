
'''Network functions'''

import sys
import pickle
import numpy as np
import networkx as nx
import gzip, io

from .bgmm import assign_samples

def build_network(input_file, scaler_file, gmm_fit_file, gzip=True):
    scaler = pickle.load(open(scaler_file, "rb"))
    gmm_fit = np.load(gmm_fit_file)

    within_label = np.argmin(np.apply_along_axis(np.linalg.norm, 1, gmm_fit['means']))
    samples = set()
    connections = []
    distances = []
    if (gzip):
        with io.TextIOWrapper(io.BufferedReader(gzip.open(input_file, 'rb'))) as distances:
            for line in distances:
                (sample1, sample2, core_dist, accessory_dist) = line.rstrip().split(" ")
                samples.add(sample1)
                if sample1 != sample2: # no self-edges
                    connections.append((sample1, sample2))
                    distances.append((core_dist, accessory_dist))

    scaled = scaler.transform(np.array(distances))
    assignments = assign_samples(scaled, gmm_fit['weights'], gmm_fit['means'], gmm_fit['covariances'])

    # build the graph
    G = nx.Graph()
    G.add_nodes_from(samples)
    for assignment, connection in zip(assignments, connections):
        if assignment == within_label:
            G.add_edge(*connection)

    return(G)

def print_clusters(network, outfile):
    clusters = sorted(nx.connected_components(network), key=len, reverse=True)
    cl_id = 0
    with open(outfile, 'w') as cluster_file:
        for cl_id, cluster in enumerate(clusters):
            for cluster_member in cluster:
                cluster_file.write("\t".join((str(cl_id), cluster_member)) + "\n")

# TODO
def print_representatives(network, outfile):
    cliques = list(nx.find_cliques(network))
