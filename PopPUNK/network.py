'''Network functions'''

# universal
import os
import sys
import re
# additional
import glob
import shutil
import subprocess
import networkx as nx
import numpy as np
from collections import defaultdict
from tempfile import mkstemp, mkdtemp

from .mash import createDatabaseDir
from .mash import constructDatabase
from .mash import queryDatabase
from .mash import getDatabaseName
from .mash import getSketchSize
from .mash import iterDistRows

def extractReferences(G, outPrefix):
    """Extract references for each cluster based on cliques

    Writes chosen references to file

    Args:
        G (networkx.Graph)
            A network used to define clusters from :func:`~constructNetwork`
        outPrefix (str)
            Prefix for output file (.refs will be appended)

    Returns:
        refFileName (str)
            The name of the file references were written to
        references (list)
            A list of the reference names
    """
    # define reference list
    references = []
    # extract cliques from network
    cliques = list(nx.find_cliques(G))
    # order list by size of clique
    cliques.sort(key = len, reverse=True)
    # iterate through cliques
    for clique in cliques:
        alreadyRepresented = 0
        for node in clique:
            if node in references:
                alreadyRepresented = 1
        if alreadyRepresented == 0:
            references.append(clique[0])

    # write references to file
    refFileName = "./" + outPrefix + "/" + outPrefix + ".refs"
    with open(refFileName, 'w') as rFile:
        for ref in references:
            rFile.write(ref + '\n')

    return references, refFileName

def constructNetwork(rlist, qlist, assignments, within_label, summarise = True):
    """Construct an unweighted, undirected network without self-loops.
    Nodes are samples and edges where samples are within the same cluster

    Will print summary statistics about the network to ``STDERR``

    Args:
        rlist (list)
            List of reference sequence labels
        qlist (list)
            List of query sequence labels
        assignments (numpy.array)
            Labels of most likely cluster assignment from :func:`~PopPUNK.bgmm.assign_samples`
        within_label (int)
            The label for the cluster representing within-strain distances
            from :func:`~PopPUNK.bgmm.findWithinLabel`
        summarise (bool)
            Whether to calculate and print network summaries with :func:`~networkSummary`

            (default = True)

    Returns:
        G (networkx.Graph)
            The resulting network
    """
    connections = []
    for assignment, (ref, query) in zip(assignments, iterDistRows(rlist, qlist, self=True)):
        if assignment == within_label:
            connections.append((ref, query))

    density_proportion = len(connections) / (0.5 * (len(rlist) * (len(rlist) + 1)))
    if density_proportion > 0.2 or len(connections) > 100000:
        sys.stderr.write("Warning: trying to create very large network\n")

    # build the graph
    G = nx.Graph()
    G.add_nodes_from(rlist)
    for connection in connections:
        G.add_edge(*connection)

    # give some summaries
    if summarise:
        (components, density, transitivity, score) = networkSummary(G)
        sys.stderr.write("Network summary:\n" + "\n".join(["\tComponents\t" + str(components),
                                                       "\tDensity\t" + "{:.4f}".format(density),
                                                       "\tTransitivity\t" + "{:.4f}".format(transitivity),
                                                       "\tScore\t" + "{:.4f}".format(score)])
                                                       + "\n")

    return G

def networkSummary(G):
    """Provides summary values about the network

    Args:
        G (networkx.Graph)
            The network of strains from :func:`~constructNetwork`

    Returns:
        components (int)
            The number of connected components (and clusters)
        density (float)
            The proportion of possible edges used
        transitivity (float)
            Network transitivity (triads/triangles)
        score (float)
            A score of network fit, given by :math:`\mathrm{transitivity} * (1-\mathrm{density})`
    """
    components = nx.number_connected_components(G)
    density = nx.density(G)
    transitivity = nx.transitivity(G)
    score = transitivity * (1-density)

    return(components, density, transitivity, score)

def addQueryToNetwork(rlist, qlist, G, kmers, assignments, model,
        dbPrefix, threads = 1, mash_exec = 'mash'):
    """Finds edges between queries and items in the reference database,
    and modifies the network to include them.

    Args:
        rlist (list)
            List of reference names
        qlist (list)
            List of query names
        G (networkx.Graph)
            Network to add to (mutated)
        kmers (list)
            List of k-mer sizes
        assignments (numpy.array)
            Cluster assignment of items in qlist
        model (ClusterModel)
            Model fitted to reference database
        threads (int)
            Number of threads to use if new db created

            (default = 1)
        mash_exec (str)
            Location of the mash executable

            (default = 'mash')
    """
    # initialise links data structure
    new_edges = []
    assigned = set()

    # store links for each query in a list of edge tuples
    for assignment, (ref, query) in zip(assignments, iterDistRows(rlist, qlist, self=False)):
        if assignment == model.within_label:
            new_edges.append((ref, query))
            assigned.add(query)


    # identify potentially new lineages in list: unassigned is a list of queries with no hits
    unassigned = set(qlist).difference(assigned)

    # process unassigned query sequences, if there are any
    if len(unassigned) > 1:
        sys.stderr.write("Found novel query clusters. Calculating distances between them:\n")

        # write unassigned queries to file as if a list of references
        tmpDirName = mkdtemp(prefix=dbPrefix, suffix="_tmp", dir="./")
        tmpHandle, tmpFile = mkstemp(prefix=dbPrefix, suffix="_tmp", dir=tmpDirName)
        with open(tmpFile, 'w') as tFile:
            for query in unassigned:
                tFile.write(query + '\n')

        # use database construction methods to find links between unassigned queries
        sketchSize = getSketchSize(dbPrefix, kmers, mash_exec)
        constructDatabase(tmpFile, kmers, sketchSize, tmpDirName, threads, mash_exec)
        qlist1, qlist2, distMat = queryDatabase(tmpHandle, kmers, tmpDirName, tmpDirName, True,
                0, False, mash_exec = mash_exec, threads = threads)
        queryAssignation = model.assign(distMat)

        # identify any links between queries and store in the same links dict
        # links dict now contains lists of links both to original database and new queries
        for assignment, (query1, query2) in zip(queryAssignation, iterDistRows(qlist1, qlist2, self=True)):
            if assignment == model.within_label:
                new_edges.append((query1, query2))

        # remove directory
        shutil.rmtree(tmpDirName)

    # finish by updating the network
    G.add_nodes_from(qlist)
    G.add_edges_from(new_edges)


def printClusters(G, outPrefix, toPrint = None):
    """Get cluster assignments

    Also writes assignments to a CSV file

    Args:
        G (networkx.Graph)
            Network used to define clusters (from :func:`~constructNetwork`)
        outPrefix (str)
            Prefix for output CSV (_clusters.csv)
        toPrint (list)
            If passed, print only IDs from this list

    Returns:
        clustering (dict)
            Dictionary of cluster assignments (keys are sequence names)
    """
    # data structure
    clustering = {}

    # identify network components
    clusters = sorted(nx.connected_components(G), key=len, reverse=True)
    outFileName = outPrefix + "/" + outPrefix + "_clusters.csv"

    # print clustering to file
    with open(outFileName, 'w') as cluster_file:
        cluster_file.write("Taxon,Cluster\n")
        for cl_id, cluster in enumerate(clusters):
            for cluster_member in cluster:
                clustering[cluster_member] = cl_id
                if toPrint != None and cluster_member in toPrint:
                    cluster_file.write(",".join((cluster_member,str(cl_id))) + "\n")

    return clustering


