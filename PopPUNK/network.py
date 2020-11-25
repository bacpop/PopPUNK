# vim: set fileencoding=<utf-8> :
# Copyright 2018-2020 John Lees and Nick Croucher

'''Network functions'''

# universal
import os
import sys
import re
# additional
import glob
import operator
import shutil
import subprocess
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from tempfile import mkstemp, mkdtemp
from collections import defaultdict, Counter
from functools import partial
from multiprocessing import Pool
import graph_tool.all as gt

from .sketchlib import addRandom

from .utils import iterDistRows
from .utils import listDistInts
from .utils import readIsolateTypeFromCsv
from .utils import readRfile
from .utils import setupDBFuncs
from .utils import isolateNameToLabel

def fetchNetwork(network_dir, model, refList, ref_graph = False,
                  core_only = False, accessory_only = False):
    """Load the network based on input options

       Returns the network as a graph-tool format graph, and sets
       the slope parameter of the passed model object.

       Args:
            network_dir (str)
                A network used to define clusters from :func:`~constructNetwork`
            model (ClusterFit)
                A fitted model object
            refList (list)
                Names of references that should be in the network
            ref_graph (bool)
                Use ref only graph, if available

                [default = False]
            core_only (bool)
                Return the network created using only core distances

                [default = False]
            accessory_only (bool)
                Return the network created using only accessory distances

                [default = False]

       Returns:
            genomeNetwork (graph)
                The loaded network
            cluster_file (str)
                The CSV of cluster assignments corresponding to this network
    """
    # If a refined fit, may use just core or accessory distances
    dir_prefix = network_dir + "/" + os.path.basename(network_dir)
    if core_only and model.type == 'refine':
        model.slope = 0
        network_file = dir_prefix + '_core_graph.gt'
        cluster_file = dir_prefix + '_core_clusters.csv'
    elif accessory_only and model.type == 'refine':
        model.slope = 1
        network_file = dir_prefix + '_accessory_graph.gt'
        cluster_file = dir_prefix + '_accessory_clusters.csv'
    else:
        if ref_graph and os.path.isfile(dir_prefix + '.refs_graph.gt'):
            network_file = dir_prefix + '.refs_graph.gt'
        else:
            network_file = dir_prefix + '_graph.gt'
        cluster_file = dir_prefix + '_clusters.csv'
        if core_only or accessory_only:
            sys.stderr.write("Can only do --core-only or --accessory-only fits from "
                             "a refined fit. Using the combined distances.\n")

    genomeNetwork = gt.load_graph(network_file)
    sys.stderr.write("Network loaded: " + str(len(list(genomeNetwork.vertices()))) + " samples\n")

    # Ensure all in dists are in final network
    networkMissing = set(map(str,set(range(len(refList))).difference(list(genomeNetwork.vertices()))))
    if len(networkMissing) > 0:
        sys.stderr.write("WARNING: Samples " + ",".join(networkMissing) + " are missing from the final network\n")

    return (genomeNetwork, cluster_file)

def getCliqueRefs(G, reference_indices = set()):
    """Recursively prune a network of its cliques. Returns one vertex from
    a clique at each stage

    Args:
        G (graph)
            The graph to get clique representatives from
        reference_indices (set)
            The unique list of vertices being kept, to add to
    """
    cliques = gt.max_cliques(G)
    try:
        # Get the first clique, and see if it has any members already
        # contained in the vertex list
        clique = frozenset(next(cliques))
        if clique.isdisjoint(reference_indices):
            reference_indices.add(list(clique)[0])

        # Remove the clique, and prune the resulting subgraph (recursively)
        subgraph = gt.GraphView(G, vfilt=[v not in clique for v in G.vertices()])
        if subgraph.num_vertices() > 1:
            getCliqueRefs(subgraph, reference_indices)
        elif subgraph.num_vertices() == 1:
            reference_indices.add(subgraph.get_vertices()[0])
    except StopIteration:
        pass
    return reference_indices

def cliquePrune(component, graph, reference_indices, components_list):
    """Wrapper function around :func:`~getCliqueRefs` so it can be
       called by a multiprocessing pool
    """
    subgraph = gt.GraphView(graph, vfilt=components_list == component)
    refs = reference_indices.copy()
    if subgraph.num_vertices() <= 2:
        refs.add(subgraph.get_vertices()[0])
        ref_list = refs
    else:
        ref_list = getCliqueRefs(subgraph, refs)
    return(list(ref_list))

def extractReferences(G, dbOrder, outPrefix, existingRefs = None, threads = 1):
    """Extract references for each cluster based on cliques

       Writes chosen references to file by calling :func:`~writeReferences`

       Args:
           G (graph)
               A network used to define clusters from :func:`~constructNetwork`
           dbOrder (list)
               The order of files in the sketches, so returned references are in the same order
           outPrefix (str)
               Prefix for output file (.refs will be appended)
           existingRefs (list)
               References that should be used for each clique

       Returns:
           refFileName (str)
               The name of the file references were written to
           references (list)
               An updated list of the reference names
    """
    if existingRefs == None:
        references = set()
        reference_indices = set()
    else:
        references = set(existingRefs)
        index_lookup = {v:k for k,v in enumerate(dbOrder)}
        reference_indices = set([index_lookup[r] for r in references])

    # Each component is independent, so can be multithreaded
    components = gt.label_components(G)[0].a

    # Turn gt threading off and on again either side of the parallel loop
    if gt.openmp_enabled():
        gt.openmp_set_num_threads(1)

    # Cliques are pruned, taking one reference from each, until none remain
    with Pool(processes=threads) as pool:
        ref_lists = pool.map(partial(cliquePrune,
                                        graph=G,
                                        reference_indices=reference_indices,
                                        components_list=components),
                             set(components))
    # Returns nested lists, which need to be flattened
    reference_indices = set([entry for sublist in ref_lists for entry in sublist])

    if gt.openmp_enabled():
        gt.openmp_set_num_threads(threads)

    # Use a vertex filter to extract the subgraph of refences
    # as a graphview
    reference_vertex = G.new_vertex_property('bool')
    for n, vertex in enumerate(G.vertices()):
        if n in reference_indices:
            reference_vertex[vertex] = True
        else:
            reference_vertex[vertex] = False
    G_ref = gt.GraphView(G, vfilt = reference_vertex)
    G_ref = gt.Graph(G_ref, prune = True) # https://stackoverflow.com/questions/30839929/graph-tool-graphview-object

    # Find any clusters which are represented by >1 references
    # This creates a dictionary: cluster_id: set(ref_idx in cluster)
    clusters_in_full_graph = printClusters(G, dbOrder, printCSV=False)
    reference_clusters_in_full_graph = defaultdict(set)
    for reference_index in reference_indices:
        reference_clusters_in_full_graph[clusters_in_full_graph[dbOrder[reference_index]]].add(reference_index)

    # Calculate the component membership within the reference graph
    ref_order = [name for idx, name in enumerate(dbOrder) if idx in frozenset(reference_indices)]
    clusters_in_reference_graph = printClusters(G_ref, ref_order, printCSV=False)
    # Record the components/clusters the references are in the reference graph
    # dict: name: ref_cluster
    reference_clusters_in_reference_graph = {}
    for reference_name in ref_order:
        reference_clusters_in_reference_graph[reference_name] = clusters_in_reference_graph[reference_name]

    # Check if multi-reference components have been split as a validation test
    # First iterate through clusters
    network_update_required = False
    for cluster_id, ref_idxs in reference_clusters_in_full_graph.items():
        # Identify multi-reference clusters by this length
        if len(ref_idxs) > 1:
            check = list(ref_idxs)
            # check if these are still in the same component in the reference graph
            for i in range(len(check)):
                component_i = reference_clusters_in_reference_graph[dbOrder[check[i]]]
                for j in range(i + 1, len(check)):
                    # Add intermediate nodes
                    component_j = reference_clusters_in_reference_graph[dbOrder[check[j]]]
                    if component_i != component_j:
                        network_update_required = True
                        vertex_list, edge_list = gt.shortest_path(G, check[i], check[j])
                        # update reference list
                        for vertex in vertex_list:
                            reference_vertex[vertex] = True
                            reference_indices.add(int(vertex))

    # update reference graph if vertices have been added
    if network_update_required:
        G_ref = gt.GraphView(G, vfilt = reference_vertex)
        G_ref = gt.Graph(G_ref, prune = True) # https://stackoverflow.com/questions/30839929/graph-tool-graphview-object

    # Order found references as in mash sketch files
    reference_names = [dbOrder[int(x)] for x in sorted(reference_indices)]
    refFileName = writeReferences(reference_names, outPrefix)
    return reference_indices, reference_names, refFileName, G_ref

def writeReferences(refList, outPrefix):
    """Writes chosen references to file

    Args:
        refList (list)
            Reference names to write
        outPrefix (str)
            Prefix for output file (.refs will be appended)

    Returns:
        refFileName (str)
            The name of the file references were written to
    """
    # write references to file
    refFileName = outPrefix + "/" + os.path.basename(outPrefix) + ".refs"
    with open(refFileName, 'w') as rFile:
        for ref in refList:
            rFile.write(ref + '\n')

    return refFileName

def constructNetwork(rlist, qlist, assignments, within_label,
                     summarise = True, edge_list = False, weights = None):
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
        edge_list (bool)
            Whether input is edges, tuples of (v1, v2). Used with lineage assignment
        weights (numpy.array)
            If passed, the core,accessory distances for each assignment, which will
            be annotated as an edge attribute

    Returns:
        G (graph)
            The resulting network
    """
    # data structures
    connections = []
    self_comparison = True
    vertex_labels = rlist

    # check if self comparison
    if rlist != qlist:
        self_comparison = False
        vertex_labels.append(qlist)

    # identify edges
    if edge_list:
        if weights is not None:
            connections = []
            for weight, (ref, query) in zip(weights, assignments):
                connections.append((ref, query, weight))
        else:
            connections = assignments
    else:
        for row_idx, (assignment, (ref, query)) in enumerate(zip(assignments,
                                                                 listDistInts(rlist, qlist,
                                                                              self = self_comparison))):
            if assignment == within_label:
                if weights is not None:
                    dist = np.linalg.norm(weights[row_idx, :])
                    edge_tuple = (ref, query, dist)
                else:
                    edge_tuple = (ref, query)
                connections.append(edge_tuple)

    # build the graph
    G = gt.Graph(directed = False)
    G.add_vertex(len(vertex_labels))

    if weights is not None:
        eweight = G.new_ep("float")
        G.add_edge_list(connections, eprops = [eweight])
        G.edge_properties["weight"] = eweight
    else:
        G.add_edge_list(connections)

    # add isolate ID to network
    vid = G.new_vertex_property('string',
                                vals = vertex_labels)
    G.vp.id = vid

    # print some summaries
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
        G (graph)
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
    component_assignments, component_frequencies = gt.label_components(G)
    components = len(component_frequencies)
    density = len(list(G.edges()))/(0.5 * len(list(G.vertices())) * (len(list(G.vertices())) - 1))
    transitivity = gt.global_clustering(G)[0]
    score = transitivity * (1-density)

    return(components, density, transitivity, score)

def addQueryToNetwork(dbFuncs, rList, qList, G, kmers,
                      assignments, model, queryDB, queryQuery = False,
                      strand_preserved = False, weights = None, threads = 1):
    """Finds edges between queries and items in the reference database,
    and modifies the network to include them.

    Args:
        dbFuncs (list)
            List of backend functions from :func:`~PopPUNK.utils.setupDBFuncs`
        rList (list)
            List of reference names
        qList (list)
            List of query names
        G (graph)
            Network to add to (mutated)
        kmers (list)
            List of k-mer sizes
        assignments (numpy.array)
            Cluster assignment of items in qlist
        model (ClusterModel)
            Model fitted to reference database
        queryDB (str)
            Query database location
        queryQuery (bool)
            Add in all query-query distances
            (default = False)
        strand_preserved (bool)
            Whether to treat strand as known (i.e. ignore rc k-mers)
            when adding random distances. Only used if queryQuery = True
            [default = False]
        weights (numpy.array)
            If passed, the core,accessory distances for each assignment, which will
            be annotated as an edge attribute
        threads (int)
            Number of threads to use if new db created

            (default = 1)
    Returns:
        distMat (numpy.array)
            Query-query distances
    """
    # initalise functions
    queryDatabase = dbFuncs['queryDatabase']

    # initialise links data structure
    new_edges = []
    assigned = set()

    # These are returned
    qqDistMat = None

    # store links for each query in a list of edge tuples
    ref_count = len(rList)
    for row_idx, (assignment, (ref, query)) in enumerate(zip(assignments, listDistInts(rList, qList, self = False))):
        if assignment == model.within_label:
            # query index needs to be adjusted for existing vertices in network
            if weights is not None:
                dist = np.linalg.norm(weights[row_idx, :])
                edge_tuple = (ref, query + ref_count, dist)
            else:
                edge_tuple = (ref, query + ref_count)
            new_edges.append(edge_tuple)
            assigned.add(qList[query])

    # Calculate all query-query distances too, if updating database
    if queryQuery:
        sys.stderr.write("Calculating all query-query distances\n")
        addRandom(queryDB, qList, kmers, strand_preserved, threads = threads)
        qlist1, qlist2, qqDistMat = queryDatabase(rNames = qList,
                                                  qNames = qList,
                                                  dbPrefix = queryDB,
                                                  queryPrefix = queryDB,
                                                  klist = kmers,
                                                  self = True,
                                                  number_plot_fits = 0,
                                                  threads = threads)

        queryAssignation = model.assign(qqDistMat)
        for row_idx, (assignment, (ref, query)) in enumerate(zip(queryAssignation, listDistInts(qList, qList, self = True))):
            if assignment == model.within_label:
                if weights is not None:
                    dist = np.linalg.norm(qqDistMat[row_idx, :])
                    edge_tuple = (ref + ref_count, query + ref_count, dist)
                else:
                    edge_tuple = (ref + ref_count, query + ref_count)
                new_edges.append(edge_tuple)

    # Otherwise only calculate query-query distances for new clusters
    else:
        # identify potentially new lineages in list: unassigned is a list of queries with no hits
        unassigned = set(qList).difference(assigned)
        query_indices = {k:v+ref_count for v,k in enumerate(qList)}
        # process unassigned query sequences, if there are any
        if len(unassigned) > 1:
            sys.stderr.write("Found novel query clusters. Calculating distances between them.\n")

            # use database construction methods to find links between unassigned queries
            addRandom(queryDB, qList, kmers, strand_preserved, threads = threads)
            qlist1, qlist2, qqDistMat = queryDatabase(rNames = list(unassigned),
                                                    qNames = list(unassigned),
                                                    dbPrefix = queryDB,
                                                    queryPrefix = queryDB,
                                                    klist = kmers,
                                                    self = True,
                                                    number_plot_fits = 0,
                                                    threads = threads)

            queryAssignation = model.assign(qqDistMat)

            # identify any links between queries and store in the same links dict
            # links dict now contains lists of links both to original database and new queries
            # have to use names and link to query list in order to match to node indices
            for row_idx, (assignment, (query1, query2)) in enumerate(zip(queryAssignation, iterDistRows(qlist1, qlist2, self = True))):
                if assignment == model.within_label:
                    if weights is not None:
                        dist = np.linalg.norm(qqDistMat[row_idx, :])
                        edge_tuple = (query_indices[query1], query_indices[query2], dist)
                    else:
                        edge_tuple = (query_indices[query1], query_indices[query2])
                    new_edges.append(edge_tuple)

    # finish by updating the network
    G.add_vertex(len(qList))

    if weights is not None:
        eweight = G.new_ep("float")
        G.add_edge_list(new_edges, eprops = [eweight])
        G.edge_properties["weight"] = eweight
    else:
        G.add_edge_list(new_edges)

    # including the vertex ID property map
    for i, q in enumerate(qList):
        G.vp.id[i + len(rList)] = q

    return qqDistMat

def printClusters(G, rlist, outPrefix = "_clusters.csv", oldClusterFile = None,
                  externalClusterCSV = None, printRef = True, printCSV = True,
                  clustering_type = 'combined'):
    """Get cluster assignments

    Also writes assignments to a CSV file

    Args:
        G (graph)
            Network used to define clusters (from :func:`~constructNetwork` or
            :func:`~addQueryToNetwork`)
        outPrefix (str)
            Prefix for output CSV
            Default = "_clusters.csv"
        oldClusterFile (str)
            CSV with previous cluster assignments.
            Pass to ensure consistency in cluster assignment name.
            Default = None
        externalClusterCSV (str)
            CSV with cluster assignments from any source. Will print a file
            relating these to new cluster assignments
            Default = None
        printRef (bool)
            If false, print only query sequences in the output
            Default = True
        printCSV (bool)
            Print results to file
            Default = True
        clustering_type (str)
            Type of clustering network, used for comparison with old clusters
            Default = 'combined'

    Returns:
        clustering (dict)
            Dictionary of cluster assignments (keys are sequence names)

    """
    if oldClusterFile == None and printRef == False:
        raise RuntimeError("Trying to print query clusters with no query sequences")

    # get a sorted list of component assignments
    component_assignments, component_frequencies = gt.label_components(G)
    component_frequency_ranks = len(component_frequencies) - rankdata(component_frequencies, method = 'ordinal').astype(int)
    newClusters = [set() for rank in range(len(component_frequency_ranks))]
    for isolate_index, isolate_name in enumerate(rlist):
        component = component_assignments.a[isolate_index]
        component_rank = component_frequency_ranks[component]
        newClusters[component_rank].add(isolate_name)

    oldNames = set()

    if oldClusterFile != None:
        oldAllClusters = readIsolateTypeFromCsv(oldClusterFile, mode = 'external', return_dict = False)
        oldClusters = oldAllClusters[list(oldAllClusters.keys())[0]]
        new_id = len(oldClusters.keys()) + 1 # 1-indexed
        while new_id in oldClusters:
            new_id += 1 # in case clusters have been merged

        # Samples in previous clustering
        for prev_cluster in oldClusters.values():
            for prev_sample in prev_cluster:
                oldNames.add(prev_sample)

    # Assign each cluster a name
    clustering = {}
    foundOldClusters = []

    for newClsIdx, newCluster in enumerate(newClusters):

        # Ensure consistency with previous labelling
        if oldClusterFile != None:
            merge = False
            cls_id = None

            # Samples in this cluster that are not queries
            ref_only = oldNames.intersection(newCluster)

            # A cluster with no previous observations
            if len(ref_only) == 0:
                cls_id = str(new_id)    # harmonise data types; string flexibility helpful
                new_id += 1
            else:
                # Search through old cluster IDs to find a match
                for oldClusterName, oldClusterMembers in oldClusters.items():
                    join = ref_only.intersection(oldClusterMembers)
                    if len(join) > 0:
                        # Check cluster is consistent with previous definitions
                        if oldClusterName in foundOldClusters:
                            sys.stderr.write("WARNING: Old cluster " + oldClusterName + " split"
                                             " across multiple new clusters\n")
                        else:
                            foundOldClusters.append(oldClusterName)

                        # Query has merged clusters
                        if len(join) < len(ref_only):
                            merge = True
                            if cls_id == None:
                                cls_id = oldClusterName
                            else:
                                cls_id += "_" + oldClusterName
                        # Exact match -> same name as before
                        elif len(join) == len(ref_only):
                            assert merge == False # should not have already been part of a merge
                            cls_id = oldClusterName
                            break

            # Report merges
            if merge:
                merged_ids = cls_id.split("_")
                sys.stderr.write("Clusters " + ",".join(merged_ids) + " have merged into " + cls_id + "\n")

        # Otherwise just number sequentially starting from 1
        else:
            cls_id = newClsIdx + 1

        for cluster_member in newCluster:
            clustering[cluster_member] = cls_id

    # print clustering to file
    if printCSV:
        outFileName = outPrefix + "_clusters.csv"
        with open(outFileName, 'w') as cluster_file:
            cluster_file.write("Taxon,Cluster\n")

            # sort the clusters by frequency - define a list with a custom sort order
            # first line gives tuples e.g. (1, 28), (2, 17) - cluster 1 has 28 members, cluster 2 has 17 members
            # second line takes first element - the cluster IDs sorted by frequency
            freq_order = sorted(dict(Counter(clustering.values())).items(), key=operator.itemgetter(1), reverse=True)
            freq_order = [x[0] for x in freq_order]

            # iterate through cluster dictionary sorting by value using above custom sort order
            for cluster_member, cluster_name in sorted(clustering.items(), key=lambda i:freq_order.index(i[1])):
                if printRef or cluster_member not in oldNames:
                    cluster_file.write(",".join((cluster_member, str(cluster_name))) + "\n")

        if externalClusterCSV is not None:
            printExternalClusters(newClusters, externalClusterCSV, outPrefix, oldNames, printRef)

    return(clustering)

def printExternalClusters(newClusters, extClusterFile, outPrefix,
                          oldNames, printRef = True):
    """Prints cluster assignments with respect to previously defined
    clusters or labels.

    Args:
        newClusters (set iterable)
            The components from the graph G, defining the PopPUNK clusters
        extClusterFile (str)
            A CSV file containing definitions of the external clusters for
            each sample (does not need to contain all samples)
        outPrefix (str)
            Prefix for output CSV (_external_clusters.csv)
        oldNames (list)
            A list of the reference sequences
        printRef (bool)
            If false, print only query sequences in the output

            Default = True
    """
    # Object to store output csv datatable
    d = defaultdict(list)

    # Read in external clusters
    extClusters = \
        readIsolateTypeFromCsv(extClusterFile,
                               mode = 'external',
                               return_dict = True)

    # Go through each cluster (as defined by poppunk) and find the external
    # clusters that had previously been assigned to any sample in the cluster
    for ppCluster in newClusters:
        # Store clusters as a set to avoid duplicates
        prevClusters = defaultdict(set)
        for sample in ppCluster:
            for extCluster in extClusters:
                if sample in extClusters[extCluster]:
                    prevClusters[extCluster].add(extClusters[extCluster][sample])

        # Go back through and print the samples that were found
        for sample in ppCluster:
            if printRef or sample not in oldNames:
                d['sample'].append(sample)
                for extCluster in extClusters:
                    if extCluster in prevClusters:
                        d[extCluster].append(";".join(prevClusters[extCluster]))
                    else:
                        d[extCluster].append("NA")

    pd.DataFrame(data=d).to_csv(outPrefix + "_external_clusters.csv",
                                columns = ["sample"] + list(extClusters.keys()),
                                index = False)
