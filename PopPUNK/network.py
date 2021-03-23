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
import pickle
import graph_tool.all as gt
import dendropy

# GPU support
try:
    import cugraph
    import cudf
    gpu_lib = True
except ImportError as e:
    gpu_lib = False

from .__main__ import accepted_weights_types

from .sketchlib import addRandom

from .utils import iterDistRows
from .utils import listDistInts
from .utils import readIsolateTypeFromCsv
from .utils import readRfile
from .utils import setupDBFuncs
from .utils import isolateNameToLabel

from .unwords import gen_unword

def fetchNetwork(network_dir, model, refList, ref_graph = False,
                  core_only = False, accessory_only = False, use_gpu = False):
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
            use_gpu (bool)
                Use cugraph library to load graph

       Returns:
            genomeNetwork (graph)
                The loaded network
            cluster_file (str)
                The CSV of cluster assignments corresponding to this network
    """
    # If a refined fit, may use just core or accessory distances
    dir_prefix = network_dir + "/" + os.path.basename(network_dir)

    # load CUDA libraries
    if use_gpu and not gpu_lib:
        sys.stderr.write('Unable to load GPU libraries; exiting\n')
        sys.exit(1)

    if use_gpu:
        graph_suffix = '.csv.gz'
    else:
        graph_suffix = '.gt'

    if core_only and model.type == 'refine':
        model.slope = 0
        network_file = dir_prefix + '_core_graph' + graph_suffix
        cluster_file = dir_prefix + '_core_clusters.csv'
    elif accessory_only and model.type == 'refine':
        model.slope = 1
        network_file = dir_prefix + '_accessory_graph' + graph_suffix
        cluster_file = dir_prefix + '_accessory_clusters.csv'
    else:
        if ref_graph and os.path.isfile(dir_prefix + '.refs_graph' + graph_suffix):
            network_file = dir_prefix + '.refs_graph' + graph_suffix
        else:
            network_file = dir_prefix + '_graph' + graph_suffix
        cluster_file = dir_prefix + '_clusters.csv'
        if core_only or accessory_only:
            sys.stderr.write("Can only do --core-only or --accessory-only fits from "
                             "a refined fit. Using the combined distances.\n")

    # Load network file
    genomeNetwork = load_network_file(network_file, use_gpu = use_gpu)

    # Ensure all in dists are in final network
    checkNetworkVertexCount(refList, genomeNetwork, use_gpu)

    return genomeNetwork, cluster_file

def load_network_file(fn, use_gpu = False):
    """Load the network based on input options

       Returns the network as a graph-tool format graph, and sets
       the slope parameter of the passed model object.

       Args:
            fn (str)
                Network file name
            use_gpu (bool)
                Use cugraph library to load graph

       Returns:
            genomeNetwork (graph)
                The loaded network
    """
    # Load the network from the specified file
    if use_gpu:
        G_df = cudf.read_csv(fn, compression = 'gzip')
        genomeNetwork = cugraph.Graph()
        if 'weights' in G_df.columns:
            G_df.columns = ['source','destination','weights']
            genomeNetwork.from_cudf_edgelist(G_df, edge_attr='weights', renumber=False)
        else:
            G_df.columns = ['source','destination']
            genomeNetwork.from_cudf_edgelist(G_df,renumber=False)
        sys.stderr.write("Network loaded: " + str(genomeNetwork.number_of_vertices()) + " samples\n")
    else:
        genomeNetwork = gt.load_graph(fn)
        sys.stderr.write("Network loaded: " + str(len(list(genomeNetwork.vertices()))) + " samples\n")

    return genomeNetwork

def checkNetworkVertexCount(seq_list, G, use_gpu):
    """Checks the number of network vertices matches the number
    of sequence names.

    Args:
        seq_list (list)
            The list of sequence names
        G (graph)
            The network of sequences
        use_gpu (bool)
            Whether to use cugraph for graph analyses
    """
    vertex_list = set(get_vertex_list(G, use_gpu = use_gpu))
    networkMissing = set(set(range(len(seq_list))).difference(vertex_list))
    if len(networkMissing) > 0:
        sys.stderr.write("ERROR: " + str(len(networkMissing)) + " samples are missing from the final network\n")
        sys.exit(1)

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
    if gt.openmp_enabled():
        gt.openmp_set_num_threads(1)
    subgraph = gt.GraphView(graph, vfilt=components_list == component)
    refs = reference_indices.copy()
    if subgraph.num_vertices() <= 2:
        refs.add(subgraph.get_vertices()[0])
        ref_list = refs
    else:
        ref_list = getCliqueRefs(subgraph, refs)
    return(list(ref_list))

def extractReferences(G, dbOrder, outPrefix, type_isolate = None,
                        existingRefs = None, threads = 1, use_gpu = False):
    """Extract references for each cluster based on cliques

       Writes chosen references to file by calling :func:`~writeReferences`

       Args:
           G (graph)
               A network used to define clusters from :func:`~constructNetwork`
           dbOrder (list)
               The order of files in the sketches, so returned references are in the same order
           outPrefix (str)
               Prefix for output file (.refs will be appended)
           type_isolate (str)
               Isolate to be included in set of references
           existingRefs (list)
               References that should be used for each clique
           use_gpu (bool)
               Use cugraph for graph analysis (default = False)

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

    # Add type isolate, if necessary
    type_isolate_index = None
    if type_isolate is not None:
        if type_isolate in dbOrder:
            type_isolate_index = dbOrder.index(type_isolate)
        else:
            sys.stderr.write('Type isolate ' + type_isolate + ' not found\n')
            sys.exit(1)

    if use_gpu:
        if not gpu_lib:
            sys.stderr.write('Unable to load GPU libraries; exiting\n')
            sys.exit(1)

        # For large network, use more approximate method for extracting references
        reference = {}
        # Record the original components to which sequences belonged
        component_assignments = cugraph.components.connectivity.connected_components(G)
        # Leiden method has resolution parameter - higher values give greater precision
        partition_assignments, score = cugraph.leiden(G, resolution = 0.1)
        # group by partition, which becomes the first column, so retrieve second column
        reference_index_df = partition_assignments.groupby('partition').nth(0)
        reference_indices = reference_index_df['vertex'].to_arrow().to_pylist()

        # Add type isolate if necessary - before edges are added
        if type_isolate_index is not None and type_isolate_index not in reference_indices:
            reference_indices.append(type_isolate_index)

        # Order found references as in sketchlib database
        reference_names = [dbOrder[int(x)] for x in sorted(reference_indices)]
        refFileName = writeReferences(reference_names, outPrefix)

        # Extract reference edges
        G_df = G.view_edge_list()
        if 'src' in G_df.columns:
            G_df.rename(columns={'src': 'source','dst': 'destination'}, inplace=True)
        G_ref_df = G_df[G_df['source'].isin(reference_indices) & G_df['destination'].isin(reference_indices)]
        # Add self-loop if needed
        max_in_vertex_labels = max(reference_indices)
        G_ref = add_self_loop(G_ref_df,max_in_vertex_labels, renumber = False)

        # Check references in same component in overall graph are connected in the reference graph
        # First get components of original reference graph
        reference_component_assignments = cugraph.components.connectivity.connected_components(G_ref)
        reference_component_assignments.rename(columns={'labels': 'ref_labels'}, inplace=True)
        # Merge with component assignments from overall graph
        combined_vertex_assignments = reference_component_assignments.merge(component_assignments,
                                                                            on = 'vertex',
                                                                            how = 'left')
        combined_vertex_assignments = combined_vertex_assignments[combined_vertex_assignments['vertex'].isin(reference_indices)]
        # Find the number of components in the reference graph associated with each component in the overall graph -
        # should be one if there is a one-to-one mapping of components - else links need to be added
        max_ref_comp_count = combined_vertex_assignments.groupby(['labels'], sort = False)['ref_labels'].nunique().max()
        if max_ref_comp_count > 1:
            # Iterate through components
            for component, component_df in combined_vertex_assignments.groupby(['labels'], sort = False):
                # Find components in the overall graph matching multiple components in the reference graph
                if component_df.groupby(['labels'], sort = False)['ref_labels'].nunique().iloc[0] > 1:
                    # Make a graph of the component from the overall graph
                    vertices_in_component = component_assignments[component_assignments['labels']==component]['vertex']
                    references_in_component = vertices_in_component[vertices_in_component.isin(reference_indices)].values
                    G_component_df = G_df[G_df['source'].isin(vertices_in_component) & G_df['destination'].isin(vertices_in_component)]
                    G_component = cugraph.Graph()
                    G_component.from_cudf_edgelist(G_component_df)
                    # Find single shortest path from a reference to all other nodes in the component
                    traversal = cugraph.traversal.sssp(G_component,source = references_in_component[0])
                    reference_index_set = set(reference_indices)
                    # Add predecessors to reference sequences on the SSSPs
                    predecessor_list = traversal[traversal['vertex'].isin(reference_indices)]['predecessor'].values
                    predecessors = set(predecessor_list[predecessor_list >= 0].flatten().tolist())
                    # Add predecessors to reference set and check whether this results in complete paths
                    # where complete paths are indicated by references' predecessors being within the set of
                    # references
                    while len(predecessors) > 0 and len(predecessors - reference_index_set) > 0:
                        reference_index_set = reference_index_set.union(predecessors)
                        predecessor_list = traversal[traversal['vertex'].isin(reference_indices)]['predecessor'].values
                        predecessors = set(predecessor_list[predecessor_list >= 0].flatten().tolist())
                    # Add expanded reference set to the overall list
                    reference_indices = list(reference_index_set)
            # Create new reference graph
            G_ref_df = G_df[G_df['source'].isin(reference_indices) & G_df['destination'].isin(reference_indices)]
            G_ref = add_self_loop(G_ref_df, max_in_vertex_labels, renumber = False)

    else:
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

        # Add type isolate if necessary - before edges are added
        if type_isolate_index is not None and type_isolate_index not in reference_indices:
            reference_indices.add(type_isolate_index)

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

    # Order found references as in sketch files
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

def network_to_edges(prev_G_fn, rlist, previous_pkl = None, weights = False,
                    use_gpu = False):
    """Load previous network, extract the edges to match the
    vertex order specified in rlist, and also return weights if specified.

    Args:
        prev_G_fn (str)
            Path of file containing existing network.
        rlist (list)
            List of reference sequence labels in new network
        previous_pkl (str)
            Path of pkl file containing names of sequences in
            previous network
        weights (bool)
            Whether to return edge weights
            (default = False)
        use_gpu (bool)
            Whether to use cugraph for graph analyses

    Returns:
        source_ids (list)
            Source nodes for each edge
        target_ids (list)
            Target nodes for each edge
        edge_weights (list)
            Weights for each new edge
    """
    # get list for translating node IDs to rlist
    prev_G = load_network_file(prev_G_fn, use_gpu = use_gpu)

    # load list of names in previous network
    if previous_pkl is not None:
        with open(previous_pkl, 'rb') as pickle_file:
            old_rlist, old_qlist, self = pickle.load(pickle_file)
        if self:
            old_ids = old_rlist
        else:
            old_ids = old_rlist + old_qlist
    else:
        sys.stderr.write('Missing .pkl file containing names of sequences in '
                         'previous network\n')
        sys.exit(1)

    # Get edges as lists of source,destination,weight using original IDs
    if use_gpu:
        G_df = prev_G.view_edge_list()
        if weights:
            G_df.columns = ['source','destination','weight']
            edge_weights = G_df['weight'].to_arrow().to_pylist()
        else:
            G_df.columns = ['source','destination']
        old_source_ids = G_df['source'].to_arrow().to_pylist()
        old_target_ids = G_df['destination'].to_arrow().to_pylist()
    else:
        # get the source and target nodes
        old_source_ids = gt.edge_endpoint_property(prev_G, prev_G.vertex_index, "source")
        old_target_ids = gt.edge_endpoint_property(prev_G, prev_G.vertex_index, "target")
        # get the weights
        if weights:
            edge_weights = list(prev_G.ep['weight'])

    # Update IDs to new versions
    old_id_indices = [rlist.index(x) for x in old_ids]
    # translate to indices
    source_ids = [old_id_indices[x] for x in old_source_ids]
    target_ids = [old_id_indices[x] for x in old_target_ids]

    # return values
    if weights:
        return source_ids, target_ids, edge_weights
    else:
        return source_ids, target_ids

def constructNetwork(rlist, qlist, assignments, within_label,
                     summarise = True, edge_list = False, weights = None,
                     weights_type = 'euclidean', sparse_input = None,
                     previous_network = None, previous_pkl = None, use_gpu = False):
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
        weights_type (str)
            Specifies the type of weight to be annotated on the graph - options are core,
            accessory or euclidean distance
        sparse_input (numpy.array)
            Sparse distance matrix from lineage fit
        previous_network (str)
            Name of file containing a previous network to be integrated into this new
            network
        previous_pkl (str)
            Name of file containing the names of the sequences in the previous_network
        use_gpu (bool)
            Whether to use GPUs for network construction

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

    # Check weights type is valid
    if weights_type not in accepted_weights_types:
        sys.stderr.write("Unable to calculate distance type " + str(weights_type) + "; "
                         "accepted types are " + str(accepted_weights_types) + "\n")
        sys.exit(1)
    if edge_list and sparse_input:
        raise RuntimeError("Cannot construct network from edge list and sparse matrix")

    # identify edges
    connections = []
    if edge_list:
        if weights is not None:
            for weight, (ref, query) in zip(weights, assignments):
                connections.append((ref, query, weight))
        else:
            connections = assignments
    elif sparse_input is not None:
        for ref, query, weight in zip(sparse_input.row, sparse_input.col, sparse_input.data):
            connections.append((ref, query, weight))
    else:
        for row_idx, (assignment, (ref, query)) in enumerate(zip(assignments,
                                                                 listDistInts(rlist, qlist,
                                                                              self = self_comparison))):
            if assignment == within_label:
                if weights is not None:
                    if weights_type == 'euclidean':
                        dist = np.linalg.norm(weights[row_idx, :])
                    elif weights_type == 'core':
                        dist = weights[row_idx, 0]
                    elif weights_type == 'accessory':
                        dist = weights[row_idx, 1]
                    edge_tuple = (ref, query, dist)
                else:
                    edge_tuple = (ref, query)
                connections.append(edge_tuple)

    # read previous graph
    if previous_network is not None:
        if previous_pkl is not None:
            if weights is not None or sparse_input is not None:
                extra_sources, extra_targets, extra_weights = network_to_edges(previous_network,
                                                                                    rlist,
                                                                                    previous_pkl = previous_pkl,
                                                                                    weights = True,
                                                                                    use_gpu = use_gpu)
                for (ref, query, weight) in zip(extra_sources, extra_targets, extra_weights):
                    edge_tuple = (ref, query, weight)
                    connections.append(edge_tuple)
            else:
                extra_sources, extra_targets = network_to_edges(prev_G,
                                                                rlist,
                                                                previous_pkl = previous_pkl,
                                                                weights = False,
                                                                use_gpu = use_gpu)
                for (ref, query) in zip(extra_sources, extra_targets):
                    edge_tuple = (ref, query)
                    connections.append(edge_tuple)
        else:
            sys.stderr.write('A distance pkl corresponding to ' + previous_pkl + ' is required for loading\n')
            sys.exit(1)

    # load GPU libraries if necessary
    if use_gpu:

        if not gpu_lib:
           sys.stderr.write('Unable to load GPU libraries; exiting\n')
           sys.exit(1)

        # Set memory management for large networks
        cudf.set_allocator("managed")

        # create DataFrame using edge tuples
        if weights is not None or sparse_input is not None:
            G_df = cudf.DataFrame(connections, columns =['source', 'destination', 'weights'])
        else:
            G_df = cudf.DataFrame(connections, columns =['source', 'destination'])

        # ensure the highest-integer node is included in the edge list
        # by adding a self-loop if necessary; see https://github.com/rapidsai/cugraph/issues/1206
        max_in_df = np.amax([G_df['source'].max(),G_df['destination'].max()])
        max_in_vertex_labels = len(vertex_labels)-1
        use_weights = False
        if weights is not None:
            use_weights = True
        G = add_self_loop(G_df, max_in_vertex_labels, weights = use_weights, renumber = False)

    else:

        # build the graph
        G = gt.Graph(directed = False)
        G.add_vertex(len(vertex_labels))

        if weights is not None or sparse_input is not None:
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
        (metrics, scores) = networkSummary(G, use_gpu = use_gpu)
        sys.stderr.write("Network summary:\n" + "\n".join(["\tComponents\t\t\t\t" + str(metrics[0]),
                                                       "\tDensity\t\t\t\t\t" + "{:.4f}".format(metrics[1]),
                                                       "\tTransitivity\t\t\t\t" + "{:.4f}".format(metrics[2]),
                                                       "\tMean betweenness\t\t\t" + "{:.4f}".format(metrics[3]),
                                                       "\tWeighted-mean betweenness\t\t" + "{:.4f}".format(metrics[4]),
                                                       "\tScore\t\t\t\t\t" + "{:.4f}".format(scores[0]),
                                                       "\tScore (w/ betweenness)\t\t\t" + "{:.4f}".format(scores[1]),
                                                       "\tScore (w/ weighted-betweenness)\t\t" + "{:.4f}".format(scores[2])])
                                                       + "\n")

    return G

def networkSummary(G, calc_betweenness=True, use_gpu = False):
    """Provides summary values about the network

    Args:
        G (graph)
            The network of strains from :func:`~constructNetwork`
        calc_betweenness (bool)
            Whether to calculate betweenness stats
        use_gpu (bool)
            Whether to use cugraph for graph analysis

    Returns:
        metrics (list)
            List with # components, density, transitivity, mean betweenness
            and weighted mean betweenness
        scores (list)
            List of scores
    """
    if use_gpu:

        if not gpu_lib:
           sys.stderr.write('Unable to load GPU libraries; exiting\n')
           sys.exit(1)

        component_assignments = cugraph.components.connectivity.connected_components(G)
        component_nums = component_assignments['labels'].unique().astype(int)
        components = len(component_nums)
        density = G.number_of_edges()/(0.5 * G.number_of_vertices() * G.number_of_vertices() - 1)
        triangle_count = cugraph.community.triangle_count.triangles(G)
        degree_df = G.in_degree()
        triad_count = sum([d * (d - 1) for d in degree_df['degree'].to_pandas()])
        transitivity = 2 * triangle_count/triad_count
    else:
        component_assignments, component_frequencies = gt.label_components(G)
        components = len(component_frequencies)
        density = len(list(G.edges()))/(0.5 * len(list(G.vertices())) * (len(list(G.vertices())) - 1))
        transitivity = gt.global_clustering(G)[0]

    mean_bt = 0
    weighted_mean_bt = 0
    if calc_betweenness:
        betweenness = []
        sizes = []

        if use_gpu:
            component_frequencies = component_assignments['labels'].value_counts(sort = True, ascending = False)
            for component in component_nums.to_pandas():
                size = component_frequencies[component_frequencies.index == component].iloc[0].astype(int)
                if size > 3:
                    component_vertices = component_assignments['vertex'][component_assignments['labels']==component]
                    subgraph = cugraph.subgraph(G, component_vertices)
                    max_betweeness_k = 1000
                    if len(component_vertices) >= max_betweeness_k:
                        component_betweenness = cugraph.betweenness_centrality(subgraph, k = max_betweeness_k)
                    else:
                        component_betweenness = cugraph.betweenness_centrality(subgraph)
                    betweenness.append(component_betweenness['betweenness_centrality'].max())
                    sizes.append(size)
        else:
            for component, size in enumerate(component_frequencies):
                if size > 3:
                    vfilt = component_assignments.a == component
                    subgraph = gt.GraphView(G, vfilt=vfilt)
                    betweenness.append(max(gt.betweenness(subgraph, norm = True)[0].a))
                    sizes.append(size)

        if len(betweenness) > 1:
            mean_bt = np.mean(betweenness)
            weighted_mean_bt = np.average(betweenness, weights=sizes)
        elif len(betweenness) == 1:
            mean_bt = betweenness[0]
            weighted_mean_bt = betweenness[0]

    # Calculate scores
    metrics = [components, density, transitivity, mean_bt, weighted_mean_bt]
    base_score = transitivity * (1 - density)
    scores = [base_score, base_score * (1 - metrics[3]), base_score * (1 - metrics[4])]
    return(metrics, scores)

def addQueryToNetwork(dbFuncs, rList, qList, G, kmers,
                      assignments, model, queryDB, queryQuery = False,
                      strand_preserved = False, weights = None, threads = 1,
                      use_gpu = False):
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
        use_gpu (bool)
            Whether to use cugraph for analysis

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
        if len(qList) == 1:
            qqDistMat = np.zeros((0, 2), dtype=np.float32)
        else:
            sys.stderr.write("Calculating all query-query distances\n")
            addRandom(queryDB, qList, kmers, strand_preserved, threads = threads)
            qqDistMat = queryDatabase(rNames = qList,
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
            qqDistMat = queryDatabase(rNames = list(unassigned),
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
            for row_idx, (assignment, (query1, query2)) in enumerate(zip(queryAssignation, iterDistRows(qList, qList, self = True))):
                if assignment == model.within_label:
                    if weights is not None:
                        dist = np.linalg.norm(qqDistMat[row_idx, :])
                        edge_tuple = (query_indices[query1], query_indices[query2], dist)
                    else:
                        edge_tuple = (query_indices[query1], query_indices[query2])
                    new_edges.append(edge_tuple)

    # finish by updating the network
    if use_gpu:

        if not gpu_lib:
           sys.stderr.write('Unable to load GPU libraries; exiting\n')
           sys.exit(1)

        # construct updated graph
        G_current_df = G.view_edge_list()
        if weights is not None:
            G_current_df.columns = ['source','destination','weights']
            G_extra_df = cudf.DataFrame(new_edges, columns =['source','destination','weights'])
            G_df = cudf.concat([G_current_df,G_extra_df], ignore_index = True)
        else:
            G_current_df.columns = ['source','destination']
            G_extra_df = cudf.DataFrame(new_edges, columns =['source','destination'])
            G_df = cudf.concat([G_current_df,G_extra_df], ignore_index = True)

        # use self-loop to ensure all nodes are present
        max_in_vertex_labels = ref_count + len(qList) - 1
        include_weights = False
        if weights is not None:
            include_weights = True
        G = add_self_loop(G_df, max_in_vertex_labels, weights = include_weights)

    else:
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

    return G, qqDistMat

def add_self_loop(G_df, seq_num, weights = False, renumber = True):
    """Adds self-loop to cugraph graph to ensure all nodes are included in
    the graph, even if singletons.

    Args:
        G_df (cudf)
            cudf data frame containing edge list
        seq_num (int)
            The expected number of nodes in the graph
        renumber (bool)
            Whether to renumber the vertices when added to the graph

    Returns:
        G_new (graph)
            Dictionary of cluster assignments (keys are sequence names)
    """
    # use self-loop to ensure all nodes are present
    min_in_df = np.amin([G_df['source'].min(), G_df['destination'].min()])
    if min_in_df.item() > 0:
        G_self_loop = cudf.DataFrame()
        G_self_loop['source'] = [0]
        G_self_loop['destination'] = [0]
        if weights:
            G_self_loop['weights'] = 0.0
        G_df = cudf.concat([G_df,G_self_loop], ignore_index = True)
    max_in_df = np.amax([G_df['source'].max(),G_df['destination'].max()])
    if max_in_df.item() != seq_num:
        G_self_loop = cudf.DataFrame()
        G_self_loop['source'] = [seq_num]
        G_self_loop['destination'] = [seq_num]
        if weights:
            G_self_loop['weights'] = 0.0
        G_df = cudf.concat([G_df,G_self_loop], ignore_index = True)
    # Construct graph
    G_new = cugraph.Graph()
    G_new.from_cudf_edgelist(G_df, renumber = renumber)
    return G_new


def printClusters(G, rlist, outPrefix=None, oldClusterFile=None,
                  externalClusterCSV=None, printRef=True, printCSV=True,
                  clustering_type='combined', write_unwords=True,
                  use_gpu = False):
    """Get cluster assignments

    Also writes assignments to a CSV file

    Args:
        G (graph)
            Network used to define clusters (from :func:`~constructNetwork` or
            :func:`~addQueryToNetwork`)
        outPrefix (str)
            Prefix for output CSV
            Default = None
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
        write_unwords (bool)
            Write clusters with a pronouncable name rather than numerical index
            Default = True
        use_gpu (bool)
            Whether to use cugraph for network analysis
    Returns:
        clustering (dict)
            Dictionary of cluster assignments (keys are sequence names)

    """
    if oldClusterFile == None and printRef == False:
        raise RuntimeError("Trying to print query clusters with no query sequences")
    if write_unwords and not printCSV:
        write_unwords = False

    # get a sorted list of component assignments
    if use_gpu:
        if not gpu_lib:
           sys.stderr.write('Unable to load GPU libraries; exiting\n')
           sys.exit(1)

        component_assignments = cugraph.components.connectivity.connected_components(G)
        component_frequencies = component_assignments['labels'].value_counts(sort = True, ascending = False)
        newClusters = [set() for rank in range(component_frequencies.size)]
        for isolate_index, isolate_name in enumerate(rlist): # assume sorted at the moment
            component = component_assignments['labels'].iloc[isolate_index].item()
            component_rank_bool = component_frequencies.index == component
            component_rank = np.argmax(component_rank_bool.to_array())
            newClusters[component_rank].add(isolate_name)
    else:
        component_assignments, component_frequencies = gt.label_components(G)
        component_frequency_ranks = len(component_frequencies) - rankdata(component_frequencies, method = 'ordinal').astype(int)
        # use components to determine new clusters
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
    cluster_unword = {}
    if write_unwords:
        unword_generator = gen_unword()

    for newClsIdx, newCluster in enumerate(newClusters):
        needs_unword = False
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
                needs_unword = True
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
                            needs_unword = True
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
            needs_unword = True

        if write_unwords and needs_unword:
            unword = next(unword_generator)
        else:
            unword = None

        for cluster_member in newCluster:
            clustering[cluster_member] = cls_id
            if unword is not None:
                cluster_unword[cluster_member] = unword

    # print clustering to file
    if printCSV:
        outFileName = outPrefix + "_clusters.csv"
        with open(outFileName, 'w') as cluster_file:
            cluster_file.write("Taxon,Cluster\n")
            if write_unwords:
                unword_file = open(outPrefix + "_unword_clusters.csv", 'w')
                unword_file.write("Taxon,Cluster_name\n")

            # sort the clusters by frequency - define a list with a custom sort order
            # first line gives tuples e.g. (1, 28), (2, 17) - cluster 1 has 28 members, cluster 2 has 17 members
            # second line takes first element - the cluster IDs sorted by frequency
            freq_order = sorted(dict(Counter(clustering.values())).items(), key=operator.itemgetter(1), reverse=True)
            freq_order = [x[0] for x in freq_order]

            # iterate through cluster dictionary sorting by value using above custom sort order
            for cluster_member, cluster_name in sorted(clustering.items(), key=lambda i:freq_order.index(i[1])):
                if printRef or cluster_member not in oldNames:
                    cluster_file.write(",".join((cluster_member, str(cluster_name))) + "\n")
                if write_unwords and cluster_member in cluster_unword:
                    unword_file.write(",".join((cluster_member, cluster_unword[cluster_member])) + "\n")

            if write_unwords:
                unword_file.close()

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

def generate_minimum_spanning_tree(G, from_cugraph = False):
    """Generate a minimum spanning tree from a network

    Args:
       G (network)
           Graph tool network
       from_cugraph (bool)
            If a pre-calculated MST from cugraph
            [default = False]

    Returns:
       mst_network (str)
           Minimum spanning tree (as graph-tool graph)
    """
    #
    # Create MST
    #
    if not from_cugraph:
        sys.stderr.write("Starting calculation of minimum-spanning tree\n")

        # Test if weighted network and calculate minimum spanning tree
        if "weight" in G.edge_properties:
            mst_edge_prop_map = gt.min_spanning_tree(G, weights = G.ep["weight"])
            mst_network = gt.GraphView(G, efilt = mst_edge_prop_map)
            mst_network = gt.Graph(mst_network, prune = True)
        else:
            sys.stderr.write("generate_minimum_spanning_tree requires a weighted graph\n")
            raise RuntimeError("MST passed unweighted graph")
    else:
        mst_network = G

    # Find seed nodes as those with greatest outdegree in each component
    seed_vertices = set()
    component_assignments, component_frequencies = gt.label_components(mst_network)
    for component_index in range(len(component_frequencies)):
        component_members = component_assignments.a == component_index
        component = gt.GraphView(mst_network, vfilt = component_members)
        component_vertices = component.get_vertices()
        out_degrees = component.get_out_degrees(component_vertices)
        seed_vertex = list(component_vertices[np.where(out_degrees == np.amax(out_degrees))])
        seed_vertices.add(seed_vertex[0]) # Can only add one otherwise not MST

    # If multiple components, calculate distances between seed nodes
    if len(component_frequencies) > 1:
        # Extract distances
        connections = []
        max_weight = float(np.max(G.edge_properties["weight"].a))
        for ref in seed_vertices:
            seed_edges = G.get_all_edges(ref, [G.ep['weight']])
            found = False  # Not all edges may be in graph
            for seed_edge in seed_edges:
                if seed_edge[1] in seed_vertices:
                    found = True
                    connections.append((seed_edge))
            # TODO: alternative would be to requery the DB (likely quick)
            if found == False:
                for query in seed_vertices:
                    if query != ref:
                        connections.append((ref, query, max_weight))

        # Construct graph
        seed_G = gt.Graph(directed = False)
        seed_G.add_vertex(len(seed_vertex))
        eweight = seed_G.new_ep("float")
        seed_G.add_edge_list(connections, eprops = [eweight])
        seed_G.edge_properties["weight"] = eweight
        seed_mst_edge_prop_map = gt.min_spanning_tree(seed_G, weights = seed_G.ep["weight"])
        seed_mst_network = gt.GraphView(seed_G, efilt = seed_mst_edge_prop_map)
        # Insert seed MST into original MST - may be possible to use graph_union with include=True & intersection
        deep_edges = seed_mst_network.get_edges([seed_mst_network.ep["weight"]])
        mst_network.add_edge_list(deep_edges)

    sys.stderr.write("Completed calculation of minimum-spanning tree\n")
    return mst_network

def get_vertex_list(G, use_gpu = False):
    """Generate a list of node indices

    Args:
       G (network)
           Graph tool network
       use_gpu (bool)
            Whether graph is a cugraph or not
            [default = False]

    Returns:
       vlist (list)
           List of integers corresponding to nodes
    """

    if use_gpu:
        vlist = range(G.number_of_vertices())
    else:
        vlist = list(G.vertices())

    return vlist

def save_network(G, prefix = None, suffix = None, use_gpu = False):
    """Save a network to disc

    Args:
       G (network)
           Graph tool network
       prefix (str)
           Prefix for output file
       use_gpu (bool)
           Whether graph is a cugraph or not
           [default = False]

    """
    file_name = prefix + "/" + os.path.basename(prefix)
    if suffix is not None:
        file_name = file_name + suffix
    if use_gpu:
        G.to_pandas_edgelist().to_csv(file_name + '.csv.gz',
                compression='gzip', index = False)
    else:
        G.save(file_name + '.gt',
                fmt = 'gt')
