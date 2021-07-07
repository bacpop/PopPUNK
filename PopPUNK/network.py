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
import poppunk_refine

# Load GPU libraries
try:
    import cupyx
    import cugraph
    import cudf
    import cupy as cp
    from numba import cuda
    import rmm
    gpu_lib = True
except ImportError as e:
    gpu_lib = False

from .__main__ import accepted_weights_types
from .__main__ import betweenness_sample_default

from .sketchlib import addRandom

from .utils import iterDistRows
from .utils import listDistInts
from .utils import readIsolateTypeFromCsv
from .utils import readRfile
from .utils import setupDBFuncs
from .utils import isolateNameToLabel
from .utils import check_and_set_gpu

from .unwords import gen_unword

def fetchNetwork(network_dir, model, refList, ref_graph = False,
                  core_only = False, accessory_only = False, use_gpu = False):
    """Load the network based on input options

       Returns the network as a graph-tool format graph, and sets
       the slope parameter of the passed model object.

       Args:
            network_dir (str)
                A network used to define clusters
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

    # load CUDA libraries - here exit without switching to CPU libraries
    # to avoid loading an unexpected file
    use_gpu = check_and_set_gpu(use_gpu, gpu_lib, quit_on_fail = True)

    if use_gpu:
        graph_suffix = '.csv.gz'
    else:
        graph_suffix = '.gt'

    if core_only and model.type == 'refine':
        if ref_graph:
            network_file = dir_prefix + '_core.refs_graph' + graph_suffix
        else:
            network_file = dir_prefix + '_core_graph' + graph_suffix
        cluster_file = dir_prefix + '_core_clusters.csv'
    elif accessory_only and model.type == 'refine':
        if ref_graph:
            network_file = dir_prefix + '_accessory.refs_graph' + graph_suffix
        else:
            network_file = dir_prefix + '_accessory_graph' + graph_suffix
        cluster_file = dir_prefix + '_accessory_clusters.csv'
    else:
        if ref_graph and os.path.isfile(dir_prefix + '.refs_graph' + graph_suffix):
            network_file = dir_prefix + '.refs_graph' + graph_suffix
        else:
            network_file = dir_prefix + '_graph' + graph_suffix
        cluster_file = dir_prefix + '_clusters.csv'
        if core_only or accessory_only:
            sys.stderr.write("Can only do --core or --accessory fits from "
                             "a refined fit. Using the combined distances.\n")

    # Load network file
    sys.stderr.write("Loading network from " + network_file + "\n")
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

def translate_network_indices(G_ref_df, reference_indices):
    """Extract references for each cluster based on cliques

       Writes chosen references to file by calling :func:`~writeReferences`

       Args:
           G_ref_df (cudf data frame)
               List of edges in reference network
           reference_indices (list)
               The ordered list of reference indices in the original network

       Returns:
           G_ref (cugraph network)
               Network of reference sequences
    """
    # Translate network indices to match name order
    G_ref_df['source'] = [reference_indices.index(x) for x in G_ref_df['old_source'].to_arrow().to_pylist()]
    G_ref_df['destination'] = [reference_indices.index(x) for x in G_ref_df['old_destination'].to_arrow().to_pylist()]
    G_ref = add_self_loop(G_ref_df, len(reference_indices) - 1, renumber = True)
    return(G_ref)

def extractReferences(G, dbOrder, outPrefix, outSuffix = '', type_isolate = None,
                        existingRefs = None, threads = 1, use_gpu = False):
    """Extract references for each cluster based on cliques

       Writes chosen references to file by calling :func:`~writeReferences`

       Args:
           G (graph)
               A network used to define clusters
           dbOrder (list)
               The order of files in the sketches, so returned references are in the same order
           outPrefix (str)
               Prefix for output file
           outSuffix (str)
               Suffix for output file  (.refs will be appended)
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

        # Extract reference edges
        G_df = G.view_edge_list()
        if 'src' in G_df.columns:
            G_df.rename(columns={'src': 'old_source','dst': 'old_destination'}, inplace=True)
        else:
            G_df.rename(columns={'source': 'old_source','destination': 'old_destination'}, inplace=True)
        G_ref_df = G_df[G_df['old_source'].isin(reference_indices) & G_df['old_destination'].isin(reference_indices)]
        # Translate network indices to match name order
        G_ref = translate_network_indices(G_ref_df, reference_indices)

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
            G_ref_df = G_df[G_df['old_source'].isin(reference_indices) & G_df['old_destination'].isin(reference_indices)]
            G_ref = translate_network_indices(G_ref_df, reference_indices)

    else:
        # Each component is independent, so can be multithreaded
        components = gt.label_components(G)[0].a

        # Turn gt threading off and on again either side of the parallel loop
        if gt.openmp_enabled():
            gt.openmp_set_num_threads(1)

        # Cliques are pruned, taking one reference from each, until none remain
        sys.setrecursionlimit = 5000
        with Pool(processes=threads) as pool:
            ref_lists = pool.map(partial(cliquePrune,
                                            graph=G,
                                            reference_indices=reference_indices,
                                            components_list=components),
                                 set(components))
        sys.setrecursionlimit = 1000
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
    refFileName = writeReferences(reference_names, outPrefix, outSuffix = outSuffix)
    return reference_indices, reference_names, refFileName, G_ref

def writeReferences(refList, outPrefix, outSuffix = ""):
    """Writes chosen references to file

    Args:
        refList (list)
            Reference names to write
        outPrefix (str)
            Prefix for output file
        outSuffix (str)
            Suffix for output file (.refs will be appended)

    Returns:
        refFileName (str)
            The name of the file references were written to
    """
    # write references to file
    refFileName = outPrefix + "/" + os.path.basename(outPrefix) + outSuffix + ".refs"
    with open(refFileName, 'w') as rFile:
        for ref in refList:
            rFile.write(ref + '\n')
    return refFileName

def network_to_edges(prev_G_fn, rlist, adding_qq_dists = False,
                        old_ids = None, previous_pkl = None, weights = False,
                        use_gpu = False):
    """Load previous network, extract the edges to match the
    vertex order specified in rlist, and also return weights if specified.

    Args:
        prev_G_fn (str or graph object)
            Path of file containing existing network, or already-loaded
            graph object
        adding_qq_dists (bool)
            Boolean specifying whether query-query edges are being added
            to an existing network, such that not all the sequence IDs will
            be found in the old IDs, which should already be correctly ordered
        rlist (list)
            List of reference sequence labels in new network
        old_ids (list)
            List of IDs of vertices in existing network
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
    # Load graph from file if passed string; else use graph object passed in
    # as argument
    if isinstance(prev_G_fn, str):
        prev_G = load_network_file(prev_G_fn, use_gpu = use_gpu)
    else:
        prev_G = prev_G_fn

    # load list of names in previous network if pkl name supplied
    if previous_pkl is not None:
        with open(previous_pkl, 'rb') as pickle_file:
            old_rlist, old_qlist, self = pickle.load(pickle_file)
        if self:
            old_ids = old_rlist
        else:
            old_ids = old_rlist + old_qlist
    elif old_ids is None:
        sys.stderr.write('Missing .pkl file containing names of sequences in '
                         'previous network\n')
        sys.exit(1)

    # Get edges as lists of source,destination,weight using original IDs
    if use_gpu:
        G_df = prev_G.view_edge_list()
        if weights:
            if len(G_df.columns) < 3:
                sys.stderr.write('Loaded network does not have edge weights; try a different '
                                    'network or turn off graph weights\n')
                exit(1)
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
            if prev_G.edge_properties.keys() is None or 'weight' not in prev_G.edge_properties.keys():
                sys.stderr.write('Loaded network does not have edge weights; try a different '
                                    'network or turn off graph weights\n')
                exit(1)
            edge_weights = list(prev_G.ep['weight'])

    # If appending queries to an existing network, then the recovered links can be left
    # unchanged, as the new IDs are the queries, and the existing sequences will not be found
    # in the list of IDs
    if adding_qq_dists:
        source_ids = old_source_ids
        target_ids = old_target_ids
    else:
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

def print_network_summary(G, betweenness_sample = betweenness_sample_default, use_gpu = False):
    """Wrapper function for printing network information

    Args:
        G (graph)
            List of reference sequence labels
        betweenness_sample (int)
            Number of sequences per component used to estimate betweenness using
            a GPU. Smaller numbers are faster but less precise [default = 100]
        use_gpu (bool)
            Whether to use GPUs for network construction
    """
    # print some summaries
    (metrics, scores) = networkSummary(G, betweenness_sample = betweenness_sample, use_gpu = use_gpu)
    sys.stderr.write("Network summary:\n" + "\n".join(["\tComponents\t\t\t\t" + str(metrics[0]),
                                                   "\tDensity\t\t\t\t\t" + "{:.4f}".format(metrics[1]),
                                                   "\tTransitivity\t\t\t\t" + "{:.4f}".format(metrics[2]),
                                                   "\tMean betweenness\t\t\t" + "{:.4f}".format(metrics[3]),
                                                   "\tWeighted-mean betweenness\t\t" + "{:.4f}".format(metrics[4]),
                                                   "\tScore\t\t\t\t\t" + "{:.4f}".format(scores[0]),
                                                   "\tScore (w/ betweenness)\t\t\t" + "{:.4f}".format(scores[1]),
                                                   "\tScore (w/ weighted-betweenness)\t\t" + "{:.4f}".format(scores[2])])
                                                   + "\n")

def initial_graph_properties(rlist, qlist):
    """Initial processing of sequence names for
    network construction.

    Args:
        rlist (list)
            List of reference sequence labels
        qlist (list)
            List of query sequence labels

    Returns:
        vertex_labels (list)
            Ordered list of sequences in network
        self_comparison (bool)
            Whether the network is being constructed from all-v-all distances or
            reference-v-query information
    """
    if rlist == qlist:
        self_comparison = True
        vertex_labels = rlist
    else:
        self_comparison = False
        vertex_labels = rlist +  qlist
    return vertex_labels, self_comparison

def process_weights(distMat, weights_type):
    """Calculate edge weights from the distance matrix
    Args:
        distMat (2 column ndarray)
            Numpy array of pairwise distances
        weights_type (str)
            Measure to calculate from the distMat to use as edge weights in network
            - options are core, accessory or euclidean distance

    Returns:
        processed_weights (list)
            Edge weights
    """
    processed_weights = []
    if weights_type is not None and distMat is not None:
        # Check weights type is valid
        if weights_type not in accepted_weights_types:
            sys.stderr.write("Unable to calculate distance type " + str(weights_type) + "; "
                             "accepted types are " + str(accepted_weights_types) + "\n")
        if weights_type == 'euclidean':
            processed_weights = np.linalg.norm(distMat, axis = 1).tolist()
        elif weights_type == 'core':
            processed_weights = distMat[:, 0].tolist()
        elif weights_type == 'accessory':
            processed_weights = distMat[:, 1].tolist()
    else:
        sys.stderr.write('Require distance matrix to calculate distances\n')
    return processed_weights
    
def process_previous_network(previous_network = None, adding_qq_dists = False, old_ids = None,
                                previous_pkl = None, vertex_labels = None, weights = False, use_gpu = False):
    """Extract edge types from an existing network

    Args:
        previous_network (str or graph object)
            Name of file containing a previous network to be integrated into this new
            network, or already-loaded graph object
        adding_qq_dists (bool)
            Boolean specifying whether query-query edges are being added
            to an existing network, such that not all the sequence IDs will
            be found in the old IDs, which should already be correctly ordered
        old_ids (list)
            Ordered list of vertex names in previous network
        previous_pkl (str)
            Name of file containing the names of the sequences in the previous_network
            ordered based on the original network construction
        vertex_labels (list)
            Ordered list of sequence labels
        weights (bool)
            Whether weights should be extracted from the previous network
        use_gpu (bool)
            Whether to use GPUs for network construction

    Returns:
        extra_sources (list)
            List of source node identifiers
        extra_targets (list)
            List of destination node identifiers
        extra_weights (list or None)
            List of edge weights
    """
    if previous_pkl is not None or old_ids is not None:
        if weights:
            # Extract from network
            extra_sources, extra_targets, extra_weights = network_to_edges(previous_network,
                                                                            vertex_labels,
                                                                            adding_qq_dists = adding_qq_dists,
                                                                            old_ids = old_ids,
                                                                            previous_pkl = previous_pkl,
                                                                            weights = True,
                                                                            use_gpu = use_gpu)
        else:
            # Extract from network
            extra_sources, extra_targets = network_to_edges(previous_network,
                                                            vertex_labels,
                                                            adding_qq_dists = adding_qq_dists,
                                                            old_ids = old_ids,
                                                            previous_pkl = previous_pkl,
                                                            weights = False,
                                                            use_gpu = use_gpu)
            extra_weights = None
    else:
        sys.stderr.write('A distance pkl corresponding to ' + previous_pkl + ' is required for loading\n')
        sys.exit(1)
    
    return extra_sources, extra_targets, extra_weights

def construct_network_from_edge_list(rlist, qlist, edge_list,
    weights = None, distMat = None, previous_network = None, adding_qq_dists = False,
    old_ids = None, previous_pkl = None, betweenness_sample = betweenness_sample_default,
    summarise = True, use_gpu = False):
    """Construct an undirected network using a data frame of edges. Nodes are samples and
    edges where samples are within the same cluster

    Will print summary statistics about the network to ``STDERR``

    Args:
        rlist (list)
            List of reference sequence labels
        qlist (list)
            List of query sequence labels
        G_df (cudf or pandas data frame)
            Data frame in which the first two columns are the nodes linked by edges
        weights (list)
            List of edge weights
        distMat (2 column ndarray)
            Numpy array of pairwise distances
        previous_network (str or graph object)
            Name of file containing a previous network to be integrated into this new
            network, or the already-loaded graph object
        adding_qq_dists (bool)
            Boolean specifying whether query-query edges are being added
            to an existing network, such that not all the sequence IDs will
            be found in the old IDs, which should already be correctly ordered
        old_ids (list)
            Ordered list of vertex names in previous network
        previous_pkl (str)
            Name of file containing the names of the sequences in the previous_network
        betweenness_sample (int)
            Number of sequences per component used to estimate betweenness using
            a GPU. Smaller numbers are faster but less precise [default = 100]
        summarise (bool)
            Whether to calculate and print network summaries with :func:`~networkSummary`
            (default = True)
        use_gpu (bool)
            Whether to use GPUs for network construction

    Returns:
        G (graph)
            The resulting network
    """
    
    # Check GPU library use
    use_gpu = check_and_set_gpu(use_gpu, gpu_lib, quit_on_fail = True)
    
    # data structures
    vertex_labels, self_comparison = initial_graph_properties(rlist, qlist)

    # Create new network
    if use_gpu:
        # benchmarking concurs with https://stackoverflow.com/questions/55922162/recommended-cudf-dataframe-construction
        if len(edge_list) > 1:
            edge_array = cp.array(edge_list, dtype = np.int32)
            edge_gpu_matrix = cuda.to_device(edge_array)
            G_df = cudf.DataFrame(edge_gpu_matrix, columns = ['source','destination'])
        else:
            # Cannot generate an array when one edge
            G_df = cudf.DataFrame(columns = ['source','destination'])
            G_df['source'] = [edge_list[0][0]]
            G_df['destination'] = [edge_list[0][1]]
        if weights is not None:
            G_df['weights'] = weights
        G = construct_network_from_df(rlist, qlist, G_df,
                                        weights = (weights is not None),
                                        distMat = distMat,
                                        adding_qq_dists = adding_qq_dists,
                                        old_ids = old_ids,
                                        previous_network = previous_network,
                                        previous_pkl = previous_pkl,
                                        summarise = False,
                                        use_gpu = use_gpu)
    else:
        # Load previous network
        if previous_network is not None:
            extra_sources, extra_targets, extra_weights = \
                process_previous_network(previous_network = previous_network,
                                            adding_qq_dists = adding_qq_dists,
                                            old_ids = old_ids,
                                            previous_pkl = previous_pkl,
                                            vertex_labels = vertex_labels,
                                            weights = (weights is not None),
                                            use_gpu = use_gpu)
        # Construct list of tuples for graph-tool
        # Include information from previous graph if supplied
        weighted_edges = []
        if weights is not None:
            for ((src, dest), weight) in zip(edge_list, weights):
                edge_list.append((src, dest, weight))
            if previous_network is not None:
                for (src, dest, weight) in zip(extra_sources, extra_targets, extra_weights):
                    weighted_edges.append((src, dest, weight))
        else:
            if previous_network is not None:
                for (src, dest) in zip(extra_sources, extra_targets):
                    weighted_edges.append((src, dest))
        edge_list = weighted_edges
        # build the graph
        G = gt.Graph(directed = False)
        G.add_vertex(len(vertex_labels))
        if weights is not None:
            eweight = G.new_ep("float")
            G.add_edge_list(edge_list, eprops = [eweight])
            G.edge_properties["weight"] = eweight
        else:
            G.add_edge_list(edge_list)
    if summarise:
        print_network_summary(G, betweenness_sample = betweenness_sample, use_gpu = use_gpu)

    return G

def construct_network_from_df(rlist, qlist, G_df,
    weights = False, distMat = None, previous_network = None, adding_qq_dists = False,
    old_ids = None, previous_pkl = None, betweenness_sample = betweenness_sample_default,
    summarise = True, use_gpu = False):
    """Construct an undirected network using a data frame of edges. Nodes are samples and
    edges where samples are within the same cluster

    Will print summary statistics about the network to ``STDERR``

    Args:
        rlist (list)
            List of reference sequence labels
        qlist (list)
            List of query sequence labels
        G_df (cudf or pandas data frame)
            Data frame in which the first two columns are the nodes linked by edges
        weights (bool)
            Whether weights in the G_df data frame should be included in the network
        distMat (2 column ndarray)
            Numpy array of pairwise distances
        previous_network (str or graph object)
            Name of file containing a previous network to be integrated into this new
            network, or the already-loaded graph object
        adding_qq_dists (bool)
            Boolean specifying whether query-query edges are being added
            to an existing network, such that not all the sequence IDs will
            be found in the old IDs, which should already be correctly ordered
        old_ids (list)
            Ordered list of vertex names in previous network
        previous_pkl (str)
            Name of file containing the names of the sequences in the previous_network
        betweenness_sample (int)
            Number of sequences per component used to estimate betweenness using
            a GPU. Smaller numbers are faster but less precise [default = 100]
        summarise (bool)
            Whether to calculate and print network summaries with :func:`~networkSummary`
            (default = True)
        use_gpu (bool)
            Whether to use GPUs for network construction

    Returns:
        G (graph)
            The resulting network
    """
    
    # Check GPU library use
    use_gpu = check_and_set_gpu(use_gpu, gpu_lib, quit_on_fail = True)
    
    # data structures
    vertex_labels, self_comparison = initial_graph_properties(rlist, qlist)

    # Check df format is correct
    if weights:
        G_df.columns = ['source','destination','weights']
    else:
        G_df.columns = ['source','destination']

    # Load previous network
    if previous_network is not None:
        extra_sources, extra_targets, extra_weights = process_previous_network(previous_network = previous_network,
                                                                                adding_qq_dists = adding_qq_dists,
                                                                                old_ids = old_ids,
                                                                                previous_pkl = previous_pkl,
                                                                                vertex_labels = vertex_labels,
                                                                                weights = weights,
                                                                                use_gpu = use_gpu)
        if use_gpu:
            G_extra_df = cudf.DataFrame()
        else:
            G_extra_df = pd.DataFrame()
        G_extra_df['source'] = extra_sources
        G_extra_df['destination'] = extra_targets
        if extra_weights is not None:
            G_extra_df['weights'] = extra_weights
        G_df = cudf.concat([G_df,G_extra_df], ignore_index = True)

    if use_gpu:
        # direct conversion
        # ensure the highest-integer node is included in the edge list
        # by adding a self-loop if necessary; see https://github.com/rapidsai/cugraph/issues/1206
        max_in_df = np.amax([G_df['source'].max(),G_df['destination'].max()])
        max_in_vertex_labels = len(vertex_labels)-1
        use_weights = False
        if weights:
            use_weights = True
        G = add_self_loop(G_df, max_in_vertex_labels, weights = use_weights, renumber = False)
    else:
        # Convert bool to list of weights or None
        if weights:
            weights = G_df['weights']
        else:
            weights = None
        # Convert data frame to list of tuples
        connections = list(zip(*[G_df[c].values.tolist() for c in G_df[['source','destination']]]))
        G = construct_network_from_edge_list(rlist, qlist, connections,
                                            weights = weights,
                                            distMat = distMat,
                                            previous_network = previous_network,
                                            old_ids = old_ids,
                                            previous_pkl = previous_pkl,
                                            summarise = False,
                                            use_gpu = use_gpu)
    if summarise:
        print_network_summary(G, betweenness_sample = betweenness_sample, use_gpu = use_gpu)
    return G

def construct_network_from_sparse_matrix(rlist, qlist, sparse_input,
    weights = None, previous_network = None, previous_pkl = None,
    betweenness_sample = betweenness_sample_default, summarise = True, use_gpu = False):
    """Construct an undirected network using a sparse matrix. Nodes are samples and
    edges where samples are within the same cluster

    Will print summary statistics about the network to ``STDERR``

    Args:
        rlist (list)
            List of reference sequence labels
        qlist (list)
            List of query sequence labels
        sparse_input (numpy.array)
            Sparse distance matrix from lineage fit
        weights (list)
            List of weights for each edge in the network
        distMat (2 column ndarray)
            Numpy array of pairwise distances
        previous_network (str)
            Name of file containing a previous network to be integrated into this new
            network
        previous_pkl (str)
            Name of file containing the names of the sequences in the previous_network
        betweenness_sample (int)
            Number of sequences per component used to estimate betweenness using
            a GPU. Smaller numbers are faster but less precise [default = 100]
        summarise (bool)
            Whether to calculate and print network summaries with :func:`~networkSummary`
            (default = True)
        use_gpu (bool)
            Whether to use GPUs for network construction

    Returns:
        G (graph)
            The resulting network
    """
    
    # Check GPU library use
    use_gpu = check_and_set_gpu(use_gpu, gpu_lib, quit_on_fail = True)
    
    if use_gpu:
        G_df = cudf.DataFrame()
    else:
        G_df = pd.DataFrame()
    G_df['source'] = sparse_input.row
    G_df['destination'] =  sparse_input.col
    G_df['weights'] = sparse_input.data
    G = construct_network_from_df(rlist, qlist, G_df,
                                    weights = True,
                                    previous_network = previous_network,
                                    previous_pkl = previous_pkl,
                                    betweenness_sample = betweenness_sample,
                                    summarise = False,
                                    use_gpu = use_gpu)
    if summarise:
        print_network_summary(G, betweenness_sample = betweenness_sample, use_gpu = use_gpu)
    return G

def construct_network_from_assignments(rlist, qlist, assignments, within_label = 1, int_offset = 0,
    weights = None, distMat = None, weights_type = None, previous_network = None, old_ids = None,
    adding_qq_dists = False, previous_pkl = None, betweenness_sample = betweenness_sample_default,
    summarise = True, use_gpu = False):
    """Construct an undirected network using sequence lists, assignments of pairwise distances
    to clusters, and the identifier of the cluster assigned to within-strain distances.
    Nodes are samples and edges where samples are within the same cluster

    Will print summary statistics about the network to ``STDERR``

    Args:
        rlist (list)
            List of reference sequence labels
        qlist (list)
            List of query sequence labels
        assignments (numpy.array or int)
            Labels of most likely cluster assignment
        within_label (int)
            The label for the cluster representing within-strain distances
        int_offset (int)
            Constant integer to add to each node index
        weights (list)
            List of weights for each edge in the network
        distMat (2 column ndarray)
            Numpy array of pairwise distances
        weights_type (str)
            Measure to calculate from the distMat to use as edge weights in network
            - options are core, accessory or euclidean distance
        previous_network (str)
            Name of file containing a previous network to be integrated into this new
            network
        old_ids (list)
            Ordered list of vertex names in previous network
        adding_qq_dists (bool)
            Boolean specifying whether query-query edges are being added
            to an existing network, such that not all the sequence IDs will
            be found in the old IDs, which should already be correctly ordered
        previous_pkl (str)
            Name of file containing the names of the sequences in the previous_network
        betweenness_sample (int)
            Number of sequences per component used to estimate betweenness using
            a GPU. Smaller numbers are faster but less precise [default = 100]
        summarise (bool)
            Whether to calculate and print network summaries with :func:`~networkSummary`
            (default = True)
        use_gpu (bool)
            Whether to use GPUs for network construction

    Returns:
        G (graph)
            The resulting network
    """
    
    # Check GPU library use
    use_gpu = check_and_set_gpu(use_gpu, gpu_lib, quit_on_fail = True)

    # Filter weights to only the relevant edges
    if weights is not None:
        print("Weights: " + str(weights))
        print("Assignments: " + str(assignments))
        weights = weights[assignments == within_label]
    elif distMat is not None and weights_type is not None:
        if isinstance(assignments, list):
            assignments = np.array(assignments)
        distMat = distMat[assignments == within_label,:]
        weights = process_weights(distMat, weights_type)

    # Convert edge indices to tuples
    connections = poppunk_refine.generateTuples(assignments,
                                                within_label,
                                                self = (rlist == qlist),
                                                num_ref = len(rlist),
                                                int_offset = int_offset)

    # Construct network using edge list
    G = construct_network_from_edge_list(rlist, qlist, connections,
                                            weights = weights,
                                            distMat = distMat,
                                            previous_network = previous_network,
                                            adding_qq_dists = adding_qq_dists,
                                            old_ids = old_ids,
                                            previous_pkl = previous_pkl,
                                            summarise = False,
                                            use_gpu = use_gpu)
    if summarise:
        print_network_summary(G, betweenness_sample = betweenness_sample, use_gpu = use_gpu)

    return G

def get_cugraph_triangles(G):
    """Counts the number of triangles in a cugraph
    network. Can be removed when the cugraph issue
    https://github.com/rapidsai/cugraph/issues/1043 is fixed.

    Args:
        G (cugraph network)
            Network to be analysed
    
    Returns:
        triangle_count (int)
            Count of triangles in graph
    """
    nlen = G.number_of_vertices()
    df = G.view_edge_list()
    A = cp.full((nlen, nlen), 0, dtype = cp.int32)
    A[df.src.values, df.dst.values] = 1
    A = cp.maximum( A, A.transpose() )
    triangle_count = int(cp.around(cp.trace(cp.matmul(A, cp.matmul(A, A)))/6,0))
    return triangle_count

def networkSummary(G, calc_betweenness=True, betweenness_sample = betweenness_sample_default,
                    use_gpu = False):
    """Provides summary values about the network

    Args:
        G (graph)
            The network of strains
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

        use_gpu = check_and_set_gpu(use_gpu, gpu_lib, quit_on_fail = True)

        component_assignments = cugraph.components.connectivity.connected_components(G)
        component_nums = component_assignments['labels'].unique().astype(int)
        components = len(component_nums)
        density = G.number_of_edges()/(0.5 * G.number_of_vertices() * G.number_of_vertices() - 1)
        # consistent with graph-tool for small graphs - triangle counts differ for large graphs
        # could reflect issue https://github.com/rapidsai/cugraph/issues/1043
        # this command can be restored once the above issue is fixed - scheduled for cugraph 0.20
#        triangle_count = cugraph.community.triangle_count.triangles(G)/3
        triangle_count = 3*get_cugraph_triangles(G)
        degree_df = G.in_degree()
        # consistent with graph-tool
        triad_count = 0.5 * sum([d * (d - 1) for d in degree_df[degree_df['degree'] > 1]['degree'].to_pandas()])
        if triad_count > 0:
            transitivity = triangle_count/triad_count
        else:
            transitivity = 0.0
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
                    if len(component_vertices) >= betweenness_sample:
                        component_betweenness = cugraph.betweenness_centrality(subgraph,
                                                                                k = betweenness_sample,
                                                                                normalized = True)
                    else:
                        component_betweenness = cugraph.betweenness_centrality(subgraph,
                                                                                normalized = True)
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
                      assignments, model, queryDB, distances = None, distance_type = 'euclidean',
                      queryQuery = False, strand_preserved = False, weights = None, threads = 1,
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
        distances (str)
            Prefix of distance files for extending network
        distance_type (str)
            Distance type to use as weights in network
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
    
    # do not calculate weights unless specified
    if weights is None:
        distance_type = None

    # initialise links data structure
    new_edges = []
    assigned = set()

    # These are returned
    qqDistMat = None

    # store links for each query in a list of edge tuples
    ref_count = len(rList)

    # Add queries to network
    G = construct_network_from_assignments(rList,
                                            qList,
                                            assignments,
                                            within_label = model.within_label,
                                            previous_network = G,
                                            old_ids = rList,
                                            distMat = weights,
                                            weights_type = distance_type,
                                            summarise = False,
                                            use_gpu = use_gpu)

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

            if distance_type == 'core':
                queryAssignation = model.assign(qqDistMat, slope = 0)
            elif distance_type == 'accessory':
                queryAssignation = model.assign(qqDistMat, slope = 1)
            else:
                queryAssignation = model.assign(qqDistMat)

            # Add queries to network
            G = construct_network_from_assignments(qList,
                                                    qList,
                                                    queryAssignation,
                                                    int_offset = ref_count,
                                                    within_label = model.within_label,
                                                    previous_network = G,
                                                    old_ids = rList,
                                                    adding_qq_dists = True,
                                                    distMat = qqDistMat,
                                                    weights_type = distance_type,
                                                    summarise = False,
                                                    use_gpu = use_gpu)

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
                                      
            if distance_type == 'core':
                queryAssignation = model.assign(qqDistMat, slope = 0)
            elif distance_type == 'accessory':
                queryAssignation = model.assign(qqDistMat, slope = 1)
            else:
                queryAssignation = model.assign(qqDistMat)

            # identify any links between queries and store in the same links dict
            # links dict now contains lists of links both to original database and new queries
            # have to use names and link to query list in order to match to node indices
            for row_idx, (assignment, (query1, query2)) in enumerate(zip(queryAssignation, iterDistRows(qList, qList, self = True))):
                if assignment == model.within_label:
                    if weights is not None:
                        if distance_type == 'core':
                            dist = weights[row_idx, 0]
                        elif distance_type == 'accessory':
                            dist = weights[row_idx, 1]
                        else:
                            dist = np.linalg.norm(weights[row_idx, :])
                        edge_tuple = (query_indices[query1], query_indices[query2], dist)
                    else:
                        edge_tuple = (query_indices[query1], query_indices[query2])
                    new_edges.append(edge_tuple)
            
            G = construct_network_from_assignments(qList,
                                                    qList,
                                                    queryAssignation,
                                                    int_offset = ref_count,
                                                    within_label = model.within_label,
                                                    previous_network = G,
                                                    old_ids = rList + qList,
                                                    adding_qq_dists = True,
                                                    distMat = qqDistMat,
                                                    weights_type = distance_type,
                                                    summarise = False,
                                                    use_gpu = use_gpu)

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
    if weights:
        G_new.from_cudf_edgelist(G_df, edge_attr = 'weights', renumber = renumber)
    else:
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
            Network used to define clusters
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
        use_gpu = check_and_set_gpu(use_gpu, gpu_lib, quit_on_fail = True)
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

    if "sample" not in d:
        sys.stderr.write("WARNING: No new samples found, cannot write external clusters\n")
    else:
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
    if from_cugraph:
        mst_network = G
    else:
        sys.stderr.write("Starting calculation of minimum-spanning tree\n")

        # Test if weighted network and calculate minimum spanning tree
        if "weight" in G.edge_properties:
            mst_edge_prop_map = gt.min_spanning_tree(G, weights = G.ep["weight"])
            mst_network = gt.GraphView(G, efilt = mst_edge_prop_map)
            mst_network = gt.Graph(mst_network, prune = True)
        else:
            sys.stderr.write("generate_minimum_spanning_tree requires a weighted graph\n")
            raise RuntimeError("MST passed unweighted graph")

    # Find seed nodes as those with greatest outdegree in each component
    num_components = 1
    seed_vertices = set()
    if from_cugraph:
        mst_df = cugraph.components.connectivity.connected_components(mst_network)
        num_components_idx = mst_df['labels'].max()
        num_components = mst_df.iloc[num_components_idx]['labels']
        if num_components > 1:
            mst_df['degree'] = mst_network.in_degree()['degree']
            # idxmax only returns first occurrence of maximum so should maintain
            # MST - check cuDF implementation is the same
            max_indices = mst_df.groupby(['labels'])['degree'].idxmax()
            seed_vertices = mst_df.iloc[max_indices]['vertex']
            num_components = seed_vertices.size()
    else:
        component_assignments, component_frequencies = gt.label_components(mst_network)
        num_components = len(component_frequencies)
        if num_components > 1:
            for component_index in range(len(component_frequencies)):
                component_members = component_assignments.a == component_index
                component = gt.GraphView(mst_network, vfilt = component_members)
                component_vertices = component.get_vertices()
                out_degrees = component.get_out_degrees(component_vertices)
                seed_vertex = list(component_vertices[np.where(out_degrees == np.amax(out_degrees))])
                seed_vertices.add(seed_vertex[0]) # Can only add one otherwise not MST
            

    # If multiple components, add distances between seed nodes
    if num_components > 1:
        
        # Extract edges and maximum edge length - as DF for cugraph
        # list of tuples for graph-tool
        if from_cugraph:
            # With cugraph the MST is already calculated
            # so no extra edges can be retrieved from the graph
            G_df = G.view_edge_list()
            max_weight = G_df['weights'].max()
            first_seed = seed_vertices[0]
            G_seed_link_df = cudf.DataFrame()
            G_seed_link_df['dst'] = seed_vertices.iloc[1:seed_vertices.size()]
            G_seed_link_df['src'] = seed_vertices.iloc[0]
            G_seed_link_df['weights'] = seed_vertices.iloc[0]
            G_df = G_df.append(G_seed_link_df)
        else:
            # With graph-tool look to retrieve edges in larger graph
            connections = []
            max_weight = float(np.max(G.edge_properties["weight"].a))

            # Identify edges between seeds to link components together
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
        if from_cugraph:
            mst_network = G_df.from_cudf_edgelist(edge_attr='weights', renumber=False)
        else:
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

def save_network(G, prefix = None, suffix = None, use_graphml = False,
                use_gpu = False):
    """Save a network to disk

    Args:
       G (network)
           Graph tool network
       prefix (str)
           Prefix for output file
       use_graphml (bool)
           Whether to output a graph-tool file
           in graphml format
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
        if use_graphml:
            G.save(file_name + '.graphml',
                    fmt = 'graphml')
        else:
            G.save(file_name + '.gt',
                    fmt = 'gt')

def cugraph_to_graph_tool(G, rlist):
    """Save a network to disk

    Args:
       G (cugraph network)
         Cugraph network
       rlist (list)
         List of sequence names
           
    Returns:
      G (graph-tool network)
          Graph tool network
    """
    edge_df = G.view_edge_list()
    edge_tuple = edge_df[['src', 'dst']].values.tolist()
    edge_weights = None
    if 'weights' in edge_df.columns:
        edge_weights = edge_df['weights'].values_host
    G = construct_network_from_edge_list(rlist, rlist,
                                           edge_tuple,
                                           weights = edge_weights,
                                           summarise=False)
    vid = G.new_vertex_property('string',
                                vals = rlist)
    G.vp.id = vid
    return G
