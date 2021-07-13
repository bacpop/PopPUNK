#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018-2020 John Lees and Nick Croucher

import os
import sys
import argparse
import pickle
import h5py
import pandas as pd
from scipy import sparse

# Load GPU libraries
try:
    import cupyx
    import cugraph
    import cudf
    import cupy
    from numba import cuda
    import rmm
    gpu_lib = True
except ImportError as e:
    gpu_lib = False

def setGtThreads(threads):
    import graph_tool.all as gt
    # Check on parallelisation of graph-tools
    if gt.openmp_enabled():
        gt.openmp_set_num_threads(threads)
        sys.stderr.write('\nGraph-tools OpenMP parallelisation enabled:')
        sys.stderr.write(' with ' + str(gt.openmp_get_num_threads()) + ' threads\n')

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


def read_rlist_from_distance_pickle(fn, allow_non_self = True):
    """Return the list of reference sequences from a distance pickle.

    Args:
        fn (str)
            Name of distance pickle
        allow_non_self (bool)
            Whether non-self distance datasets are permissible
    Returns:
        rlist (list)
            List of reference sequence names
    """
    with open(fn, 'rb') as pickle_file:
        rlist, qlist, self = pickle.load(pickle_file)
        if not allow_non_self and not self:
            sys.stderr.write("Thi analysis requires an all-v-all"
                             " distance dataset\n")
            sys.exit(1)
    return rlist

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

def check_and_set_gpu(use_gpu, gpu_lib, quit_on_fail = False):
    """Check GPU libraries can be loaded and set managed memory.

    Args:
        use_gpu (bool)
            Whether GPU packages have been requested
        gpu_lib (bool)
            Whether GPU packages are available
    Returns:
        use_gpu (bool)
            Whether GPU packages can be used
    """
    # load CUDA libraries
    if use_gpu and not gpu_lib:
        if quit_on_fail:
            sys.stderr.write('Unable to load GPU libraries; exiting\n')
            sys.exit(1)
        else:
            sys.stderr.write('Unable to load GPU libraries; using CPU libraries '
            'instead\n')
            use_gpu = False

    # Set memory management for large networks
    if use_gpu:
        rmm.reinitialize(managed_memory=True)
        cudf.set_allocator("managed")
        if "cupy" in sys.modules:
            cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)
        if "cuda" in sys.modules:
            cuda.set_memory_manager(rmm.RMMNumbaManager)
        assert(rmm.is_initialized())

    return use_gpu

def sparse_mat_to_network(sparse_mat, rlist, use_gpu = False):
    """Generate a network from a lineage rank fit

    Args:
       sparse_mat (scipy or cupyx sparse matrix)
         Sparse matrix of kNN from lineage fit
       rlist (list)
         List of sequence names
       use_gpu (bool)
         Whether GPU libraries should be used

    Returns:
      G (network)
          Graph tool or cugraph network
    """
    if use_gpu:
        G_df = cudf.DataFrame(columns = ['source','destination','weights'])
        G_df['source'] = sparse_mat.row
        G_df['destination'] = sparse_mat.col
        G_df['weights'] = sparse_mat.data
        max_in_vertex_labels = len(rlist)-1
        G = add_self_loop(G_df, max_in_vertex_labels, weights = True, renumber = False)
    else:
        connections = []
        for (src,dst) in zip(sparse_mat.row,sparse_mat.col):
            connections.append(src,dst)
        G = construct_network_from_edge_list(rlist,
                                               rlist,
                                               connections,
                                               weights = sparse_mat.data,
                                               summarise=False)

    return G

# command line parsing
def get_options():

    parser = argparse.ArgumentParser(description='Get information about a PopPUNK database',
                                     prog='poppunk_db_info')

    # input options
    parser.add_argument('--ref-db',
                        required = True,
                        help='PopPUNK database directory')
    parser.add_argument('--network',
                        required = True,
                        help='Network or lineage fit file for analysis')
    parser.add_argument('--distances',
                        default = None,
                        help='Prefix of distance files')
    parser.add_argument('--threads',
                        default = 1,
                        help='Number of cores to use in analysis')
    parser.add_argument('--use-gpu',
                        default = False,
                        action = 'store_true',
                        help='Whether GPU libraries should be used in analysis')
    parser.add_argument('--output',
                        required = True,
                        help='Prefix for output files')

    return parser.parse_args()

# main code
if __name__ == "__main__":

    # Check input ok
    args = get_options()

    # Check whether GPU libraries can be loaded
    use_gpu = check_and_set_gpu(args.use_gpu, gpu_lib, quit_on_fail = False)

    # Set threads for graph-tool
    setGtThreads(args.threads)

    # Open and process sequence database
    h5_fn = os.path.join(args.ref_db,os.path.basename(args.ref_db) + '.h5')
    ref_db = h5py.File(h5_fn, 'r')
    sample_names = list(ref_db['sketches'].keys())
    
    sample_sequence_length = {}
    sample_missing_bases = {}
    sample_base_frequencies = {name: [] for name in sample_names}
    
    for sample_name in sample_names:
        sample_base_frequencies[sample_name] = ref_db['sketches/' + sample_name].attrs['base_freq']
        sample_sequence_length[sample_name] = ref_db['sketches/' + sample_name].attrs['length']
        sample_missing_bases[sample_name] = ref_db['sketches/' + sample_name].attrs['missing_bases']
    
    # Process distance file
    distance_pkl = os.path.join(args.ref_db,os.path.basename(args.ref_db) + '.dists.pkl')
    if args.distances is not None:
        distance_pkl = args.distances + '.dists.pkl'
    rlist = read_rlist_from_distance_pickle(distance_pkl)
    
    # Open network file
    if args.network.endswith('.gt'):
        G = load_network_file(args.network, use_gpu = False)
    elif args.network.endswith('.csv.gz'):
        if use_gpu:
            G = load_network_file(args.network, use_gpu = True)
        else:
            sys.stderr.write('Unable to load necessary GPU libraries\n')
            exit(1)
    elif args.network.endswith('.npz'):
        sparse_mat = sparse.load_npz(args.network)
        G = sparse_mat_to_network(sparse_mat, sample_names, use_gpu = use_gpu)
    else:
        sys.stderr.write('Unrecognised suffix: expected ".gt", ".csv.gz" or ".npz"\n')
        exit(1)

    # Analyse network
    if use_gpu:
        component_assignments_df = cugraph.components.connectivity.connected_components(G)
        component_counts_df = component_assignments_df.groupby('labels')['labels'].count()
        component_counts_df.name = 'component_count'
        component_information_df = component_assignments_df.merge(component_counts_df, on = ['labels'], how = 'left')
        outdegree_df = G.out_degree()
        graph_properties_df = component_information_df.merge(outdegree_df, on = ['vertex'])
    else:
        graph_properties_df = pd.DataFrame()
        graph_properties_df['vertex'] = np.arange(len(rlist))
        graph_properties_df['labels'] = gt.label_components(G)[0].a
        graph_properties_df['degree'] = G.get_out_degrees(G.get_vertices())
        graph_properties_df['component_count'] = component_assignments.groupby('partition')['vertex'].transform('count')
    graph_properties_df['vertex'] = rlist
    
    # Merge data and print output
    with open(args.output,'w') as out_file:
        out_file.write('Sample,Length,Missing_bases,Frequency_A,Frequency_C,Frequency_G,Frequency_T,Component,Component_size,Node_degree\n')
        for i,sample_name in enumerate(sample_names):
            out_file.write(sample_name + ',' + str(sample_sequence_length[sample_name]) + ',' + str(sample_missing_bases[sample_name]) + ',')
            for frequency in sample_base_frequencies[sample_name]:
                out_file.write(str(frequency) + ',')
            graph_properties_row = graph_properties_df.iloc[graph_properties_df['vertex']==sample_name,:]
            out_file.write(str(graph_properties_row['labels'].values[0]) + ',')
            out_file.write(str(graph_properties_row['component_count'].values[0]) + ',')
            out_file.write(str(graph_properties_row['degree'].values[0]))
            out_file.write("\n")

    sys.exit(0)
