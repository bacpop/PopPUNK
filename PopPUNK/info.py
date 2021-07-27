#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018-2020 John Lees and Nick Croucher

# universal
import os
import sys
import pickle
# additional
import h5py
import argparse
import numpy as np
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

    # Import functions
    from .network import add_self_loop
    from .network import load_network_file
    from .network import sparse_mat_to_network
    from .utils import check_and_set_gpu
    from .utils import setGtThreads

    # Check input ok
    args = get_options()

    # Check whether GPU libraries can be loaded
    use_gpu = check_and_set_gpu(args.use_gpu, gpu_lib, quit_on_fail = False)

    # Set threads for graph-tool
    setGtThreads(args.threads)

    # Open and process sequence database
    h5_fn = os.path.join(args.ref_db,os.path.basename(args.ref_db) + '.h5')
    ref_db = h5py.File(h5_fn, 'r')
    
    # Print overall database information
    ref_db = h5py.File(args.db, 'r')
    print("PopPUNK database:\t\t" + args.db)

    sketch_version = ref_db['sketches'].attrs['sketch_version']
    print("Sketch version:\t\t\t" + sketch_version)

    num_samples = len(ref_db['sketches'].keys())
    print("Number of samples:\t\t" + str(num_samples))

    first_sample = list(ref_db['sketches'].keys())[0]
    kmer_size = ref_db['sketches/' + first_sample].attrs['kmers']
    print("K-mer sizes:\t\t\t" + ",".join([str(x) for x in kmer_size]))

    sketch_size = int(ref_db['sketches/' + first_sample].attrs['sketchsize64']) * 64
    print("Sketch size:\t\t\t" + str(sketch_size))

    if 'random' in ref_db.keys():
        has_random = True
    else:
        has_random = False
    print("Contains random matches:\t" + str(has_random))

    try:
        codon_phased = ref_db['sketches'].attrs['codon_phased'] == 1
    except KeyError:
        codon_phased = False
    print("Codon phased seeds:\t\t" + str(codon_phased))
    
    # Print sample information
    sample_names = list(ref_db['sketches'].keys())
    sample_sequence_length = {}
    sample_missing_bases = {}
    sample_base_frequencies = {name: [] for name in sample_names}
    
    for sample_name in sample_names:
        sample_base_frequencies[sample_name] = ref_db['sketches/' + sample_name].attrs['base_freq']
        sample_sequence_length[sample_name] = ref_db['sketches/' + sample_name].attrs['length']
        sample_missing_bases[sample_name] = ref_db['sketches/' + sample_name].attrs['missing_bases']
    
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
        component_counts_df = component_assignments_df.groupby('labels')['vertex'].count()
        component_counts_df.name = 'component_count'
        component_information_df = component_assignments_df.merge(component_counts_df, on = ['labels'], how = 'left')
        outdegree_df = G.out_degree()
        graph_properties_df = component_information_df.merge(outdegree_df, on = ['vertex'])
    else:
        graph_properties_df = pd.DataFrame()
        graph_properties_df['vertex'] = np.arange(len(sample_names))
        graph_properties_df['labels'] = gt.label_components(G)[0].a
        graph_properties_df['degree'] = G.get_out_degrees(G.get_vertices())
        graph_properties_df['component_count'] = component_assignments.groupby('partition')['vertex'].transform('count')
    graph_properties_df = graph_properties_df.sort_values('vertex', axis = 0) # inplace not implemented for cudf
    graph_properties_df['vertex'] = sample_names
    
    # Merge data and print output
    with open(args.output,'w') as out_file:
        out_file.write(
            'Sample,Length,Missing_bases,Frequency_A,Frequency_C,Frequency_G,Frequency_T,Component_label,Component_size,Node_degree\n'
        )
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
