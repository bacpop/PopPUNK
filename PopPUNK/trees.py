# vim: set fileencoding=<utf-8> :
# Copyright 2018-2020 John Lees and Nick Croucher

'''Functions for construction and processing of trees'''

import sys
import os
import io
import subprocess
import numpy as np
import pandas as pd

from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
from Bio.Phylo.TreeConstruction import DistanceMatrix
from treeswift import Tree, Node

from .utils import check_and_set_gpu

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

def buildRapidNJ(rapidnj, refList, coreMat, outPrefix, threads = 1):
    """Use rapidNJ for more rapid tree building

    Creates a phylip of core distances, system call to rapidnj executable, loads tree as
    dendropy object (cleaning quotes in node names), removes temporary files.

    Args:
        rapidnj (str)
            Location of rapidnj executable
        refList (list)
            Names of sequences in coreMat (same order)
        coreMat (numpy.array)
            NxN core distance matrix produced in :func:`~outputsForMicroreact`
        outPrefix (str)
            Prefix for all generated output files, which will be placed in `outPrefix` subdirectory
        threads (int)
            Number of threads to use

    Returns:
        tree (str)
            Newick-formatted NJ tree from core distances
    """
    # generate phylip matrix
    phylip_name = outPrefix + "/" + os.path.basename(outPrefix) + "_core_distances.phylip"
    with open(phylip_name, 'w') as pFile:
        pFile.write(str(len(refList))+"\n")
        for coreDist, ref in zip(coreMat, refList):
            pFile.write(ref)
            pFile.write(' '+' '.join(map(str, coreDist)))
            pFile.write("\n")

    # construct tree
    tree_filename = outPrefix + "/" + os.path.basename(outPrefix) + "_core_NJ.nwk"
    rapidnj_cmd = rapidnj + " " + phylip_name + " -n -i pd -o t -x " + tree_filename + ".raw -c " + str(threads)
    try:
        # run command
        subprocess.run(rapidnj_cmd, shell=True, check=True)

        # remove quotation marks for microreact
        with open(tree_filename + ".raw", 'r') as f, open(tree_filename, 'w') as fo:
            for line in f:
                fo.write(line.replace("'", ''))
        # tidy unnecessary files
        os.remove(tree_filename+".raw")
        os.remove(phylip_name)

    # record errors
    except subprocess.CalledProcessError as e:
        sys.stderr.write("Could not run command " + rapidnj_cmd + "; returned code: " + str(e.returncode) + "\n")
        sys.exit(1)

    # read tree and return
    tree = Phylo.read(tree_filename, "newick")
    os.remove(tree_filename)
    return tree

def write_tree(tree, prefix, suffix, overwrite):
    """Prints a Newick-formatted string to an output file

    Args:
        tree (str)
            Newick-formatted string representation of tree
        prefix (str)
            Prefix for output files
        suffix (str)
            Suffix for output files
        overwrite (bool)
            Whether to overwrite existing files
    """
    tree_filename = prefix + "/" + os.path.basename(prefix) + suffix
    if overwrite or not os.path.isfile(tree_filename):
        with open(tree_filename, 'w') as tree_file:
            tree_file.write(tree)
    else:
        sys.stderr.write("Unable to write phylogeny to " + tree_filename + "\n")

def tree_as_string(tree):
    """Converts a Phylo object into a newick string

    Args:
        tree (Phylo)
            Phylo object

    Returns:
        tree_string (str)
            Newick formatted string representing tree
    """
    output_str = io.StringIO()
    Phylo.write(tree, output_str, "newick")
    output_str.seek(0)
    tree_string = output_str.getvalue()
    tree_string = tree_string.replace("'","")
    return tree_string

def load_tree(prefix, type, distances = 'core'):
    """Checks for existing trees from previous runs.

    Args:
        prefix (str)
            Output prefix used for search
        type (str)
            Type of tree (NJ or MST)

    Returns:
        tree_string (str)
            Newick-formatted string of NJ tree
    """
    tree_string = None
    tree_prefix = os.path.join(prefix,os.path.basename(prefix))
    for suffix in ['_' + distances + '_' + type + ".tree",'_' + distances + '_' + type + ".nwk"]:
            tree_fn = tree_prefix + suffix
            if os.path.isfile(tree_fn):
                sys.stderr.write("Reading existing tree from " + tree_fn + "\n")
                tree = Phylo.read(tree_fn, "newick")
                tree_string = tree_as_string(tree)
                break

    return tree_string

def generate_nj_tree(coreMat, seqLabels, outPrefix, rapidnj, threads):
    """Generate phylogeny using dendropy or RapidNJ

    Writes a neighbour joining tree (.nwk) from core distances.

    Args:
        coreMat (numpy.array)
            n x n array of core distances for n samples.
        seqLabels (list)
            Processed names of sequences being analysed.
        outPrefix (str)
            Output prefix for core distances file
        rapidnj (str)
            A string with the location of the rapidnj executable for tree-building. If None, will
            use dendropy by default
        threads (int)
            Number of threads to use with rapidnj

    Returns:
        tree_string (str)
            Newick-formatted string of NJ tree
    """
    # Save distances to file

    # calculate phylogeny
    sys.stderr.write("Building phylogeny\n")
    if rapidnj is not None:
        core_dist_file = outPrefix + "/" + os.path.basename(outPrefix) + "_core_dists.csv"
        np.savetxt(core_dist_file, coreMat, delimiter=",", header = ",".join(seqLabels), comments="")
        tree = buildRapidNJ(rapidnj, seqLabels, coreMat, outPrefix, threads = threads)
        os.remove(core_dist_file)
    else:
        matrix = []
        for row, idx in enumerate(coreMat):
            matrix.append(row[0:idx].tolist())
        pdm = DistanceMatrix(seqLabels, matrix)
        tree = DistanceTreeConstructor().nj(pdm)

    tree.root_at_midpoint()
    tree_string = tree_as_string(tree)

    return tree_string

def mst_to_phylogeny(mst_network, names, use_gpu = False):
    """Convert a MST graph to a phylogeny

    Args:
       mst_network (network)
           Minimum spanning tree from
           :func:`~PopPUNK.network.generate_minimum_spanning_tree`
       names (list)
            Names of sequences in network
       use_gpu (bool)
            Whether to use GPU-specific libraries for processing

    Returns:
       mst_network (str)
           Minimum spanning tree (as graph-tool graph)
    """
    #
    # MST graph -> phylogeny
    #

    use_gpu = check_and_set_gpu(use_gpu, gpu_lib)


    # Identify edges
    if use_gpu:
        tree_nodes = [Node(label=v) for v in names]
        mst_edges_df = mst_network.view_edge_list()
    else:
        tree_nodes = [Node(label=names[int(v)]) for v in mst_network.get_vertices()]
        mst_edges_df = pd.DataFrame(mst_network.get_edges(),
                                    columns = ['src', 'dst'])
        mst_edges_df['weights'] = list(mst_network.ep['weight'])
    seed_node_index = int(mst_edges_df[['src','dst']].stack().mode()[0])
    tree = Tree()
    tree.root = tree_nodes[seed_node_index]

    # Generate links of tree
    parent_node_indices = [seed_node_index]
    added_nodes = set(parent_node_indices)
    i = 0
    while i < len(parent_node_indices): # NB loop end will increase
        # select links from data frames
        mst_links_dst = mst_edges_df[['dst','weights']].loc[mst_edges_df['src']==parent_node_indices[i]]
        mst_links_src = mst_edges_df[['src','weights']].loc[mst_edges_df['dst']==parent_node_indices[i]]
        mst_links_src.columns = ['dst','weights']
        mst_links = mst_links_dst.append(mst_links_src)
        if use_gpu:
            mst_links = mst_links.to_pandas()
        for (child_node, edge_length) in mst_links.itertuples(index=False, name=None):
            if child_node not in added_nodes:
                tree_nodes[parent_node_indices[i]].add_child(tree_nodes[child_node])
                tree_nodes[child_node].set_edge_length(edge_length)
                added_nodes.add(child_node)
                parent_node_indices.append(child_node)
        i = i + 1

    # Add zero length branches for internal nodes in MST
    for node in tree.traverse_preorder():
        if not node.is_leaf():
            new_child = Node(label=node.label, edge_length=0.0)
            node.set_label(None)
            node.add_child(new_child)

    # Return tree as string
    tree_string = tree.newick()
    return tree_string
