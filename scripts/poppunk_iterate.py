#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2021-2022 John Lees, Nick Croucher and Bin Zhao

from collections import defaultdict
import sys
import os
import argparse
import re
from copy import deepcopy
import numpy as np
from treeswift import Tree, Node
import h5py

import pp_sketchlib

# https://stackoverflow.com/a/17954769
from contextlib import contextmanager
@contextmanager
def stderr_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stderr.fileno()

    def _redirect_stderr(to):
        sys.stderr.close()
        os.dup2(to.fileno(), fd)
        sys.stderr = os.fdopen(fd, 'w')

    with os.fdopen(os.dup(fd), 'w') as old_stderr:
        with open(to, 'w') as file:
            _redirect_stderr(to=file)
        try:
            yield
        finally:
            _redirect_stderr(to=old_stderr)


# command line parsing
def get_options():

    parser = argparse.ArgumentParser(
        description="Cluster QC and analysis from multi-boundary method", prog="iterate"
    )

    # input options
    parser.add_argument(
        "--db", required=True, help="Output directory with results of --multi-boundary"
    )
    parser.add_argument(
        "--h5",
        default=None,
        help="Location of .h5 DB file [default = <--db>/<--db>.h5]",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Prefix for output files [default = <--db>/<--db>_iterate",
    )
    parser.add_argument(
        "--cutoff",
        default=0.1,
        type=float,
        help="Proportional distance cutoff below which to generate clusters, between 0 and 1 [default = 0.1]",
    )
    parser.add_argument(
        "--cpus", default=1, type=int, help="Number of CPUs to use [default = 1]"
    )

    return parser.parse_args()


def read_next_cluster_file(db_prefix):
    """Iterator over clusters with decreasing resolution

    Input:
        db_prefix:
            Prefix of the output directory with results of --multi-boundary

    Returns:
        all_clusters:
            dictionary of clusters (keys are cluster ids; values sets of members)
        no_singletons:
            all_clusters with singletons removed
        cluster_idx:
            The iterated ID of the file read
    """
    cluster_idx = 0
    while True:
        all_clusters = defaultdict(set)
        no_singletons = defaultdict(set)
        cluster_file = db_prefix + "_boundary" + str(cluster_idx) + "_clusters.csv"
        if os.path.isfile(cluster_file):
            with open(cluster_file) as f:
                f.readline()  # skip header
                for line in f:
                    name, cluster = line.rstrip().split(",")
                    all_clusters[int(cluster)].add(name)

            for cluster in all_clusters:
                if len(all_clusters[cluster]) > 1:
                    no_singletons[cluster] = all_clusters[cluster]
            yield (all_clusters, no_singletons, cluster_idx)

            cluster_idx += 1
        else:
            break


def is_nested(cluster_dict, child_members, node_list):
    """Check if a cluster is nested within another cluster that has
    already been added to the tree

    Input:
        cluster_dict:
            Dictionary of clusters (keys are cluster ids; values sets of members)
        child_members:
            Set of members of the child cluster
        node_list:
            List of clusters IDs already in the tree

    Returns:
        node:
            The node in the tree that contains the cluster
    """
    parent = None
    for node in node_list:
        if child_members.issubset(cluster_dict[node]) and \
          (parent == None or len(cluster_dict[node]) < len(cluster_dict[parent])):
            parent = node
    return parent


# main code
if __name__ == "__main__":

    # Check input ok
    args = get_options()
    if args.cutoff >= 1 or args.cutoff <= 0:
        raise RuntimeError("--cutoff must be between 0 and 1\n")
    if args.output is None:
        args.output = f"{args.db}/{args.db}_iterate"
    if args.h5 is None:
        args.h5 = f"{args.db}/{args.db}"
    else:
        # Remove the .h5 suffix if present
        h5_prefix = re.match("^(.+)\.h5$", args.h5)
        if h5_prefix != None:
            args.h5 = h5_prefix.group(1)

    # Set up reading clusters
    db_name = f"{args.db}/{os.path.basename(args.db)}"
    cluster_it = read_next_cluster_file(db_name)
    all_clusters, iterated_clusters, first_idx = next(cluster_it)
    all_samples = set()
    for cluster_samples in all_clusters.values():
        all_samples.update(cluster_samples)
    cluster_idx = max(iterated_clusters.keys())

    # Run cluster QC
    for (all_clusters, no_singletons, refine_idx) in cluster_it:
        for new_cluster in no_singletons.values():
            valid = True
            for old_cluster in iterated_clusters.values():
                if new_cluster == old_cluster or not (
                    new_cluster.issubset(old_cluster)
                    or old_cluster.issubset(new_cluster)
                    or len(new_cluster.intersection(old_cluster)) == 0
                ):
                    valid = False
                    break
            if valid:
                cluster_idx += 1
                iterated_clusters[cluster_idx] = new_cluster
    sorted_clusters = sorted(
        iterated_clusters, key=lambda k: len(iterated_clusters[k]), reverse=True
    )

    # Calculate core distances
    # Set up sketchlib args
    ref_db_handle = h5py.File(f"{args.h5}.h5", "r")
    first_sample_name = list(ref_db_handle["sketches"].keys())[0]
    kmers = ref_db_handle[f"sketches/{first_sample_name}"].attrs["kmers"]
    kmers = np.asarray(sorted(kmers))
    random_correct = True
    jaccard = False
    use_gpu = False
    deviceid = 0

    # Run a query for each cluster
    pi_values = {}
    max_pi = -1.0
    # Hide the progress bars
    with stderr_redirected():
        for cluster in sorted_clusters:
            rNames = list(iterated_clusters[cluster])
            distMat = pp_sketchlib.queryDatabase(
                args.h5,
                args.h5,
                rNames,
                rNames,
                kmers,
                random_correct,
                jaccard,
                args.cpus,
                use_gpu,
                deviceid,
            )
            pi_values[cluster] = np.mean(distMat[:, 0])
            max_pi = max(max_pi, pi_values[cluster])

    # Nest the clusters
    tree = Tree()
    root_node = Node(label="root")
    tree.root = root_node

    tree_clusters = deepcopy(iterated_clusters)
    node_list = {"root": root_node}
    tree_clusters["root"] = all_samples.copy()
    for cluster in sorted_clusters:
        new_node = Node(label="cluster" + str(cluster))
        parent_cluster = is_nested(
            tree_clusters, tree_clusters[cluster], node_list.keys()
        )
        if parent_cluster:
            node_list[parent_cluster].add_child(new_node)
            # Remove nested samples from the parent
            tree_clusters[parent_cluster] -= tree_clusters[cluster]

        node_list[cluster] = new_node

    # Add all the samples to the tree by looking through the list where children
    # have been removed
    for cluster in tree_clusters:
        for sample in tree_clusters[cluster]:
            node_list[cluster].add_child(Node(label=sample))

    # Write output
    tree.write_tree_newick(f"{args.output}.tree.nwk", hide_rooted_prefix=True)
    with open(f"{args.output}.clusters.csv", "w") as f:
        f.write("Cluster,Avg_Pi,Taxa\n")
        for cluster in sorted_clusters:
            f.write(f"{str(cluster)},{str(pi_values[cluster])},{';'.join(iterated_clusters[cluster])}\n")

    # Now add lengths in for the cut algorithm
    for node in tree.traverse_preorder(leaves=False):
        cluster_label = re.match(r"^cluster(\d+)$", node.get_label())
        if cluster_label and cluster_label.group(1):
            node.set_edge_length(pi_values[int(cluster_label.group(1))])
        elif node.get_label() != "root":
            raise RuntimeError(f"Couldn't parse cluster {node.get_label()}")
    tree.scale_edges(1 / max_pi)

    cut_clusters = set()
    for leaf in tree.traverse_leaves():
        parent_node = leaf.get_parent()
        if not parent_node.is_root():
            if parent_node.get_edge_length() < args.cutoff and parent_node.get_parent().is_root():
                cut_clusters.add(parent_node)
            elif parent_node.get_edge_length() < args.cutoff and not parent_node.get_parent().is_root():
            # For each leaf, go back up the tree to find the cluster which
            # crosses the cutoff threshold
                while parent_node.get_edge_length() < args.cutoff:
                    # Go up a level
                    child_node = parent_node
                    parent_node = child_node.get_parent()
                    if parent_node.is_root():
                        cut_clusters.add(child_node)
                        break
                    if (child_node.get_edge_length() < args.cutoff and \
                        parent_node.get_edge_length() > args.cutoff):
                        cut_clusters.add(child_node)
                        break

    # In the case where a grand-parent node's average core distance is smaller
    # than child node's both great parent node and child node will be selected
    # Remove any ancestors to fix this
    unwanted_parents = set()
    for selected_node in cut_clusters:
        for parent in selected_node.traverse_ancestors(include_self=False):
            if parent.get_label() in cut_clusters:
                unwanted_parents.add(parent)
                break
    cut_clusters -= unwanted_parents

    included_samples = set()
    with open(f"{args.output}.cutoff_clusters.csv", "w") as f:
        f.write("Isolate,Cluster\n")
        for idx, selected_node in enumerate(cut_clusters):
            cluster_label = re.match(r"^cluster(\d+)$", selected_node.get_label())
            if cluster_label and cluster_label.group(1):
                for sample in iterated_clusters[int(cluster_label.group(1))]:
                    included_samples.add(sample)
                    f.write(f"{sample},{idx + 1}\n")
            elif selected_node.get_label() != "root":
                raise RuntimeError(f"Couldn't parse cluster {selected_node.get_label()}")
        singletons = all_samples - included_samples
        for idx, sample in enumerate(singletons):
            f.write(f"{sample},{idx + len(cut_clusters) + 1}\n")

    sys.exit(0)
