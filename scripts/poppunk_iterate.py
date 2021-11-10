#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2021 John Lees, Nick Croucher and Bin Zhao

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
    if args.output is None:
        args.output = args.db + "/" + args.db + "_iterate"
    if args.h5 is None:
        args.h5 = args.db + "/" + args.db
    else:
        # Remove the .h5 suffix if present
        h5_prefix = re.match("^(.+)\.h5$", args.h5)
        if h5_prefix != None:
            args.h5 = h5_prefix.group(1)

    # Set up reading clusters
    db_name = args.db + "/" + os.path.basename(args.db)
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
    ref_db_handle = h5py.File(args.h5 + ".h5", "r")
    first_sample_name = list(ref_db_handle["sketches"].keys())[0]
    kmers = ref_db_handle["sketches/" + first_sample_name].attrs["kmers"]
    kmers = np.asarray(sorted(kmers))
    random_correct = True
    jaccard = False
    use_gpu = False
    deviceid = 0

    # Run a query for each cluster
    pi_values = {}
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

    # Nest the clusters
    tree = Tree()
    root_node = Node(label="root")
    tree.root = root_node

    tree_clusters = deepcopy(iterated_clusters)
    node_list = {"root": root_node}
    tree_clusters["root"] = all_samples
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
    tree.write_tree_newick(args.output + ".tree.nwk", hide_rooted_prefix=True)
    with open(args.output + ".clusters.csv", "w") as f:
        f.write("Cluster,Avg_Pi,Taxa\n")
        for cluster in sorted_clusters:
            f.write(
                str(cluster)
                + ","
                + str(pi_values[cluster])
                + ","
                + ";".join(iterated_clusters[cluster])
                + "\n"
            )

    sys.exit(0)
