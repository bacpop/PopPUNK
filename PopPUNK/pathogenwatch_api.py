#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2019 John Lees and Nick Croucher

# core
import sys
import os
import shutil
import json
import operator
from collections import Counter

# poppunk
from .__main__ import assign_query
from .utils import readClusters

# main code
def main():
    db_loc = 'pw_ref' # hard coded for now

    # Query docs
    q_file_name = "qlist.txt"
    query_ids = []
    tmp_query_files = []
    in_json = open('all_in.json', 'r')
    for input_doc in in_json.readline():
    #for input_doc in sys.stdin.readline():
        query = json.loads(input_doc)
        query_ids.append(query['fileId'])

        # Write fasta for sketching (delete later)
        tmp_sequence_file = query['fileId']
        tmp_query_files.append(tmp_sequence_file)
        with open(tmp_sequence_file, 'w') as qfile:
            qfile.write(query['content'])

        sys.stderr.write("Read sample doc for " + query['fileId'] + "\n")

    with open(q_file_name, 'w') as qlist:
        for query_file in tmp_query_files:
            qlist.write(query_file + "\n")

    # run assign query mode
    tmp_output_dir = 'tmp_pw_db'
    options = {
        'ref_db' : db_loc,
        'q_files' : q_file_name,
        'output' : tmp_output_dir,
        'update_db' : True,
        'full_db' : False,
        'distances' : db_loc + '/' + db_loc + '.dists',
        'microreact' : True,
        'cytoscape' : False,
        'kmers' : [13, 17, 21, 25, 29],
        'sketch_sizes' : [10000, 10000, 100000, 10000, 10000],
        'ignore_length' : False,
        'threads' : 1,
        'mash' : 'mash',
        'overwrite' : True,
        'plot_fit' : False,
        'no_stream' : True,
        'max_a_dist' : 0.5,
        'model_dir' : db_loc,
        'previous_clustering' : None,
        'external_clustering' : None,
        'core_only' : False,
        'accessory_only' : False,
        'phandango' : False,
        'grapetree' : False,
        'info_csv' : None,
        'rapidnj' : 'rapidnj',
        'perplexity' : 20
    }

    sys.stderr.write("Running PopPUNK\n")
    clusters = assign_query(options['ref_db'], options['q_files'], options['output'], options['update_db'],
                 options['full_db'], options['distances'], options['microreact'], options['cytoscape'],
                 options['kmers'], options['sketch_sizes'], options['ignore_length'], options['threads'],
                 options['mash'], options['overwrite'], options['plot_fit'], options['no_stream'],
                 options['max_a_dist'], options['model_dir'], options['previous_clustering'],
                 options['external_clustering'], options['core_only'], options['accessory_only'],
                 options['phandango'], options['grapetree'], options['info_csv'], options['rapidnj'],
                 options['perplexity'])['combined']

    # reformat output as new strain and results JSON
    sys.stderr.write("Streaming output\n")

    # Find novel clusters
    prev_clusters = readClusters(db_loc + "/" + os.path.basename(db_loc) + "_clusters.csv")
    if set(clusters.keys()).difference(set(prev_clusters.keys())):
        new_strains = set(clusters.keys()).difference(set(prev_clusters.keys()))
    else:
        new_strains = []

    # iterate through cluster dictionary sorting by value
    freq_order = sorted(dict(Counter(clusters.values())).items(), key=operator.itemgetter(1), reverse=True)
    freq_order = [x[0] for x in freq_order]
    new_strain_dict = {}
    for cluster_member, cluster_name in sorted(clusters.items(), key=lambda i:freq_order.index(i[1])):
        new_strain_dict[cluster_member] = str(cluster_name)

    # pick up tree and dot from file
    with open(tmp_output_dir + "/" + tmp_output_dir + "_core_NJ.nwk", 'r') as nj_file:
        nj_tree = nj_file.read()
    with open(tmp_output_dir + "/" + tmp_output_dir + "_perplexity" +
              str(options['perplexity']) + "_accessory_tsne.dot", 'r') as dot_file:
        accessory_dot = dot_file.read()

    # stream results out
    results = {'new_strains': new_strains, 'other': new_strain_dict, 'tree': nj_tree, 'dot': accessory_dot}
    json.dump(results, sys.stdout)

    # delete tmp files
    with open(q_file_name, 'w') as qlist:
        for query_file in tmp_query_files:
            os.remove(query_file)
    os.remove(q_file_name)
    shutil.rmtree(tmp_output_dir)

if __name__ == "__main__":
    main()

    sys.exit(0)
