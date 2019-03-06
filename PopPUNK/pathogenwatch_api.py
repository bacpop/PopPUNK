#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2019 John Lees and Nick Croucher

# core
import sys
import os
import shutil
import json
import pandas as pd

# poppunk
from .__main__ import assign_query

# main code
def main():
    db_loc = 'pw_ref' # hard coded for now
    cluster_loc = db_loc + '/gpsc_definitive.csv'

    # Query docs
    q_file_name = "qlist.txt"
    input_fasta = sys.stdin.read()
    with open('query.fa', 'w') as qfile:
        qfile.write(input_fasta)

    with open(q_file_name, 'w') as qlist:
        qlist.write("query.fa\n")

    # run assign query mode
    tmp_output_dir = 'tmp_pw_db'
    options = {
        'ref_db' : db_loc,
        'q_files' : q_file_name,
        'output' : tmp_output_dir,
        'update_db' : False,
        'full_db' : True,
        'distances' : db_loc + '/' + db_loc + '.dists',
        'microreact' : False,
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
        'external_clustering' : cluster_loc,
        'core_only' : False,
        'accessory_only' : False,
        'phandango' : False,
        'grapetree' : False,
        'info_csv' : None,
        'rapidnj' : None,
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

    gpsc_assignment = pd.read_csv(db_loc + "/" + os.path.basename(db_loc) + "_external_clusters.csv",
                                  header = 0, index_col = 0)
    gpsc_assigned = gpsc_assignment.loc['query.fa', 'GPSC']
    if pd.isna(gpsc_assigned):
        gpsc = 'novel'
    else:
        gpsc = gpsc_assigned.split(";")

    # stream results out
    results = {'strain': gpsc}
    json.dump(results, sys.stdout)

    # delete tmp files
    with open(q_file_name, 'r') as qlist:
        for query_file in qlist:
            os.remove(query_file)
    os.remove(q_file_name)
    shutil.rmtree(tmp_output_dir)

if __name__ == "__main__":
    main()

    sys.exit(0)
