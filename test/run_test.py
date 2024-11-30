#!/usr/bin/env python
# Copyright 2018-2020 John Lees and Nick Croucher

"""Tests for PopPUNK"""

import subprocess
import os
import sys
import shutil

if not os.path.isfile("12754_4#89.contigs_velvet.fa"):
    sys.stderr.write("Extracting example dataset\n")
    subprocess.run("tar xf example_set.tar.bz2", shell=True, check=True)

if os.environ.get("POPPUNK_PYTHON"):
    python_cmd = os.environ.get("POPPUNK_PYTHON")
else:
    python_cmd = "python"

sys.stderr.write("Running database creation (--create-db)\n")
subprocess.run(python_cmd + " ../poppunk-runner.py --create-db --r-files references.txt --min-k 13 --k-step 3 --plot-fit 5 --output example_db --overwrite", shell=True, check=True)

# create database with different QC options
sys.stderr.write("Running database QC test (--qc-db)\n")
subprocess.run(python_cmd + " ../poppunk-runner.py --qc-db --ref-db example_db --type-isolate \"12754_4#79\" --overwrite", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --qc-db --ref-db example_db --output example_qc --type-isolate \"12754_4#79\" --length-range 2000000 3000000 --overwrite", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --qc-db --ref-db example_db --output example_qc --type-isolate \"12754_4#79\" --remove-samples remove.txt --overwrite", shell=True, check=True)

#fit GMM
sys.stderr.write("Running GMM model fit (--fit-model gmm)\n")
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model bgmm --ref-db example_db --K 4 --overwrite", shell=True, check=True)

#fit dbscan
sys.stderr.write("Running DBSCAN model fit (--fit-model dbscan)\n")
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model dbscan --ref-db example_db --output example_dbscan --overwrite --graph-weights --for-refine", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model dbscan --ref-db example_db --output example_dbscan --overwrite --graph-weights", shell=True, check=True)

#refine model with GMM
sys.stderr.write("Running model refinement (--fit-model refine)\n")
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model refine --ref-db example_db --output example_refine --neg-shift 0.15 --overwrite", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model refine --ref-db example_db --output example_refine --manual-start manual.txt --overwrite", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model refine --ref-db example_db --output example_refine --neg-shift 0.15 --overwrite --indiv-refine both", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model refine --ref-db example_db --output example_refine --neg-shift 0.15 --overwrite --indiv-refine both --no-local", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model refine --ref-db example_db --output example_refine --neg-shift 0.15 --overwrite --unconstrained", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model refine --ref-db example_db --output example_refine --neg-shift 0.15 --overwrite --score-idx 1", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model refine --ref-db example_db --output example_refine --neg-shift 0.15 --overwrite --score-idx 2", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model threshold --threshold 0.003 --ref-db example_db --output example_threshold", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model refine --ref-db example_db --output example_refine --neg-shift 0.15 --summary-sample 15 --overwrite", shell=True, check=True)

sys.stderr.write("Running multi boundary refinement (--multi-boundary and poppunk_iterate.py)\n")
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model refine --ref-db example_db --output example_iterate --neg-shift -0.2 --overwrite --multi-boundary 10", shell=True, check=True)
subprocess.run(python_cmd + " ../scripts/poppunk_iterate.py --db example_iterate --h5 example_db/example_db", shell=True, check=True)

# lineage clustering
sys.stderr.write("Running lineage clustering test (--fit-model lineage)\n")
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model lineage --output example_lineages --ranks 1,2,3,5 --ref-db example_db --overwrite", shell=True, check=True)

#use model
sys.stderr.write("Running with an existing model (--use-model)\n")
subprocess.run(python_cmd + " ../poppunk-runner.py --use-model --ref-db example_db --model-dir example_db --output example_use --overwrite", shell=True, check=True)

#test pruning a database with a graph
subprocess.run(python_cmd + " ../poppunk-runner.py --qc-db --ref-db example_db --output example_network_qc --type-isolate \"12754_4#79\" --remove-samples remove.txt --overwrite", shell=True, check=True)

# tests of other command line programs
sys.stderr.write("Testing C++ extension\n")
subprocess.run(python_cmd + " test-refine.py", shell=True, check=True)

# assign query
sys.stderr.write("Running query assignment\n")
subprocess.run(python_cmd + " ../poppunk_assign-runner.py --query some_queries.txt --db example_db --model-dir example_refine --output example_query --overwrite --core --accessory", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_assign-runner.py --serial --query some_queries.txt --db example_db --model-dir example_refine --output example_query --overwrite --core --accessory", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_assign-runner.py --stable core --query some_queries.txt --db example_db --model-dir example_refine --output example_query_stable --previous-clustering example_refine --overwrite", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_assign-runner.py --query some_queries.txt --db example_db --model-dir example_refine --output example_query --run-qc --length-range 2900000 3000000 --max-zero-dist 1 --overwrite", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_assign-runner.py --query some_queries.txt --db example_db --model-dir example_refine --output example_query --run-qc --max-pi-dist 0.04 --max-zero-dist 1 --betweenness --overwrite", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_assign-runner.py --query more_queries.txt --db example_db --model-dir example_refine --output example_query --run-qc --max-zero-dist 0.3 --overwrite", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_assign-runner.py --query more_queries.txt --db example_db --model-dir example_refine --output example_query --run-qc --max-zero-dist 1 --max-merge 3 --overwrite", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_assign-runner.py --query some_queries.txt --db example_db --model-dir example_dbscan --output example_query_update --update-db --graph-weights --overwrite", shell=True, check=True) # uses graph weights
subprocess.run(python_cmd + " ../poppunk_assign-runner.py --query even_more_queries.txt --db example_query_update --model-dir example_dbscan --previous-clustering example_query_update --output example_query_update_2 --update-db --graph-weights --overwrite", shell=True, check=True) # uses graph weights
subprocess.run(python_cmd + " ../poppunk_assign-runner.py --query single_query.txt --db example_db --model-dir example_refine --output example_single_query --update-db --overwrite", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_assign-runner.py --query inref_query.txt --db example_db --model-dir example_refine --output example_single_query --write-references", shell=True, check=True) # matched name, but should be renamed in the output
subprocess.run(python_cmd + " ../poppunk_assign-runner.py --query some_queries.txt --db example_db --model-dir example_refine --model-dir example_lineages --output example_lineage_query --overwrite", shell=True, check=True)

#external clustering
sys.stderr.write("Running assign with external clustering (--fit-model refine)\n")
subprocess.run(python_cmd + " ../poppunk_assign-runner.py --query some_queries.txt --db example_db --model-dir example_refine --output example_query --overwrite --external-clustering example_external_clusters.csv", shell=True, check=True)

# test updating order is correct
sys.stderr.write("Running distance matrix order check (--update-db)\n")
subprocess.run(python_cmd + " test-update.py", shell=True, check=True)

# viz
sys.stderr.write("Running visualisations (poppunk_visualise)\n")
subprocess.run(python_cmd + " ../poppunk_visualise-runner.py --ref-db example_db --output example_viz --microreact", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_visualise-runner.py --ref-db example_db --output example_viz --cytoscape --network-file example_db/example_db_graph.gt", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_visualise-runner.py --ref-db example_db --output example_viz --phandango", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_visualise-runner.py --ref-db example_db --output example_viz --grapetree", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_visualise-runner.py --ref-db example_db --output example_viz_subset --microreact --include-files subset.txt", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_visualise-runner.py --ref-db example_db --query-db example_query --output example_viz_query --microreact", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_visualise-runner.py --ref-db example_db --previous-clustering example_lineages/example_lineages_lineages.csv --model-dir example_lineages --output example_lineage_viz --microreact", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_visualise-runner.py --distances example_query/example_query.dists --ref-db example_db --model-dir example_lineages --query-db example_lineage_query --output example_viz_query_lineages --microreact", shell=True, check=True)

# MST
sys.stderr.write("Running MST\n")
subprocess.run(python_cmd + " ../poppunk_visualise-runner.py --ref-db example_db --output example_mst --microreact --tree both", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_mst-runner.py --distance-pkl example_db/example_db.dists.pkl --rank-fit example_lineages/example_lineages_rank_5_fit.npz --previous-clustering example_dbscan/example_dbscan_clusters.csv --output example_sparse_mst --no-plot", shell=True, check=True)

# mandrake
sys.stderr.write("Running mandrake viz\n")
subprocess.run(python_cmd + " ../poppunk_mandrake-runner.py --distances example_db/example_db.dists --output example_mandrake --perplexity 5", shell=True, check=True)

# references
sys.stderr.write("Running poppunk_references\n")
subprocess.run(python_cmd + " ../poppunk_references-runner.py --network example_db/example_db_graph.gt --distances example_db/example_db.dists --ref-db example_db --output example_refs --model example_db", shell=True, check=True)

# info
sys.stderr.write("Running poppunk_info\n")
subprocess.run(python_cmd + " ../poppunk_info-runner.py --simple --db example_db", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_info-runner.py --db example_db", shell=True, check=True)

# lineages from strains
sys.stderr.write("Running poppunk_lineages_from_strains\n")
subprocess.run(python_cmd + " ../poppunk_lineages-runner.py --create-db example_db --db-scheme example_lineage_scheme.pkl --output lineage_creation_output --overwrite", shell=True, check=True)
if not os.path.exists('lineage_creation_output.csv'):
    sys.stderr.write('Failed to create lineages from strain database\n')
    sys.exit(1)
subprocess.run(python_cmd + " ../poppunk_lineages-runner.py --query-db some_queries.txt --db-scheme example_lineage_scheme.pkl --output lineage_querying_output --overwrite", shell=True, check=True)
if not os.path.exists('lineage_querying_output.csv'):
    sys.stderr.write('Failed to query lineages from strain database\n')
    sys.exit(1)

# beebop test
subprocess.run(python_cmd + " ../poppunk-runner.py --create-db --r-files rfile12.txt  --min-k 13 --k-step 3 --output batch12 --overwrite", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model dbscan --ref-db batch12 --output batch12 --overwrite", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model refine --ref-db batch12 --output batch12 --overwrite", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_assign-runner.py --db batch12 --query rfile3.txt --output batch3 --external-clustering batch12_external_clusters.csv --save-partial-query-graph  --overwrite", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_visualise-runner.py --ref-db batch12 --query-db batch3 --output batch123_viz --external-clustering batch12_external_clusters.csv --previous-query-clustering batch3/batch3_external_clusters.csv --cytoscape --rapidnj rapidnj --use-partial-query-graph ./batch3/batch3_query.subset --recalculate-distances --extend-query-graph --overwrite", shell=True, check=True)

# citations
sys.stderr.write("Printing citations\n")
subprocess.run(python_cmd + " ../poppunk-runner.py --citation --fit-model bgmm --ref-db example_db --K 4", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_assign-runner.py --citation --query some_queries.txt --db example_db --output example_query", shell=True, check=True)

sys.stderr.write("Tests completed\n")

