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

#easy run
sys.stderr.write("Running database creation (--create-db)\n")
subprocess.run(python_cmd + " ../poppunk-runner.py --create-db --r-files references.txt --min-k 13 --k-step 3 --output example_db --qc-filter prune --overwrite", shell=True, check=True)

# create database with different QC options
sys.stderr.write("Running database QC test (--create-db)\n")
subprocess.run(python_cmd + " ../poppunk-runner.py --create-db --r-files references.txt --min-k 13 --k-step 3 --output example_qc --qc-filter continue --length-range 2000000 3000000 --overwrite", shell=True, check=True)

# test updating order is correct
sys.stderr.write("Running distance matrix order check (--update-db)\n")
subprocess.run(python_cmd + " test-update.py", shell=True, check=True)

#fit GMM
sys.stderr.write("Running GMM model fit (--fit-model gmm)\n")
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model bgmm --ref-db example_db --K 4 --overwrite", shell=True, check=True)

#fit dbscan
sys.stderr.write("Running DBSCAN model fit (--fit-model dbscan)\n")
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model dbscan --ref-db example_db --output example_dbscan --overwrite --graph-weights", shell=True, check=True)

#refine model with GMM
sys.stderr.write("Running model refinement (--fit-model refine)\n")
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model refine --ref-db example_db --output example_refine --neg-shift 0.8 --overwrite", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model refine --ref-db example_db --output example_refine --neg-shift 0.8 --overwrite --indiv-refine both", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model refine --ref-db example_db --output example_refine --neg-shift 0.8 --overwrite --score-idx 1", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model refine --ref-db example_db --output example_refine --neg-shift 0.8 --overwrite --score-idx 2", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model threshold --threshold 0.003 --ref-db example_db --output example_threshold", shell=True, check=True)

# lineage clustering
sys.stderr.write("Running lineage clustering test (--fit-model lineage)\n")
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model lineage --output example_lineages --ranks 1,2,3,5 --ref-db example_db --overwrite", shell=True, check=True)

#use model
sys.stderr.write("Running with an existing model (--use-model)\n")
subprocess.run(python_cmd + " ../poppunk-runner.py --use-model --ref-db example_db --model-dir example_db --output example_use --overwrite", shell=True, check=True)

# tests of other command line programs
sys.stderr.write("Testing C++ extension\n")
subprocess.run(python_cmd + " test-refine.py", shell=True, check=True)

#assign query
sys.stderr.write("Running query assignment\n")
subprocess.run(python_cmd + " ../poppunk_assign-runner.py --query some_queries.txt --db example_db --output example_query --overwrite", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_assign-runner.py --query some_queries.txt --db example_db --output example_query_update --update-db --graph-weights --overwrite", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_assign-runner.py --query single_query.txt --db example_db --output example_single_query --update-db --overwrite", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_assign-runner.py --query some_queries.txt --db example_db --model-dir example_lineages --output example_lineage_query --overwrite", shell=True, check=True)

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
subprocess.run(python_cmd + " ../poppunk_visualise-runner.py --ref-db example_db --output example_mst --microreact --tree mst", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_mst-runner.py --distance-pkl example_db/example_db.dists.pkl --rank-fit example_lineages/example_lineages_rank5_fit.npz --previous-clustering example_dbscan/example_dbscan_clusters.csv --output example_sparse_mst --no-plot", shell=True, check=True)

# t-sne
sys.stderr.write("Running tsne viz\n")
subprocess.run(python_cmd + " ../poppunk_tsne-runner.py --distances example_db/example_db.dists --output example_tsne --perplexity 5 --verbosity 1", shell=True, check=True)

# prune
sys.stderr.write("Running poppunk_prune\n")
subprocess.run(python_cmd + " ../poppunk_prune-runner.py --distances example_db/example_db.dists --ref-db example_db --remove subset.txt --output example_prune", shell=True, check=True)

# references
sys.stderr.write("Running poppunk_references\n")
subprocess.run(python_cmd + " ../poppunk_references-runner.py --network example_db/example_db_graph.gt --distances example_db/example_db.dists --ref-db example_db --output example_refs --model example_db", shell=True, check=True)

# web API
sys.stderr.write("Running API tests\n")
subprocess.run(python_cmd + " test-web.py", shell=True, check=True)

sys.stderr.write("Tests completed\n")

