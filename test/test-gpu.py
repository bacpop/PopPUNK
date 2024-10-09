#!/usr/bin/env python
# Copyright 2018-2022 John Lees and Nick Croucher

"""Tests for PopPUNK using GPUs where possible"""

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
subprocess.run(python_cmd + " ../poppunk-runner.py --create-db --r-files references.txt --min-k 13 --k-step 3 --plot-fit 5 --output example_db --overwrite --gpu-dist", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --qc-db --ref-db example_db --type-isolate \"12754_4#79\" --overwrite", shell=True, check=True)

# test updating order is correct
sys.stderr.write("Running distance matrix order check (--update-db)\n")
subprocess.run(python_cmd + " test-update-gpu.py", shell=True, check=True)

#fit GMM
sys.stderr.write("Running GMM model fit (--fit-model gmm)\n")
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model bgmm --ref-db example_db --K 4 --overwrite --gpu-graph", shell=True, check=True)

#fit dbscan
sys.stderr.write("Running DBSCAN model fit (--fit-model dbscan)\n")
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model dbscan --ref-db example_db --output example_dbscan --overwrite --gpu-graph --gpu-model", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model dbscan --ref-db example_db --output example_dbscan --overwrite --gpu-graph --gpu-model --for-refine", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model dbscan --ref-db example_db --output example_dbscan --overwrite --gpu-graph --gpu-model --assign-subsample 1000", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model dbscan --ref-db example_db --output example_dbscan --overwrite --gpu-graph --gpu-model --model-subsample 5000", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model dbscan --ref-db example_db --output example_dbscan --overwrite --graph-weights --gpu-graph", shell=True, check=True)

#refine model with GMM
sys.stderr.write("Running model refinement (--fit-model refine)\n")
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model refine --ref-db example_db --output example_refine --neg-shift 0.2 --overwrite --gpu-graph", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model refine --ref-db example_db --output example_refine --manual-start manual.txt --overwrite --gpu-graph", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model refine --ref-db example_db --output example_refine --neg-shift 0.2 --overwrite --indiv-refine both --gpu-graph", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model refine --ref-db example_db --output example_refine --neg-shift 0.2 --overwrite --indiv-refine both --no-local --gpu-graph", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model refine --ref-db example_db --output example_refine --neg-shift 0.2 --overwrite --unconstrained --gpu-graph", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model refine --ref-db example_db --output example_refine --neg-shift 0.2 --overwrite --score-idx 1 --gpu-graph", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model refine --ref-db example_db --output example_refine --neg-shift 0.2 --overwrite --score-idx 2 --gpu-graph", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model threshold --threshold 0.003 --ref-db example_db --output example_threshold --gpu-graph", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model refine --ref-db example_db --output example_refine --neg-shift 0.15 --summary-sample 15 --overwrite --gpu-graph", shell=True, check=True)

# lineage clustering
sys.stderr.write("Running lineage clustering test (--fit-model lineage)\n")
subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model lineage --output example_lineages --ranks 1,2,3,5 --ref-db example_db --overwrite --gpu-graph", shell=True, check=True)

#use model
sys.stderr.write("Running with an existing model (--use-model)\n")
subprocess.run(python_cmd + " ../poppunk-runner.py --use-model --ref-db example_db --model-dir example_db --output example_use --overwrite --gpu-graph", shell=True, check=True)

# tests of other command line programs
sys.stderr.write("Testing C++ extension\n")
subprocess.run(python_cmd + " test-refine.py", shell=True, check=True)

#assign query
sys.stderr.write("Running query assignment\n")
subprocess.run(python_cmd + " ../poppunk_assign-runner.py --query some_queries.txt --db example_db --model-dir example_refine --output example_query --overwrite --gpu-dist --gpu-graph --core --accessory", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_assign-runner.py --query some_queries.txt --db example_db --model-dir example_dbscan --output example_query_update --update-db --graph-weights --overwrite --gpu-dist  --gpu-graph", shell=True, check=True) # uses graph weights
subprocess.run(python_cmd + " ../poppunk_assign-runner.py --query single_query.txt --db example_db --model-dir example_refine --output example_single_query --update-db --overwrite --gpu-dist  --gpu-graph", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_assign-runner.py --query some_queries.txt --db example_db --model-dir example_lineages --output example_lineage_query --overwrite --gpu-graph --gpu-dist", shell=True, check=True)

# viz
sys.stderr.write("Running visualisations (poppunk_visualise)\n")
subprocess.run(python_cmd + " ../poppunk_visualise-runner.py --ref-db example_db --output example_viz --microreact --gpu-graph", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_visualise-runner.py --ref-db example_db --output example_viz --cytoscape --network-file example_db/example_db_graph.csv.gz --gpu-graph", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_visualise-runner.py --ref-db example_db --output example_viz --phandango --gpu-graph", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_visualise-runner.py --ref-db example_db --output example_viz --grapetree --gpu-graph", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_visualise-runner.py --ref-db example_db --output example_viz_subset --microreact --include-files subset.txt --gpu-graph", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_visualise-runner.py --ref-db example_db --query-db example_query --output example_viz_query --microreact --gpu-graph", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_visualise-runner.py --ref-db example_db --previous-clustering example_lineages/example_lineages_lineages.csv --model-dir example_lineages --output example_lineage_viz --microreact --gpu-graph", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_visualise-runner.py --distances example_query/example_query.dists --ref-db example_db --model-dir example_lineages --query-db example_lineage_query --output example_viz_query_lineages --microreact --gpu-graph", shell=True, check=True)

# MST
sys.stderr.write("Running MST\n")
subprocess.run(python_cmd + " ../poppunk_visualise-runner.py --ref-db example_db --output example_mst --microreact --tree both --gpu-graph", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_mst-runner.py --distance-pkl example_db/example_db.dists.pkl --rank-fit example_lineages/example_lineages_rank_5_fit.npz --previous-clustering example_dbscan/example_dbscan_clusters.csv --output example_sparse_mst --no-plot --gpu-graph", shell=True, check=True)

# mandrake
sys.stderr.write("Running mandrake viz\n")
subprocess.run(python_cmd + " ../poppunk_mandrake-runner.py --distances example_db/example_db.dists --output example_mandrake --perplexity 5 --use-gpu", shell=True, check=True)

# references
sys.stderr.write("Running poppunk_references\n")
subprocess.run(python_cmd + " ../poppunk_references-runner.py --network example_db/example_db_graph.csv.gz --distances example_db/example_db.dists --ref-db example_db --output example_refs --model example_db --use-gpu", shell=True, check=True)

# info
sys.stderr.write("Running poppunk_info\n")
subprocess.run(python_cmd + " ../poppunk_info-runner.py --db example_db --output example_db.info.csv --use-gpu", shell=True, check=True)

# citations
sys.stderr.write("Printing citations\n")
subprocess.run(python_cmd + " ../poppunk-runner.py --citation --fit-model bgmm --ref-db example_db --K 4", shell=True, check=True)
subprocess.run(python_cmd + " ../poppunk_assign-runner.py --citation --query some_queries.txt --db example_db --output example_query", shell=True, check=True)

sys.stderr.write("Tests completed\n")

