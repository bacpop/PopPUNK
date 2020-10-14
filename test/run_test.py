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

#easy run
sys.stderr.write("Running database creation + DBSCAN model fit + fit refinement (--easy-run)\n")
subprocess.run("python ../poppunk-runner.py --easy-run --r-files references.txt --min-k 13 --k-step 3 --output example_db --full-db --qc-filter prune --overwrite", shell=True, check=True)

# create database with different QC options
sys.stderr.write("Running database QC test (--create-db)\n")
subprocess.run("python ../poppunk-runner.py --create-db --r-files references.txt --min-k 13 --k-step 3 --output example_qc --full-db --qc-filter continue --length-range 2000000 3000000 --overwrite", shell=True, check=True)

#fit GMM
sys.stderr.write("Running GMM model fit (--fit-model)\n")
subprocess.run("python ../poppunk-runner.py --fit-model --distances example_db/example_db.dists --ref-db example_db --output example_db --full-db --K 4 --overwrite", shell=True, check=True)

#refine model with GMM
sys.stderr.write("Running model refinement (--refine-model)\n")
subprocess.run("python ../poppunk-runner.py --refine-model --distances example_db/example_db.dists --ref-db example_db --output example_refine --neg-shift 0.8 --overwrite", shell=True, check=True)

#use model
sys.stderr.write("Running with an existing model (--use-model)\n")
subprocess.run("python ../poppunk-runner.py --use-model --ref-db example_db --model-dir example_db --distances example_db/example_db.dists --output example_use --overwrite", shell=True, check=True)

# lineage clustering
sys.stderr.write("Running lineage clustering test (--lineage-clustering)\n")
subprocess.run("python ../poppunk-runner.py --lineage-clustering --distances example_db/example_db.dists --output example_lineages --ranks 1,2,3,5 --full-db --ref-db example_db --overwrite", shell=True, check=True)


# tests of other command line programs

#assign query
sys.stderr.write("Running query assignment\n")
subprocess.run("python ../poppunk_assign-runner.py --q-files queries.txt --distances example_db/example_db.dists --ref-db example_db --output example_query --update-db --overwrite", shell=True, check=True)

# assign query to lineages
sys.stderr.write("Running query assignment (--assign-lineages)\n")
subprocess.run("python ../poppunk_assign-runner.py --assign-lineages --q-files some_queries.txt --distances example_db/example_db.dists --ref-db example_db --model-dir example_lineages --output example_lineage_query --update-db --overwrite", shell=True, check=True)

# viz
sys.stderr.write("Running visualisations (poppunk_visualise)\n")
subprocess.run("python ../poppunk_visualise-runner.py --distances example_db/example_db.dists --ref-db example_db --output example_viz --microreact --subset subset.txt", shell=True, check=True)

# t-sne
sys.stderr.write("Running tsne viz\n")
subprocess.run("python ../poppunk_tsne-runner.py --distances example_db/example_db.dists --output example_tsne --perplexity 5 --verbosity 1", shell=True, check=True)

# prune
sys.stderr.write("Running poppunk_prune\n")
subprocess.run("python ../poppunk_prune-runner.py --distances example_db/example_db.dists --ref-db example_db --remove subset.txt --output example_prune", shell=True, check=True)

# references
sys.stderr.write("Running poppunk_references\n")
subprocess.run("python ../poppunk_references-runner.py --network example_db/example_db_graph.gt --distances example_db/example_db.dists --ref-db example_db --output example_refs --model example_db", shell=True, check=True)

sys.stderr.write("Tests completed\n")

