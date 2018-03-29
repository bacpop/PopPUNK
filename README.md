# PopPUNK (POPulation Partitioning Using Nucleotide Kmers)

Step 1) Use k-mers to calculate core and accessory distances

Step 2) Use core and accessory distance distribution to define strains

Step 3) Pick references from strains, which can be used to assign new
query sequences

## Quick usage
Create the database:
```
poppunk --create-db \\
   --r-files reference_list.txt \\
   --output poppunk_db \\
   --threads 2 \\
   --k-step 2 \\
   --min-k 9 \\
   --plot-fit 5
```

Fit the model:
```
poppunk --fit-model \\
   --distances poppunk_db/poppunk_db.dists \\
   --output poppunk_db \\
   --full-db \\
   --dpgmm \\
   --K 3
```

If running without having installed through PyPI, run `python poppunk-runner.py` instead of `poppunk`.

