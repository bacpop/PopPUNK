# PopPUNK (POPulation Partitioning Using Nucleotide Kmers)

Step 1) Use k-mers to calculate core and accessory distances

Step 2) Use core and accessory distance distribution to define strains

Step 3) Pick references from strains, which can be used to assign new
query sequences

See the [documentation](http://poppunk.readthedocs.io/en/master/).

## Installation
The easiest way is through pip, which we would also recommend being
a [miniconda](https://conda.io/miniconda.html) install:
```
pip install poppunk
```

The command `poppunk` will then be directly executable. Alternatively
clone this repository:
```
git clone git@github.com:johnlees/PopPUNK.git
```
Then run with `python poppunk-runner.py`.

### Dependencies

In brackets are the versions we used

* python3
* `DendroPy` (4.3.0)
* `hdbscan` (0.8.13)
* `matplotlib` (2.1.2)
* `networkx` (2.1)
* `numba` (0.36.2)
* `numpy` (1.14.1)
* `pandas` (0.22.0)
* `scikit-learn` (0.19.1)
* `scipy` (1.0.0)
* `sharedmem` (0.3.5)

### Test installation
Run the following command:
```
cd test && python run_test.py
```

If successful, you can clean the test data by running:
```
cd test && python clean_test.py
```

## Quick usage
Easy run mode, go from assemblies to clusters of strains:
```
poppunk --easy-run --r-files reference_list.txt --output poppunk_db
```

Or in two parts. First, create the database:
```
poppunk --create-db \
   --r-files reference_list.txt \
   --output poppunk_db \
   --threads 2 \
   --k-step 2 \
   --min-k 9 \
   --plot-fit 5
```

Then fit the model:
```
poppunk --fit-model \
   --ref-db poppunk_db \
   --distances poppunk_db/poppunk_db.dists \
   --output poppunk_db \
   --full-db \
   --K 2
```

Once fitted, new query sequences can quickly be assigned:
```
poppunk --assign-query \
   --ref-db poppunk_db \
   --q-files query_list.txt \
   --output query_results \
   --update-db
```

If running without having installed through PyPI, run `python poppunk-runner.py` instead of `poppunk`.

See the [documentation](http://poppunk.readthedocs.io/en/master/) for
full details.


