# PopPUNK (POPulation Partitioning Using Nucleotide Kmers)

[![Build Status](https://travis-ci.org/johnlees/PopPUNK.svg?branch=v1.1.1)](https://travis-ci.org/johnlees/PopPUNK/)
[![Documentation Status](https://readthedocs.org/projects/poppunk/badge/?version=latest)](https://poppunk.readthedocs.io/)
[![PyPI version](https://badge.fury.io/py/poppunk.svg)](https://badge.fury.io/py/poppunk)
[![Anaconda package](https://anaconda.org/bioconda/poppunk/badges/version.svg)](https://anaconda.org/bioconda/poppunk)

Step 1) Use k-mers to calculate core and accessory distances

Step 2) Use core and accessory distance distribution to define strains

Step 3) Pick references from strains, which can be used to assign new
query sequences

See the [documentation](http://poppunk.readthedocs.io/en/latest/) and the
[paper](https://doi.org/10.1101/gr.241455.118).

If you find PopPUNK useful, please cite as:

Lees JA, Harris SR, Tonkin-Hill G, Gladstone RA, Lo SW, Weiser JN, Corander J, Bentley SD, Croucher NJ.
Fast and flexible bacterial genomic epidemiology with PopPUNK. *Genome Research* **29**:304-316 (2019).
doi:[10.1101/gr.241455.118](https://doi.org/10.1101/gr.241455.118)

## Installation

### Through conda (recommended)

The easiest way is through conda, which is most easily accessed by first
installing [miniconda](https://conda.io/miniconda.html). PopPUNK can then
be installed by running:
```
conda install poppunk
```
If the package cannot be found you will need to add the necessary channels:
```
conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge
```

### Through pip
If you do not have conda you can also install through pip:
```
python3 -m pip install poppunk
```
You will need to be using Python 3.

Using both of these methods command `poppunk` will then be directly executable.
Alternatively clone this repository:
```
git clone git@github.com:johnlees/PopPUNK.git
```
Then run with `python poppunk-runner.py`.

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

If running without having installed through conda or pip,
run `python poppunk-runner.py` instead of `poppunk`.

See the [documentation](http://poppunk.readthedocs.io/en/latest/) for
full details.

## Other installation notes

Installing via conda takes care of all dependencies. For other methods you will
need to install the packages listed under dependencies.

If you wish to use the [/scripts](https://poppunk.readthedocs.io/en/latest/scripts.html)
you will need to clone the github.

### Dependencies

You will need a [mash](http://mash.readthedocs.io/en/latest/) installation
which is v2.0 or higher.

The following python packages are required, which can be installed
through `pip`. In brackets are the versions we used:

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

To install `numba` through pip, you may need `gcc >=v4.8` for it to work correctly.

### Test installation
Run the following command:
```
cd test && python run_test.py
```

If successful, you can clean the test data by running:
```
cd test && python clean_test.py
```




