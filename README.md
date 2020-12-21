# POPulation Partitioning Using Nucleotide Kmers <img src='docs/images/poppunk_v2.png' align="right" height="100" />

<!-- badges: start -->
[![Dev build Status](https://dev.azure.com/jlees/PopPUNK/_apis/build/status/johnlees.PopPUNK?branchName=master)](https://dev.azure.com/jlees/PopPUNK/_build/latest?definitionId=1&branchName=master)
[![Documentation Status](https://readthedocs.org/projects/poppunk/badge/?version=latest)](https://poppunk.readthedocs.io/)
[![Anaconda package](https://anaconda.org/bioconda/poppunk/badges/version.svg)](https://anaconda.org/bioconda/poppunk)
[![PyPI version](https://badge.fury.io/py/poppunk.svg)](https://badge.fury.io/py/poppunk)
<!-- badges: end -->

See our website: <https://www.poppunk.net>

## Description

See the [documentation](http://poppunk.readthedocs.io/en/latest/) and the
[paper](https://doi.org/10.1101/gr.241455.118).

If you find PopPUNK useful, please cite us:

Lees JA, Harris SR, Tonkin-Hill G, Gladstone RA, Lo SW, Weiser JN, Corander J, Bentley SD, Croucher NJ.
Fast and flexible bacterial genomic epidemiology with PopPUNK. *Genome Research* **29**:304-316 (2019).
doi:[10.1101/gr.241455.118](https://doi.org/10.1101/gr.241455.118)

## News

### 2020-09-30
We have discovered a bug affecting the interaction of pp-sketchlib and PopPUNK.
If you have used `PopPUNK >=v2.0.0` with `pp-sketchlib <v1.5.1` label order may
be incorrect (see issue [#95](https://github.com/johnlees/PopPUNK/issues/95)).

Please upgrade to `PopPUNK >=v2.2` and `pp-sketchlib >=v1.5.1`. If this is not
possible, you can either:
- Run `scripts/poppunk_pickle_fix.py` on your `.dists.pkl` file and re-run
  model fits.
- Create the database with `poppunk_sketch` directly, rather than `
  PopPUNK --create-db`

## Installation

This is for the command line version. For more details see [installation](https://poppunk.readthedocs.io/en/latest/installation.html) in the documentation.

There are other interfaces, in-browser and through galaxy, [detailed here](https://poppunk.net/pages/interfaces.html).

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

## Quick usage

See the [quickstart](https://poppunk.readthedocs.io/en/latest/quickstart.html) guide
for a brief tutorial.
