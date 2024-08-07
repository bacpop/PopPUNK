# POPulation Partitioning Using Nucleotide Kmers <img src='docs/images/poppunk_v2.png' align="right" height="100" />

<!-- badges: start -->
[![Dev build Status](https://dev.azure.com/jlees/PopPUNK/_apis/build/status/johnlees.PopPUNK?branchName=master)](https://dev.azure.com/jlees/PopPUNK/_build/latest?definitionId=1&branchName=master)
![Run tests](https://github.com/bacpop/PopPUNK/workflows/Run%20tests/badge.svg)
[![Build and publish docs](https://github.com/bacpop/PopPUNK/actions/workflows/docs_push.yml/badge.svg)](https://github.com/bacpop/PopPUNK/actions/workflows/docs_push.yml)
[![Anaconda package](https://anaconda.org/bioconda/poppunk/badges/version.svg)](https://anaconda.org/bioconda/poppunk)
[![PyPI version](https://badge.fury.io/py/poppunk.svg)](https://badge.fury.io/py/poppunk)
<!-- badges: end -->

## Description

Links:
- [Documentation](https://poppunk.bacpop.org/)
- [Databases](https://www.bacpop.org/poppunk/)
- [Paper](https://doi.org/10.1101/gr.241455.118)

If you find PopPUNK useful, please cite us:

Lees JA, Harris SR, Tonkin-Hill G, Gladstone RA, Lo SW, Weiser JN, Corander J, Bentley SD, Croucher NJ.
Fast and flexible bacterial genomic epidemiology with PopPUNK. *Genome Research* **29**:304-316 (2019).
doi:[10.1101/gr.241455.118](https://doi.org/10.1101/gr.241455.118)

You can also run your command with `--citation` to get a [list of citations](https://poppunk.readthedocs.io/en/latest/citing.html) and a suggested methods paragraph.

## News and roadmap

The [roadmap](https://poppunk.bacpop.org/roadmap.html) can be found in the documentation.

### 2024-08-07
PopPUNK 2.7.0 comes with two changes:
- Distance matrices `<db_name>.dists.npy` are no longer required or written when using
`poppunk_assign`, with or without `--update-db`. These can be very large, especially
with many samples, so this saves space and memory in model reuse and distribution. Note that
the `<db_name>.dists.pkl` file is still required (but this is small).
- We have added a `--stable` flag to `poppunk_assign`. Rather than merging hybrid clusters,
new samples will simply be assigned to their nearest neighbour. This implies `--serial` and
cannot be run with `--update-db`. This behaviour mimics the 'stable nomenclature' of schemes
such as [LIN](https://doi.org/10.1093/molbev/msac135).

### 2023-01-18
We have retired the PopPUNK website. Databases have been expanded, and can be
found here: https://www.bacpop.org/poppunk/.

### 2022-08-04
The change in scikit-learn's API in v1.0.0 and above mean that HDBSCAN models
fitted with `sklearn <=v0.24` will give an error when loaded. If you run into this,
the solution is one of:
- Downgrade sklearn to v0.24.
- Run model refinement to turn your model into a boundary model instead (this will
change clusters).
- Refit your model in an environment with `sklearn >=v1.0`.

If this is a common problem let us know, as we could write a script to 'upgrade'
HDBSCAN models.
See issue [#213](https://github.com/bacpop/PopPUNK/issues/213) for more details.

### 2021-03-15
We have fixed a number of bugs with may affect the use of `poppunk_assign` with
`--update-db`. We have also fixed a number of bugs with GPU distances. These are
'advanced' features and are not likely to be encountered in most cases, but if you do wish to use either of these features please make sure that you are using
`PopPUNK >=v2.4.0` with `pp-sketchlib >=v1.7.0`.

### 2020-09-30
We have discovered a bug affecting the interaction of pp-sketchlib and PopPUNK.
If you have used `PopPUNK >=v2.0.0` with `pp-sketchlib <v1.5.1` label order may
be incorrect (see issue [#95](https://github.com/bacpop/PopPUNK/issues/95)).

Please upgrade to `PopPUNK >=v2.2` and `pp-sketchlib >=v1.5.1`. If this is not
possible, you can either:
- Run `scripts/poppunk_pickle_fix.py` on your `.dists.pkl` file and re-run
  model fits.
- Create the database with `poppunk_sketch` directly, rather than
  `PopPUNK --create-db`

## Installation

This is for the command line version. For more details see [installation](https://poppunk.bacpop.org/installation.html) in the documentation.

Our (beta) web interface BeeBOP is now also available: https://beebop.dide.ic.ac.uk/

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

See the [overview](https://poppunk.bacpop.org/overview.html) first. There are two ways of running:

### With a supported species

1) Download an [existing database](https://www.bacpop.org/poppunk/).
2) [Run assignment](https://poppunk.bacpop.org/query_assignment.html).

### With a new species.
1) [Create sketches of input](https://poppunk.bacpop.org/sketching.html).
2) [Run QC](https://poppunk.bacpop.org/qc.html).
3) [Build a model](https://poppunk.bacpop.org/model_fitting.html).

## Docker image

A docker image is available

```
docker pull mrcide/poppunk:bacpop-20
```
