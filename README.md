# PopPUNK (POPulation Partitioning Using Nucleotide Kmers)

[![Dev build Status](https://dev.azure.com/jlees/PopPUNK/_apis/build/status/johnlees.PopPUNK?branchName=master)](https://dev.azure.com/jlees/PopPUNK/_build/latest?definitionId=1&branchName=master)
[![Documentation Status](https://readthedocs.org/projects/poppunk/badge/?version=latest)](https://poppunk.readthedocs.io/)
[![Anaconda package](https://anaconda.org/bioconda/poppunk/badges/version.svg)](https://anaconda.org/bioconda/poppunk)
[![PyPI version](https://badge.fury.io/py/poppunk.svg)](https://badge.fury.io/py/poppunk)

See our website: <https://www.poppunk.net>

## Description

See the [documentation](http://poppunk.readthedocs.io/en/latest/) and the
[paper](https://doi.org/10.1101/gr.241455.118).

If you find PopPUNK useful, please cite us:

Lees JA, Harris SR, Tonkin-Hill G, Gladstone RA, Lo SW, Weiser JN, Corander J, Bentley SD, Croucher NJ.
Fast and flexible bacterial genomic epidemiology with PopPUNK. *Genome Research* **29**:304-316 (2019).
doi:[10.1101/gr.241455.118](https://doi.org/10.1101/gr.241455.118)

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

See the [https://poppunk.readthedocs.io/en/latest/quickstart.html](quickstart) guide
for a brief tutorial.




