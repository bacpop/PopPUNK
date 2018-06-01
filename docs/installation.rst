Installation
============

The easiest way to install is through pip, which will also install the
dependencies::

   pip install poppunk

You can also clone the github to run the latest version, which is executed by::

   python poppunk-runner.py

You will also need `mash <http://mash.readthedocs.io/en/latest/>`__ (v2 or higher)
installed.

Python installation
-------------------

We recommend the use of a `miniconda <https://conda.io/miniconda.html>`__
installation.

Using the miniconda installation will also allow the use of faster linear
algebra functions by installing dependencies as follows::

   conda install numpy scipy mkl

Dependencies
------------
We tested PopPUNK with the following packages:

* ``DendroPy`` (4.3.0)
* ``hdbscan`` (0.8.13)
* ``matplotlib`` (2.1.2)
* ``networkx`` (2.1)
* ``numpy`` (1.14.1)
* ``numba`` (0.36.2)
* ``pandas`` (0.22.0)
* ``scikit-learn`` (0.19.1)
* ``scipy`` (1.0.0)
* ``sharedmem`` (0.3.5)

Optionally, you can use `rapidnj <http://birc.au.dk/software/rapidnj/>`__
if producing output with ``--microreact`` and ``--rapidnj`` options. We used
v2.3.2.

