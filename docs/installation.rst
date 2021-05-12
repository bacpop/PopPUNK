Installation
============

The easiest way to install is through conda, which will also install the
dependencies::

    conda install poppunk

Then run with ``poppunk``.

.. important::
    From v2.1.0 onwards, PopPUNK requires python3.8 to run
    (which on many default Linux installations is
    run using ``python3`` rather than ``python``).

.. important::
    From v2.1.2 onwards, PopPUNK no longer supports ``mash``. If you want to
    use older databases created with ``mash``, please downgrade to <v2

Installing with conda (recommended)
-----------------------------------
If you do not have ``conda`` you can install it through
`miniconda <https://conda.io/miniconda.html>`_ and then add the necessary
channels::

    conda config --add channels defaults
    conda config --add channels bioconda
    conda config --add channels conda-forge

Then run::

    conda install poppunk

If you are having conflict issues with conda, our advice would be:

- Remove and reinstall miniconda.
- Never install anything in the base environment
- Create a new environment for PopPUNK with ``conda create -n pp_env poppunk``

If you have an older version of PopPUNK, you can upgrade using this method -- you
may also wish to specify the version, for example ``conda install poppunk==2.3.0`` if you
wish to upgrade.

conda-forge also has some helpful tips: https://conda-forge.org/docs/user/tipsandtricks.html

Installing with pip
-------------------
If you do not have conda, you can also install through pip::

    python3 -m pip install poppunk

This may not deal with all necessary :ref:`dependencies`.

Clone the code
--------------
You can also clone the github to run the latest version, which is executed by::

    git clone https://github.com/johnlees/PopPUNK.git && cd PopPUNK
    python3 poppunk-runner.py

This will also give access to the :ref:`scripts`.

You will need to install the :ref:`dependencies` yourself (you can still use
conda or pip for this purpose).

.. _dependencies:

Dependencies
------------
This documentation refers to a conda installation with the following packages:

* python3 (3.8.2)
* ``pp-sketchlib`` (1.6.2)
* ``DendroPy`` (4.3.0)
* ``hdbscan`` (0.8.13)
* ``matplotlib`` (2.1.2)
* ``graph-tool`` (2.31)
* ``numpy`` (1.14.1)
* ``pandas`` (0.22.0)
* ``scikit-learn`` (0.19.1)
* ``scipy`` (1.0.0)
* ``sharedmem`` (0.3.5)

Optionally, you can use `rapidnj <http://birc.au.dk/software/rapidnj/>`__
if producing output with ``--microreact`` and ``--rapidnj`` options. We used
v2.3.2.
