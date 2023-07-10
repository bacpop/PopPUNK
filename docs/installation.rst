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

If you want to use GPUs, take a look at :doc:`gpu`.

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

This may not deal with all necessary dependencies, but we are working on it
and it should again be possible in an upcoming release.

Clone the code
--------------
You can also clone the github to run the latest version, which is executed by::

    git clone https://github.com/bacpop/PopPUNK.git && cd PopPUNK
    python3 poppunk-runner.py

This will also give access to the :ref:`scripts`.

You will need to install the dependencies yourself (you can still use
conda or pip for this purpose). See ``environment.yml``.


