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

GPU packages
------------
To install RAPIDS, see the `guide <https://rapids.ai/start.html#get-rapids>`__. We
would recommend installing into a clean conda environment with a command such as::

    conda create -n poppunk_gpu -c rapidsai -c nvidia -c conda-forge \
    -c bioconda -c defaults rapids=0.17 python=3.8 cudatoolkit=11.0 \
    pp-sketchlib>=1.6.2 poppunk>=2.3.0 networkx
    conda activate poppunk_gpu

The version of pp-sketchlib on conda only supports some GPUs. If this doesn't work
for you, it is possible to install from source. Add the build dependencies to your
conda environment::

    conda install cmake pybind11 highfive Eigen armadillo openblas libgomp libgfortran-ng


.. note::

    On OSX replace ``libgomp libgfortan-ng`` with ``llvm-openmp gfortran_impl_osx-64``.

Clone the sketchlib repository::

    git clone https://github.com/johnlees/pp-sketchlib.git
    cd pp-sketchlib

Edit the ``CMakeLists.txt`` if necessary to change the compute version used by your GPU.
See `the CMAKE_CUDA_COMPILER_VERSION section <https://github.com/johnlees/pp-sketchlib/blob/master/CMakeLists.txt#L65-L68>`__.

.. table:: GPU compute versions
   :widths: auto
   :align: center

   ==================  =================
    GPU                Compute version
   ==================  =================
   20xx series         75
   30xx series         86
   V100                70
   A100                80
   ==================  =================

Make sure you have CUDA toolkit installed (this is available via conda as ``cudatoolkit``)
and ``nvcc`` is on your PATH::

    export PATH=/usr/local/cuda-11.1/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

Then run::

    python setup.py install

You should see a message that the CUDA compiler is found, in which case the compilation
and installation of sketchlib will include GPU components::

    -- Looking for a CUDA compiler
    -- Looking for a CUDA compiler - /usr/local/cuda-11.1/bin/nvcc
    -- CUDA found, compiling both GPU and CPU code
    -- The CUDA compiler identification is NVIDIA 11.1.105
    -- Detecting CUDA compiler ABI info
    -- Detecting CUDA compiler ABI info - done
    -- Check for working CUDA compiler: /usr/local/cuda-11.1/bin/nvcc - skipped
    -- Detecting CUDA compile features
    -- Detecting CUDA compile features - done

You can confirm that your custom installation of sketchlib is being used by checking
the location of sketchlib library reported by ``popppunk`` points to your python
site-packages, rather than the conda version.
