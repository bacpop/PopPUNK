Using GPUs
==========

PopPUNK can use GPU acceleration of sketching (only when using sequence reads), distance
calculation, network construction and some aspects of visualisation. Installing and
configuring the required packages necessitates some extra steps, outlined below.

Installing GPU packages
-----------------------
To use GPU acceleration, PopPUNK uses ``cupy``, ``numba`` and the ``cudatoolkit``
packages from RAPIDS. Both ``cupy`` and ``numba`` can be installed as standard packages
using conda. The ``cudatoolkit`` packages need to be matched to your CUDA version.
The command ``nvidia-smi`` can be used to find the supported `CUDA version <https://stackoverflow.com/questions/53422407/different-cuda-versions-shown-by-nvcc-and-nvidia-smi>`__.
Installation of the ``cudatoolkit`` with conda (or the faster conda alternative,
`mamba <https://mamba.readthedocs.io/en/latest/installation.html>`__) should be guided
by the RAPIDS `guide <https://rapids.ai/start.html#get-rapids>`__. This information
will enable the installation of PopPUNK into a clean conda environment with a command
such as (modify the ``CUDA_VERSION`` variable as appropriate)::

    export CUDA_VERSION=11.3
    conda create -n poppunk_gpu -c rapidsai -c nvidia -c conda-forge \
    -c bioconda -c defaults rapids>=22.12 python=3.8 cudatoolkit=$CUDA_VERSION \
    pp-sketchlib>=2.0.1 poppunk>=2.6.0 networkx cupy numba
    conda activate poppunk_gpu

The version of ``pp-sketchlib`` on conda only supports some GPUs. A more general approach
is to install from source. This requires the installation of extra packages needed for
building packages from source. Additionally, it is sometimes necessary to install
versions of the CUDA compiler (``cuda-nvcc``) and runtime API (``cuda-cudart``)
that match the CUDA version. Although conda can be used, creating such a complex
environment can be slow, and therefore we recommend mamba as a faster alternative::

    export CUDA_VERSION=11.3
    mamba create -n poppunk_gpu -c rapidsai -c nvidia -c conda-forge \
    -c bioconda -c defaults rapids=22.12 python>=3.8 cudatoolkit=$CUDA_VERSION \
    cuda-nvcc=$CUDA_VERSION cuda-cudart=$CUDA_VERSION networkx cupy numba cmake \
    pybind11 highfive Eigen armadillo openblas libgomp libgfortran-ng poppunk>=2.6.0

.. note::

    On OSX replace ``libgomp libgfortan-ng`` with ``llvm-openmp gfortran_impl_osx-64``,
    and remove ``libgomp`` from ``environment.yml``.

Clone the sketchlib repository::

    git clone https://github.com/bacpop/pp-sketchlib.git
    cd pp-sketchlib

To correctly build ``pp-sketchlib``, the GPU architecture needs to be correctly
specified. The ``nvidia-smi`` command can be used to display the GPUs available
to you. This can be used to identify the corresponding compute version needed for
compilation (typically of the form ``sm_*``) using this `guide <https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/>`__
or the more limited table below. Edit the ``CMakeLists.txt`` if necessary to change
the compute version to that used by your GPU. See `the CMAKE_CUDA_COMPILER_VERSION
section <https://github.com/johnlees/pp-sketchlib/blob/master/CMakeLists.txt#L65-L68>`__.

.. table:: GPU compute versions
   :widths: auto
   :align: center

   ==================  =================
    GPU                Compute version
   ==================  =================
   20xx series         75
   30xx series         86
   40xx series         89
   V100                70
   A100                80
   A5000               86
   H100                90
   ==================  =================

Ensure ``nvcc`` is on your PATH and the CUDA libraries are available through the
``LD_LIBRARY_PATH`` variable::

    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}

The conda-installed version of ``pp-sketchlib`` can then be removed with the
command::

    conda remove --force pp-sketchlib

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

Selecting a GPU
---------------
A single GPU will be selected on systems where multiple devices are available. For
sketching and distance calculations, this can be specified by the ``--deviceid`` flag.
Alternatively, all GPU-enabled functions will used device 0 by default. Any GPU can
be set to device 0 using the system ``CUDA_VISIBLE_DEVICES`` variable, which can be set
before running PopPUNK; e.g. to use GPU device 1::

    export CUDA_VISIBLE_DEVICES=1

Using a GPU
-----------
By default, PopPUNK will use not use GPUs. To use them, you will need to add
the flag ``--gpu-sketch`` (when constructing or querying a database using reads),
``--gpu-dist`` (when constructing or querying a database from assemblies or reads),
or ``--gpu-graph`` (when querying or visualising a database, or fitting a model).
