# vim: set fileencoding=<utf-8> :
# Copyright 2018-2023 John Lees and Nick Croucher

'''Classes used for model fits'''

# universal
import os
import sys
# additional
import numpy as np
import random
import pickle
import shutil
import re
from sklearn import utils
import scipy.optimize
from scipy import stats
import scipy.sparse
import hdbscan

# Parallel support
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map, process_map
from functools import partial
import collections
try:
    from multiprocessing import Pool, RLock, shared_memory
    from multiprocessing.managers import SharedMemoryManager
    NumpyShared = collections.namedtuple('NumpyShared', ('name', 'shape', 'dtype'))
except ImportError as e:
    sys.stderr.write("This version of PopPUNK requires python v3.8 or higher\n")
    sys.exit(0)

# Load GPU libraries
try:
    import cupyx
    import cugraph
    import cudf
    import cupy as cp
    from numba import cuda
    import rmm
    from cuml import cluster
except ImportError:
    pass

import pp_sketchlib
import poppunk_refine

from .__main__ import betweenness_sample_default

from .utils import set_env
from .utils import check_and_set_gpu

# BGMM
from .bgmm import fit2dMultiGaussian
from .bgmm import findWithinLabel
from .bgmm import findBetweenLabel_bgmm
from .bgmm import log_likelihood
from .plot import plot_results
from .plot import plot_contours

# DBSCAN
from .dbscan import fitDbScan
from .dbscan import findBetweenLabel
from .dbscan import evaluate_dbscan_clusters
from .plot import plot_dbscan_results

# refine
from .refine import refineFit, multi_refine
from .refine import readManualStart
from .plot import plot_refined_results

# lineage
from .plot import distHistogram
epsilon = 1e-10

# Format for rank fits
def rankFile(rank):
    return('_rank_' + str(rank) + '_fit.npz')

def loadClusterFit(pkl_file, npz_file, outPrefix = "", max_samples = 100000,
                   use_gpu = False):
    '''Call this to load a fitted model

    Args:
        pkl_file (str)
            Location of saved .pkl file on disk
        npz_file (str)
            Location of saved .npz file on disk
        outPrefix (str)
            Output prefix for model to save to (e.g. plots)
        max_samples (int)
            Maximum samples if subsampling X
            [default = 100000]
        use_gpu (bool)
            Whether to load npz file with GPU libraries
            for lineage models

    Returns:
        load_obj (model)
            Loaded model

    '''
    with open(pkl_file, 'rb') as pickle_obj:
        fit_object, fit_type = pickle.load(pickle_obj)

    if fit_type == 'lineage':
        # Can't save multiple sparse matrices to the same file, so do some
        # file name processing
        fit_file = os.path.basename(pkl_file)
        prefix = re.match(r"^(.+)_fit\.pkl$", fit_file)
        rank_file = os.path.dirname(pkl_file) + "/" + \
                      prefix.group(1) + '_sparse_dists.npz'
        fit_data = scipy.sparse.load_npz(rank_file)
    else:
        fit_data = np.load(npz_file)

    if fit_type == "bgmm":
        sys.stderr.write("Loading BGMM 2D Gaussian model\n")
        load_obj = BGMMFit(outPrefix, max_samples)
    elif fit_type == "dbscan":
        sys.stderr.write("Loading DBSCAN model\n")
        load_obj = DBSCANFit(outPrefix, max_samples)
    elif fit_type == "refine":
        sys.stderr.write("Loading previously refined model\n")
        load_obj = RefineFit(outPrefix)
    elif fit_type == "lineage":
        sys.stderr.write("Loading lineage cluster model\n")
        load_obj = LineageFit(outPrefix,
                                *fit_object)
    else:
        raise RuntimeError("Undefined model type: " + str(fit_type))

    load_obj.load(fit_data, fit_object)
    sys.stderr.write("Completed model loading\n")
    return load_obj

def assign_samples(chunk, X, y, model, scale, chunk_size, values = False):
    """Runs a models assignment on a chunk of input

    Args:
        chunk (int)
            Index of chunk to process
        X (NumpyShared)
            n x 2 array of core and accessory distances for n samples
        y (NumpyShared)
            An n-vector to store results, with the most likely cluster memberships
            or an n by k matrix with the component responsibilities for each sample.
        weights (numpy.array)
            Component weights from :class:`~PopPUNK.models.BGMMFit`
        means (numpy.array)
            Component means from :class:`~PopPUNK.models.BGMMFit`
        covars (numpy.array)
            Component covariances from :class:`~PopPUNK.models.BGMMFit`
        scale (numpy.array)
            Scaling of core and accessory distances from :class:`~PopPUNK.models.BGMMFit`
        chunk_size (int)
            Size of each chunk in X
        values (bool)
            Whether to return the responsibilities, rather than the most
            likely assignment (used for entropy calculation).

            Default is False
    """
    # Make sure this is run single threaded
    with set_env(MKL_NUM_THREADS='1',
                 NUMEXPR_NUM_THREADS='1',
                 OMP_NUM_THREADS='1'):
        if isinstance(X, NumpyShared):
            X_shm = shared_memory.SharedMemory(name = X.name)
            X = np.ndarray(X.shape, dtype = X.dtype, buffer = X_shm.buf)
        if isinstance(y, NumpyShared):
            y_shm = shared_memory.SharedMemory(name = y.name)
            y = np.ndarray(y.shape, dtype = y.dtype, buffer = y_shm.buf)

        start = chunk * chunk_size
        end = min((chunk + 1) * chunk_size, X.shape[0])
        if start >= end:
            raise RuntimeError("start >= end in BGMM assign")

        if isinstance(model, BGMMFit):
            logprob, lpr = log_likelihood(X[start:end, :], model.weights,
                                          model.means, model.covariances, scale)
            responsibilities = np.exp(lpr - logprob[:, np.newaxis])
            # Default to return the most likely cluster
            if values == False:
                y[start:end] = responsibilities.argmax(axis=1)
            # Can return the actual responsibilities
            else:
                y[start:end, :] = responsibilities
        elif isinstance(model, DBSCANFit):
            y[start:end] = hdbscan.approximate_predict(model.hdb, X[start:end, :]/scale)[0]


class ClusterFit:
    '''Parent class for all models used to cluster distances

    Args:
        outPrefix (str)
            The output prefix used for reading/writing
    '''

    def __init__(self, outPrefix, default_dtype = np.float32):
        self.outPrefix = outPrefix
        if outPrefix != "" and not os.path.isdir(outPrefix):
            try:
                os.makedirs(outPrefix)
            except OSError:
                sys.stderr.write("Cannot create output directory " + outPrefix + "\n")
                sys.exit(1)

        self.fitted = False
        self.indiv_fitted = False
        self.default_dtype = default_dtype
        self.threads = 1

    def set_threads(self, threads):
        self.threads = threads

    def fit(self, X = None):
        '''Initial steps for all fit functions.

        Creates output directory. If preprocess is set then subsamples passed X

        Args:
            X (numpy.array)
                The core and accessory distances to cluster. Must be set if
                preprocess is set.

                (default = None)
            default_dtype (numpy dtype)
                Type to use if no X provided
        '''
        # set output dir
        if not os.path.isdir(self.outPrefix):
            if not os.path.isfile(self.outPrefix):
                os.makedirs(self.outPrefix)
            else:
                sys.stderr.write(self.outPrefix + " already exists as a file! Use a different --output\n")
                sys.exit(1)

        if X is not None:
            self.default_dtype = X.dtype

        # preprocess subsampling
        if self.preprocess:
            if X.shape[0] > self.max_samples:
                self.subsampled_X = utils.shuffle(X, random_state=random.randint(1,10000))[0:self.max_samples,]
            else:
                self.subsampled_X = np.copy(X)

            # perform scaling
            self.scale = np.amax(self.subsampled_X, axis = 0)
            self.subsampled_X /= self.scale

    def plot(self, X=None):
        '''Initial steps for all plot functions.

        Ensures model has been fitted.

        Args:
            X (numpy.array)
                The core and accessory distances to subsample.

                (default = None)
        '''
        if not self.fitted:
            raise RuntimeError("Trying to plot unfitted model")

    def no_scale(self):
        '''Turn off scaling (useful for refine, where optimization
        is done in the scaled space).
        '''
        self.scale = np.array([1, 1], dtype = self.default_dtype)

    def copy(self, prefix):
        """Copy the model to a new directory
        """
        self.outPrefix = prefix
        self.save()


class BGMMFit(ClusterFit):
    '''Class for fits using the Gaussian mixture model. Inherits from :class:`ClusterFit`.

    Must first run either :func:`~BGMMFit.fit` or :func:`~BGMMFit.load` before calling
    other functions

    Args:
        outPrefix (str)
            The output prefix used for reading/writing
        max_samples (int)
            The number of subsamples to fit the model to
            (default = 100000)
    '''

    def __init__(self, outPrefix, max_samples = 100000, max_batch_size = 100000, assign_points = True):
        ClusterFit.__init__(self, outPrefix)
        self.type = 'bgmm'
        self.preprocess = True
        self.max_samples = max_samples
        self.max_batch_size = max_batch_size
        self.assign_points = assign_points

    def fit(self, X, max_components):
        '''Extends :func:`~ClusterFit.fit`

        Fits the BGMM and returns assignments by calling
        :func:`~PopPUNK.bgmm.fit2dMultiGaussian`.

        Fitted parameters are stored in the object.

        Args:
            X (numpy.array)
                The core and accessory distances to cluster. Must be set if
                preprocess is set.
            max_components (int)
                Maximum number of mixture components to use.

        Returns:
            y (numpy.array)
                Cluster assignments of samples in X
        '''
        ClusterFit.fit(self, X)
        self.dpgmm = fit2dMultiGaussian(self.subsampled_X, max_components)
        self.weights = self.dpgmm.weights_
        self.means = self.dpgmm.means_
        self.covariances = self.dpgmm.covariances_
        self.fitted = True
        
        # Allow for partial fitting that only assigns the subsample not the full set
        if self.assign_points:
            y = self.assign(X, max_batch_size = self.max_batch_size)
        else:
            y = self.assign(self.subsampled_X, max_batch_size = self.max_batch_size)
        self.within_label = findWithinLabel(self.means, y)
        self.between_label = findBetweenLabel_bgmm(self.means, y)
        return y


    def save(self):
        '''Save the model to disk, as an npz and pkl (using outPrefix).'''
        if not self.fitted:
            raise RuntimeError("Trying to save unfitted model")
        else:
            np.savez(self.outPrefix + "/" + os.path.basename(self.outPrefix) + '_fit.npz',
             weights=self.weights,
             means=self.means,
             covariances=self.covariances,
             within=self.within_label,
             between=self.between_label,
             scale=self.scale)
            with open(self.outPrefix + "/" + os.path.basename(self.outPrefix) + '_fit.pkl', 'wb') as pickle_file:
                pickle.dump([self.dpgmm, self.type], pickle_file)


    def load(self, fit_npz, fit_obj):
        '''Load the model from disk. Called from :func:`~loadClusterFit`

        Args:
            fit_npz (dict)
                Fit npz opened with :func:`numpy.load`
            fit_obj (sklearn.mixture.BayesianGaussianMixture)
                The saved fit object
        '''
        self.dpgmm = fit_obj
        self.weights = fit_npz['weights']
        self.means = fit_npz['means']
        self.covariances = fit_npz['covariances']
        self.scale = fit_npz['scale']
        self.within_label = fit_npz['within'].item()
        self.between_label = fit_npz['between'].item()
        self.fitted = True


    def plot(self, X, y):
        '''Extends :func:`~ClusterFit.plot`

        Write a summary of the fit, and plot the results using
        :func:`PopPUNK.plot.plot_results` and :func:`PopPUNK.plot.plot_contours`

        Args:
            X (numpy.array)
                Core and accessory distances
            y (numpy.array)
                Cluster assignments from :func:`~BGMMFit.assign`
        '''
        ClusterFit.plot(self, X)
        # Generate a subsampling if one was not used in the fit
        if not hasattr(self, 'subsampled_X'):
            self.subsampled_X = utils.shuffle(X, random_state=random.randint(1,10000))[0:self.max_samples,]

        y_subsample = self.assign(self.subsampled_X, max_batch_size = self.max_batch_size, values=True, progress=False)
        avg_entropy = np.mean(np.apply_along_axis(stats.entropy, 1,
                                                  y_subsample))
        used_components = np.unique(y).size
        sys.stderr.write("Fit summary:\n" + "\n".join(["\tAvg. entropy of assignment\t" +  "{:.4f}".format(avg_entropy),
                                                        "\tNumber of components used\t" + str(used_components)]) + "\n\n")
        sys.stderr.write("Scaled component means:\n")
        for centre in self.means:
            sys.stderr.write("\t" + str(centre) + "\n")
        sys.stderr.write("\n")

        title = "DPGMM – estimated number of spatial clusters: " + str(len(np.unique(y)))
        outfile = self.outPrefix + "/" + os.path.basename(self.outPrefix) + "_DPGMM_fit"

        plot_results(X, y, self.means, self.covariances, self.scale, title, outfile)
        plot_contours(self, y, title + " assignment boundary", outfile + "_contours")


    def assign(self, X, max_batch_size = 100000, values = False, progress=True):
        '''Assign the clustering of new samples using :func:`~PopPUNK.bgmm.assign_samples`

        Args:
            X (numpy.array)
                Core and accessory distances
            values (bool)
                Return the responsibilities of assignment rather than most likely cluster
            max_batch_size (int)
                Size of batches to be assigned
            progress (bool)
                Show progress bar

                [default = True]
        Returns:
            y (numpy.array)
                Cluster assignments or values by samples
        '''
        if not self.fitted:
            raise RuntimeError("Trying to assign using an unfitted model")
        else:
            if progress:
                sys.stderr.write("Assigning distances with BGMM model\n")

            if values:
                y = np.zeros((X.shape[0], len(self.weights)), dtype=X.dtype)
            else:
                y = np.zeros(X.shape[0], dtype=int)
            block_size = max_batch_size
            with SharedMemoryManager() as smm:
                shm_X = smm.SharedMemory(size = X.nbytes)
                X_shared_array = np.ndarray(X.shape, dtype = X.dtype, buffer = shm_X.buf)
                X_shared_array[:] = X[:]
                X_shared = NumpyShared(name = shm_X.name, shape = X.shape, dtype = X.dtype)

                shm_y = smm.SharedMemory(size = y.nbytes)
                y_shared_array = np.ndarray(y.shape, dtype = y.dtype, buffer = shm_y.buf)
                y_shared_array[:] = y[:]
                y_shared = NumpyShared(name = shm_y.name, shape = y.shape, dtype = y.dtype)

                thread_map(partial(assign_samples,
                                           X = X_shared,
                                           y = y_shared,
                                           model = self,
                                           scale = self.scale,
                                           chunk_size = block_size,
                                           values = values),
                                    range((X.shape[0] - 1) // block_size + 1),
                                    max_workers=self.threads,
                                    disable=(progress == False))

                y[:] = y_shared_array[:]

        return y


class DBSCANFit(ClusterFit):
    '''Class for fits using HDBSCAN. Inherits from :class:`ClusterFit`.

    Must first run either :func:`~DBSCANFit.fit` or :func:`~DBSCANFit.load` before calling
    other functions

    Args:
        outPrefix (str)
            The output prefix used for reading/writing
        max_samples (int)
            The number of subsamples to fit the model to
            (default = 100000)
    '''

    def __init__(self, outPrefix, use_gpu = False, max_batch_size = 5000, max_samples = 100000, assign_points = True):
        ClusterFit.__init__(self, outPrefix)
        self.type = 'dbscan'
        self.preprocess = True
        self.max_batch_size = max_batch_size
        self.max_samples = max_samples
        self.assign_points = assign_points
        self.use_gpu = use_gpu # Updated below

    def fit(self, X, max_num_clusters, min_cluster_prop, use_gpu = False):
        '''Extends :func:`~ClusterFit.fit`

        Fits the distances with HDBSCAN and returns assignments by calling
        :func:`~PopPUNK.dbscan.fitDbScan`.

        Fitted parameters are stored in the object.

        Args:
            X (numpy.array)
                The core and accessory distances to cluster. Must be set if
                preprocess is set.
            max_num_clusters (int)
                Maximum number of clusters in DBSCAN fitting
            min_cluster_prop (float)
                Minimum proportion of points in a cluster in DBSCAN fitting
            use_gpu (bool)
                Whether GPU algorithms should be used in DBSCAN fitting

        Returns:
            y (numpy.array)
                Cluster assignments of samples in X
        '''
        ClusterFit.fit(self, X)

        # DBSCAN parameters
        cache_out = "./" + self.outPrefix + "_cache"
        min_samples = max(int(min_cluster_prop * self.subsampled_X.shape[0]), 10) # do not allow clusters of < 10 points
        min_samples = min(min_samples,1023) # do not allow clusters to require more than 1023 points at the start
        min_cluster_size = max(int(0.01 * self.subsampled_X.shape[0]), 10)

        # Check on initialisation of GPU libraries and memory
        # Convert to cupy if using GPU to avoid implicit numpy conversion below
        if use_gpu:
            try:
                import cudf
                from cuml import cluster
                import cupy as cp
                gpu_lib = True
            except ImportError as e:
                gpu_lib = False
            # check on GPU
            use_gpu = check_and_set_gpu(use_gpu,
                                gpu_lib,
                                quit_on_fail = True)
            if use_gpu:
                self.use_gpu = True
                self.subsampled_X = cp.asarray(self.subsampled_X)
            else:
                self.use_gpu = False

        indistinct_clustering = True
        while indistinct_clustering and min_cluster_size >= min_samples and min_samples >= 10:
            # Fit model
            self.hdb, self.labels, self.n_clusters = fitDbScan(self.subsampled_X,
                                                                min_samples,
                                                                min_cluster_size,
                                                                cache_out,
                                                                use_gpu = use_gpu)
            self.fitted = True # needed for predict

            # Test whether model fit contains distinct clusters
            if self.n_clusters > 1 and self.n_clusters <= max_num_clusters:
            
              if use_gpu:
                  # get within strain cluster
                  self.max_cluster_num = int(self.labels.max())
                  self.cluster_means = cp.full((self.n_clusters,2),0.0,dtype=float)
                  self.cluster_mins = cp.full((self.n_clusters,2),0.0,dtype=float)
                  self.cluster_maxs = cp.full((self.n_clusters,2),0.0,dtype=float)

                  for i in range(self.max_cluster_num+1):
                      labelled_rows = cp.where(self.labels==i,True,False)
                      self.cluster_means[cp.array(i),] = [cp.mean(self.subsampled_X[labelled_rows,cp.array([0])]),cp.mean(self.subsampled_X[labelled_rows,cp.array([1])])]
                      self.cluster_mins[cp.array(i),] = [cp.min(self.subsampled_X[labelled_rows,cp.array([0])]),cp.min(self.subsampled_X[labelled_rows,cp.array([1])])]
                      self.cluster_maxs[cp.array(i),] = [cp.max(self.subsampled_X[labelled_rows,cp.array([0])]),cp.max(self.subsampled_X[labelled_rows,cp.array([1])])]
                  
              else:
                  # get within strain cluster
                  self.max_cluster_num = self.labels.max()
                  self.cluster_means = np.full((self.n_clusters,2),0.0,dtype=float)
                  self.cluster_mins = np.full((self.n_clusters,2),0.0,dtype=float)
                  self.cluster_maxs = np.full((self.n_clusters,2),0.0,dtype=float)

                  for i in range(self.max_cluster_num+1):
                      self.cluster_means[i,] = [np.mean(self.subsampled_X[self.labels==i,0]),np.mean(self.subsampled_X[self.labels==i,1])]
                      self.cluster_mins[i,] = [np.min(self.subsampled_X[self.labels==i,0]),np.min(self.subsampled_X[self.labels==i,1])]
                      self.cluster_maxs[i,] = [np.max(self.subsampled_X[self.labels==i,0]),np.max(self.subsampled_X[self.labels==i,1])]

              # Run assignment
              y = self.assign(self.subsampled_X,
                              no_scale=True,
                              progress=False,
                              max_batch_size = self.subsampled_X.shape[0],
                              use_gpu = use_gpu)
              
              # Evaluate clustering
              self.within_label = findWithinLabel(self.cluster_means, y)
              self.between_label = findBetweenLabel(y, self.within_label)
              indistinct_clustering = evaluate_dbscan_clusters(self)

            # Alter minimum cluster size criterion
            if min_cluster_size < min_samples / 2:
                min_samples = min_samples // 10
            min_cluster_size = int(min_cluster_size / 2)

        # Report failure where it happens
        if indistinct_clustering:
            self.fitted = False
            sys.stderr.write("Failed to find distinct clusters in this dataset\n")
            sys.exit(1)
        elif not use_gpu:
            shutil.rmtree(cache_out)

        # Allow for partial fitting that only assigns the subsample not the full set
        if self.assign_points:
            y = self.assign(X, max_batch_size = self.max_batch_size, use_gpu = use_gpu)
        else:
            y = self.assign(self.subsampled_X, max_batch_size = self.max_batch_size, use_gpu = use_gpu)

        return y


    def save(self):
        '''Save the model to disk, as an npz and pkl (using outPrefix).'''
        if not self.fitted:
            raise RuntimeError("Trying to save unfitted model")
        else:
            np.savez(self.outPrefix + "/" + os.path.basename(self.outPrefix) + '_fit.npz',
             n_clusters=self.n_clusters,
             within=self.within_label,
             between=self.between_label,
             means=self.cluster_means,
             maxs=self.cluster_maxs,
             mins=self.cluster_mins,
             scale=self.scale,
             assign_points = self.assign_points,
             use_gpu=self.use_gpu)
            with open(self.outPrefix + "/" + os.path.basename(self.outPrefix) + '_fit.pkl', 'wb') as pickle_file:
                pickle.dump([self.hdb, self.type], pickle_file)


    def load(self, fit_npz, fit_obj):
        '''Load the model from disk. Called from :func:`~loadClusterFit`

        Args:
            fit_npz (dict)
                Fit npz opened with :func:`numpy.load`
            fit_obj (hdbscan.HDBSCAN)
                The saved fit object
        '''
        self.hdb = fit_obj
        self.labels = self.hdb.labels_
        self.n_clusters = fit_npz['n_clusters']
        self.scale = fit_npz['scale']
        self.within_label = fit_npz['within'].item()
        self.between_label = fit_npz['between'].item()
        self.cluster_means = fit_npz['means']
        self.cluster_maxs = fit_npz['maxs']
        self.cluster_mins = fit_npz['mins']
        self.scale = fit_npz['scale']
        if 'use_gpu' in fit_npz.keys():
            self.use_gpu = fit_npz['use_gpu']
        else:
            # Default for backwards compatibility
            self.use_gpu = False
        if 'assign_points' in fit_npz.keys():
            self.assign_points = fit_npz['assign_points']
        else:
            # Default for backwards compatibility
            self.assign_points = True
        self.fitted = True


    def plot(self, X=None, y=None):
        '''Extends :func:`~ClusterFit.plot`

        Write a summary of the fit, and plot the results using
        :func:`PopPUNK.plot.plot_dbscan_results`

        Args:
            X (numpy.array)
                Core and accessory distances
            y (numpy.array)
                Cluster assignments from :func:`~BGMMFit.assign`
        '''
        ClusterFit.plot(self, X)
        # Generate a subsampling if one was not used in the fit
        if not hasattr(self, 'subsampled_X'):
            self.subsampled_X = utils.shuffle(X, random_state=random.randint(1,10000))[0:self.max_samples,]

        non_noise = np.sum(self.labels != -1)
        sys.stderr.write("Fit summary:\n" + "\n".join(["\tNumber of clusters\t" + str(self.n_clusters),
                                                        "\tNumber of datapoints\t" + str(self.subsampled_X.shape[0]),
                                                        "\tNumber of assignments\t" + str(non_noise)]) + "\n\n")

        sys.stderr.write("Scaled component means\n")
        for centre in self.cluster_means:
            sys.stderr.write("\t" + str(centre) + "\n")
        sys.stderr.write("\n")

        # Harmonise scales
        if self.use_gpu:
            import cupy as cp
            self.scale = cp.asarray(self.scale)

        plot_dbscan_results(self.subsampled_X * self.scale,
                            self.assign(self.subsampled_X,
                                        max_batch_size = self.max_batch_size,
                                        no_scale=True,
                                        progress=False,
                                        use_gpu=self.use_gpu),
                            self.n_clusters,
                            self.outPrefix + "/" + os.path.basename(self.outPrefix) + "_dbscan",
                            self.use_gpu)


    def assign(self, X, no_scale = False, progress = True, max_batch_size = 5000, use_gpu = False):
        '''Assign the clustering of new samples using :func:`~PopPUNK.dbscan.assign_samples_dbscan`

        Args:
            X (numpy.array or cupy.array)
                Core and accessory distances
            no_scale (bool)
                Do not scale X
                [default = False]
            progress (bool)
                Show progress bar
                [default = True]
            max_batch_size (int)
                Batch size used for assignments
                [default = 5000]
            use_gpu (bool)
                Use GPU-enabled algorithms for clustering
                [default = False]
        Returns:
            y (numpy.array)
                Cluster assignments by samples
        '''
        if not self.fitted:
            raise RuntimeError("Trying to assign using an unfitted model")
        else:
            if no_scale:
                scale = np.array([1, 1], dtype = X.dtype)
            else:
                scale = self.scale
            if progress:
                sys.stderr.write("Assigning distances with DBSCAN model\n")
            
            # Set block size
            block_size = max_batch_size
            
            if use_gpu:
              y = np.zeros(X.shape[0], dtype=int)
              n_blocks = (X.shape[0] - 1) // block_size + 1
              for block in range(n_blocks):
                  start_index = block*block_size
                  end_index = min((block+1)*block_size-1,X.shape[0])
                  sys.stderr.write("Processing rows " + str(start_index) + " to " + str(end_index) + "\n")
                  # cuml v24.02 always returns numpy therefore make conversion explicit
                  y[start_index:end_index], y_probabilities = cluster.hdbscan.approximate_predict(self.hdb,
                                                                                                  X[start_index:end_index,],
                                                                                                  convert_dtype = True)
                  del y_probabilities
            else:
              y = np.zeros(X.shape[0], dtype=int)
              n_blocks = (X.shape[0] - 1) // block_size + 1
              with SharedMemoryManager() as smm:
                  shm_X = smm.SharedMemory(size = X.nbytes)
                  X_shared_array = np.ndarray(X.shape, dtype = X.dtype, buffer = shm_X.buf)
                  X_shared_array[:] = X[:]
                  X_shared = NumpyShared(name = shm_X.name, shape = X.shape, dtype = X.dtype)

                  shm_y = smm.SharedMemory(size = y.nbytes)
                  y_shared_array = np.ndarray(y.shape, dtype = y.dtype, buffer = shm_y.buf)
                  y_shared_array[:] = y[:]
                  y_shared = NumpyShared(name = shm_y.name, shape = y.shape, dtype = y.dtype)

                  tqdm.set_lock(RLock())
                  process_map(partial(assign_samples,
                              X = X_shared,
                              y = y_shared,
                              model = self,
                              scale = scale,
                              chunk_size = block_size,
                              values = False),
                      range(n_blocks),
                      max_workers=self.threads,
                      chunksize=min(10, max(1, n_blocks // self.threads)),
                      disable=(progress == False))

                  y[:] = y_shared_array[:]

        return y


class RefineFit(ClusterFit):
    '''Class for fits using a triangular boundary and network properties. Inherits from :class:`ClusterFit`.

    Must first run either :func:`~RefineFit.fit` or :func:`~RefineFit.load` before calling
    other functions

    Args:
        outPrefix (str)
            The output prefix used for reading/writing
    '''

    def __init__(self, outPrefix):
        ClusterFit.__init__(self, outPrefix)
        self.type = 'refine'
        self.preprocess = False
        self.within_label = -1
        self.slope = 2
        self.threshold = False
        self.unconstrained = False
        self.assign_points = True

    def fit(self, X, sample_names, model, max_move, min_move, startFile = None, indiv_refine = False,
            unconstrained = False, multi_boundary = 0, score_idx = 0, no_local = False,
            betweenness_sample = betweenness_sample_default, sample_size = None, use_gpu = False):
        '''Extends :func:`~ClusterFit.fit`

        Fits the distances by optimising network score, by calling
        :func:`~PopPUNK.refine.refineFit2D`.

        Fitted parameters are stored in the object.

        Args:
            X (numpy.array)
                The core and accessory distances to cluster. Must be set if
                preprocess is set.
            sample_names (list)
                Sample names in X (accessed by :func:`~PopPUNK.utils.iterDistRows`)
            model (ClusterFit)
                The model fit to refine
            max_move (float)
                Maximum distance to move away from start point
            min_move (float)
                Minimum distance to move away from start point
            startFile (str)
                A file defining an initial fit, rather than one from ``--fit-model``.
                See documentation for format.
                (default = None).
            indiv_refine (str)
                Run refinement for core or accessory distances separately
                (default = None).
            multi_boundary (int)
                Produce cluster output at multiple boundary positions downward
                from the optimum.
                (default = 0).
            unconstrained (bool)
                If True, search in 2D and change the slope of the boundary
            score_idx (int)
                Index of score from :func:`~PopPUNK.network.networkSummary` to use
                [default = 0]
            no_local (bool)
                Turn off the local optimisation step.
                Quicker, but may be less well refined.
            betweenness_sample (int)
                Number of sequences per component used to estimate betweenness using
                a GPU. Smaller numbers are faster but less precise [default = 100]
            sample_size (int)
                Number of nodes to subsample for graph statistic calculation
            use_gpu (bool)
                Whether to use cugraph for graph analyses

        Returns:
            y (numpy.array)
                Cluster assignments of samples in X
        '''
        ClusterFit.fit(self)
        self.scale = np.copy(model.scale)
        self.max_move = max_move
        self.min_move = min_move
        self.unconstrained = unconstrained

        # Get starting point
        model.no_scale()
        if startFile:
            self.mean0, self.mean1, scaled = readManualStart(startFile)
            if not scaled:
                self.mean0 /= self.scale
                self.mean1 /= self.scale
        elif model.type == 'dbscan':
            sys.stderr.write("Initial model-based network construction based on DBSCAN fit\n")
            self.mean0 = model.cluster_means[model.within_label, :]
            self.mean1 = model.cluster_means[model.between_label, :]
        elif model.type == 'bgmm':
            sys.stderr.write("Initial model-based network construction based on Gaussian fit\n")
            self.mean0 = model.means[model.within_label, :]
            self.mean1 = model.means[model.between_label, :]
        else:
            raise RuntimeError("Unrecognised model type")

        # Main refinement in 2D
        scaled_X = X / self.scale
        self.optimal_x, self.optimal_y, optimal_s = \
          refineFit(scaled_X,
                    sample_names,
                    self.mean0,
                    self.mean1,
                    self.scale,
                    self.max_move,
                    self.min_move,
                    slope = 2,
                    score_idx = score_idx,
                    unconstrained = unconstrained,
                    no_local = no_local,
                    num_processes = self.threads,
                    betweenness_sample = betweenness_sample,
                    sample_size = sample_size,
                    use_gpu = use_gpu)
        self.fitted = True

        # Output clusters at more positions if requested
        if multi_boundary > 1:
            sys.stderr.write("Creating multiple boundary fits\n")
            multi_refine(scaled_X,
                        sample_names,
                        self.mean0,
                        self.mean1,
                        self.scale,
                        optimal_s,
                        multi_boundary,
                        self.outPrefix,
                        num_processes = self.threads,
                        betweenness_sample = betweenness_sample,
                        sample_size = sample_size,
                        use_gpu = use_gpu)

        # Try and do a 1D refinement for both core and accessory
        self.core_boundary = self.optimal_x
        self.accessory_boundary = self.optimal_y
        if indiv_refine is not None:
            try:
                for dist_type, slope in zip(['core', 'accessory'], [0, 1]):
                    if indiv_refine == 'both' or indiv_refine == dist_type:
                        sys.stderr.write("Refining " + dist_type + " distances separately\n")
                        # optimise core distance boundary
                        core_boundary, accessory_boundary, s = \
                          refineFit(scaled_X,
                                    sample_names,
                                    self.mean0,
                                    self.mean1,
                                    self.scale,
                                    self.max_move,
                                    self.min_move,
                                    slope = slope,
                                    score_idx = score_idx,
                                    no_local = no_local,
                                    num_processes = self.threads,
                                    betweenness_sample = betweenness_sample,
                                    sample_size = sample_size,
                                    use_gpu = use_gpu)
                        if dist_type == "core":
                            self.core_boundary = core_boundary
                        if dist_type == "accessory":
                            self.accessory_boundary = accessory_boundary
                self.indiv_fitted = True
            except RuntimeError as e:
                print(e)
                sys.stderr.write("Could not separately refine core and accessory boundaries. "
                                 "Using joint 2D refinement only.\n")
        y = self.assign(X)
        return y

    def apply_threshold(self, X, threshold):
        '''Applies a boundary threshold, given by user. Does not run
        optimisation.

        Args:
            X (numpy.array)
                The core and accessory distances to cluster. Must be set if
                preprocess is set.
            threshold (float)
                The value along the x-axis (core distance) at which to
                draw the assignment boundary

        Returns:
            y (numpy.array)
                Cluster assignments of samples in X
        '''
        self.scale = np.array([1, 1], dtype = X.dtype)

        # Blank values to pass to plot
        self.mean0 = None
        self.mean1 = None
        self.min_move = None
        self.max_move = None

        # Sets threshold
        self.core_boundary = threshold
        self.accessory_boundary = np.nan
        self.optimal_x = threshold
        self.optimal_y = np.nan
        self.slope = 0

        # Flags on refine model
        self.fitted = True
        self.threshold = True
        self.indiv_fitted = False
        self.unconstrained = False

        y = self.assign(X)
        return y

    def save(self):
        '''Save the model to disk, as an npz and pkl (using outPrefix).'''
        if not self.fitted:
            raise RuntimeError("Trying to save unfitted model")
        else:
            np.savez(self.outPrefix + "/" + os.path.basename(self.outPrefix) + '_fit.npz',
             intercept=np.array([self.optimal_x, self.optimal_y]),
             core_acc_intercepts=np.array([self.core_boundary, self.accessory_boundary]),
             scale=self.scale,
             indiv_fitted=self.indiv_fitted)
            with open(self.outPrefix + "/" + os.path.basename(self.outPrefix) + '_fit.pkl', 'wb') as pickle_file:
                pickle.dump([None, self.type], pickle_file)


    def load(self, fit_npz, fit_obj):
        '''Load the model from disk. Called from :func:`~loadClusterFit`

        Args:
            fit_npz (dict)
                Fit npz opened with :func:`numpy.load`
            fit_obj (None)
                The saved fit object (not used)
        '''
        self.optimal_x = fit_npz['intercept'].item(0)
        self.optimal_y = fit_npz['intercept'].item(1)
        self.core_boundary = fit_npz['core_acc_intercepts'].item(0)
        self.accessory_boundary = fit_npz['core_acc_intercepts'].item(1)
        self.scale = fit_npz['scale']
        self.fitted = True
        if 'indiv_fitted' in fit_npz:
            self.indiv_fitted = fit_npz['indiv_fitted']
        else:
            self.indiv_fitted = False # historical behaviour for backward compatibility
        if np.isnan(self.optimal_y) and np.isnan(self.accessory_boundary):
            self.threshold = True

        # blank values to pass to plot (used in --use-model)
        self.mean0 = None
        self.mean1 = None
        self.min_move = None
        self.max_move = None

    def plot(self, X, y=None):
        '''Extends :func:`~ClusterFit.plot`

        Write a summary of the fit, and plot the results using
        :func:`PopPUNK.plot.plot_refined_results`

        Args:
            X (numpy.array)
                Core and accessory distances
            y (numpy.array)
                Assignments (unused)
        '''
        ClusterFit.plot(self, X)

        # Subsamples huge plots to save on memory
        max_points = int(0.5*(5000)**2)
        if X.shape[0] > max_points:
            plot_X = utils.shuffle(X, random_state=random.randint(1, 10000))[0:max_points, ]
        else:
            plot_X = X

        plot_refined_results(plot_X, self.assign(plot_X), self.optimal_x, self.optimal_y, self.core_boundary,
            self.accessory_boundary, self.mean0, self.mean1, self.min_move,
            self.max_move, self.scale, self.threshold, self.indiv_fitted, self.unconstrained,
            "Refined fit boundary", self.outPrefix + "/" + os.path.basename(self.outPrefix) + "_refined_fit")


    def assign(self, X, slope=None):
        '''Assign the clustering of new samples

        Args:
            X (numpy.array)
                Core and accessory distances
            slope (int)
                Override self.slope. Default - use self.slope
                Set to 0 for a vertical line, 1 for a horizontal line, or
                2 to use a slope
        Returns:
            y (numpy.array)
                Cluster assignments by samples
        '''
        if not self.fitted:
            raise RuntimeError("Trying to assign using an unfitted model")
        else:
            if slope == None:
                slope = self.slope
            if slope == 2:
                y = poppunk_refine.assignThreshold(X/self.scale, 2, self.optimal_x, self.optimal_y, self.threads)
            elif slope == 0:
                y = poppunk_refine.assignThreshold(X/self.scale, 0, self.core_boundary, 0, self.threads)
            elif slope == 1:
                y = poppunk_refine.assignThreshold(X/self.scale, 1, 0, self.accessory_boundary, self.threads)

        return y

# Wrapper function for LineageFit.__reduce_rank__ to be called by
# multiprocessing threads
def reduce_rank(lower_rank, fit, higher_rank_sparse_mat, n_samples, dtype):
    # Only modify the matrix if the method or rank differs - otherwise save in unmodified form
    if lower_rank==fit.max_search_depth and fit.reciprocal_only is False and fit.count_unique_distances is False:
        fit.__save_sparse__(higher_rank_sparse_mat[2],
                       higher_rank_sparse_mat[0],
                       higher_rank_sparse_mat[1],
                       lower_rank,
                       n_samples,
                       dtype)
    else:
      fit.__reduce_rank__(higher_rank_sparse_mat,
                          lower_rank,
                          n_samples,
                          dtype)

class LineageFit(ClusterFit):
    '''Class for fits using the lineage assignment model. Inherits from :class:`ClusterFit`.

    Must first run either :func:`~LineageFit.fit` or :func:`~LineageFit.load` before calling
    other functions

    Args:
        outPrefix (str)
            The output prefix used for reading/writing
        ranks (list)
            The ranks used in the fit
    '''

    def __init__(self, outPrefix, ranks, max_search_depth, reciprocal_only,
                  count_unique_distances, lineage_resolution, dist_col = None, use_gpu = False):
        ClusterFit.__init__(self, outPrefix)
        self.type = 'lineage'
        self.preprocess = False
        self.max_search_depth = max_search_depth+5 # Set to highest rank by default in main; need to store additional distances
                                                   # when there is redundancy (e.g. reciprocal matching, unique distance counting)
                                                   # or other sequences may be pruned out of the database
        self.nn_dists = None # stores the unprocessed kNN at the maximum search depth
        self.ranks = []
        for rank in sorted(ranks):
            if (rank < 1):
                sys.stderr.write("Rank must be at least 1")
                sys.exit(0)
            else:
                self.ranks.append(int(rank))
        self.lower_rank_dists = {}
        self.reciprocal_only = reciprocal_only
        self.count_unique_distances = count_unique_distances
        self.dist_col = dist_col
        self.resolution = lineage_resolution
        self.use_gpu = use_gpu

    def __save_sparse__(self, data, row, col, rank, n_samples, dtype, is_nn_dist = False):
        '''Save a sparse matrix in coo format
        '''
        if self.use_gpu:
            data = cp.array(data)
            data[data < epsilon] = epsilon
            if is_nn_dist:
                self.nn_dists = cupyx.scipy.sparse.coo_matrix((data, (cp.array(row), cp.array(col))),
                                                    shape=(n_samples, n_samples),
                                                    dtype = dtype)
            else:
                self.lower_rank_dists[rank] = cupyx.scipy.sparse.coo_matrix((data, (cp.array(row), cp.array(col))),
                                                    shape=(n_samples, n_samples),
                                                    dtype = dtype)
        else:
            data = np.array(data)
            data[data < epsilon] = epsilon
            if is_nn_dist:
                self.nn_dists = scipy.sparse.coo_matrix((data, (row, col)),
                                                    shape=(n_samples, n_samples),
                                                    dtype = dtype)
            else:
                self.lower_rank_dists[rank] = scipy.sparse.coo_matrix((data, (row, col)),
                                                    shape=(n_samples, n_samples),
                                                    dtype = dtype)

    def __reduce_rank__(self, higher_rank_sparse_mat, lower_rank, n_samples, dtype):
        '''Lowers the rank of a fit and saves it
        '''
        lower_rank_sparse_mat = \
            poppunk_refine.lowerRank(
                higher_rank_sparse_mat,
                n_samples,
                lower_rank,
                self.reciprocal_only,
                self.count_unique_distances,
                self.resolution,
                self.threads)
        self.__save_sparse__(lower_rank_sparse_mat[2],
                             lower_rank_sparse_mat[0],
                             lower_rank_sparse_mat[1],
                             lower_rank,
                             n_samples,
                             dtype)

    def fit(self, X):
        '''Extends :func:`~ClusterFit.fit`

        Gets assignments by using nearest neigbours.

        Args:
            X (numpy.array)
                The core and accessory distances to cluster. Must be set if
                preprocess is set.

        Returns:
            y (numpy.array)
                Cluster assignments of samples in X
        '''

        ClusterFit.fit(self, X)
        sample_size = int(round(0.5 * (1 + np.sqrt(1 + 8 * X.shape[0]))))
        if (max(self.ranks) >= sample_size or self.max_search_depth >= sample_size):
            sys.stderr.write("Rank and maximum search depth must be less than the number of samples: " + str(sample_size) + "\n")
            sys.exit(0)

        row, col, data = \
            poppunk_refine.get_kNN_distances(
                distMat=pp_sketchlib.longToSquare(distVec=X[:, [self.dist_col]],
                                                  num_threads=self.threads),
                kNN=self.max_search_depth,
                dist_col=self.dist_col,
                num_threads=self.threads
            )
        self.__save_sparse__(data, row, col, self.max_search_depth, sample_size, X.dtype,
                              is_nn_dist = True)

        # Apply filtering of links if requested and extract lower ranks - parallelisation within C++ code
        for rank in self.ranks:
            reduce_rank(
              rank,
              fit=self,
              higher_rank_sparse_mat=(row, col, data),
              n_samples=sample_size,
              dtype=X.dtype
            )

        self.fitted = True
        y = self.assign(min(self.ranks))
        return y

    def save(self):
        '''Save the model to disk, as an npz and pkl (using outPrefix).'''
        if not self.fitted:
            raise RuntimeError("Trying to save unfitted model")
        else:
            scipy.sparse.save_npz(
                    self.outPrefix + "/" + os.path.basename(self.outPrefix) + \
                    '_sparse_dists.npz',
                    self.nn_dists)
            for rank in self.ranks:
                scipy.sparse.save_npz(
                    self.outPrefix + "/" + os.path.basename(self.outPrefix) + \
                    rankFile(rank),
                    self.lower_rank_dists[rank])
            with open(self.outPrefix + "/" + os.path.basename(self.outPrefix) + \
                      '_fit.pkl', 'wb') as pickle_file:
                pickle.dump([[self.ranks,
                                self.max_search_depth,
                                self.reciprocal_only,
                                self.count_unique_distances,
                                self.dist_col],
                                self.type],
                            pickle_file)

    def load(self, fit_npz, fit_obj):
        '''Load the model from disk. Called from :func:`~loadClusterFit`

        Args:
            fit_npz (dict)
                Fit npz opened with :func:`numpy.load`
            fit_obj (sklearn.mixture.BayesianGaussianMixture)
                The saved fit object
        '''
        self.ranks, self.max_search_depth, self.reciprocal_only, self.count_unique_distances, self.dist_col = fit_obj
        self.nn_dists = fit_npz
        self.fitted = True

    def plot(self, X, y = None):
        '''Extends :func:`~ClusterFit.plot`

        Write a summary of the fit, and plot the results using
        :func:`PopPUNK.plot.plot_results` and :func:`PopPUNK.plot.plot_contours`

        Args:
            X (numpy.array)
                Core and accessory distances
            y (any)
                Unused variable for compatibility with other
                plotting functions
        '''
        ClusterFit.plot(self, X)
        for rank in self.ranks:
            if self.use_gpu:
                hist_data = self.lower_rank_dists[rank].get().data
            else:
                hist_data = self.lower_rank_dists[rank].data
            distHistogram(hist_data,
                              rank,
                              self.outPrefix + "/" + os.path.basename(self.outPrefix))

    def assign(self, rank):
        '''Get the edges for the network. A little different from other methods,
        as it doesn't go through the long form distance vector (as coo_matrix
        is basically already in the correct gt format)

        Args:
            rank (int)
                Rank to assign at
        Returns:
            y (list of tuples)
                Edges to include in network
        '''
        if not self.fitted:
            raise RuntimeError("Trying to assign using an unfitted model")
        else:
            y = []
            for row, col in zip(self.lower_rank_dists[rank].row, self.lower_rank_dists[rank].col):
                y.append((row, col))

        return y

    def edge_weights(self, rank):
        '''Get the distances for each edge returned by assign

        Args:
            rank (int)
                Rank assigned at
        Returns:
            weights (list)
                Distance for each assignment
        '''
        if not self.fitted:
            raise RuntimeError("Trying to get weights from an unfitted model")
        else:
            return (self.lower_rank_dists[rank].data)

    def extend(self, qqDists, qrDists):
        '''Update the sparse distance matrix of nearest neighbours after querying

        Args:
            qqDists (numpy or cupy ndarray)
                Two column array of query-query distances
            qqDists (numpy or cupy ndarray)
                Two column array of reference-query distances

        Returns:
            y (list of tuples)
                Edges to include in network
        '''

        # Convert data structures if using GPU
        if self.use_gpu:
            qqDists = cp.array(qqDists)
            qrDists = cp.array(qrDists)

        # Reshape qq and qr dist matrices
        qqSquare = pp_sketchlib.longToSquare(distVec=qqDists[:, [self.dist_col]],
                                             num_threads=self.threads)
        qqSquare[qqSquare < epsilon] = epsilon

        n_ref = self.nn_dists.shape[0]
        n_query = qqSquare.shape[1]
        qrRect = qrDists[:, [self.dist_col]].reshape(n_query, n_ref).T
        qrRect[qrRect < epsilon] = epsilon

        higher_rank = \
          poppunk_refine.extend(
            (self.nn_dists.row, self.nn_dists.col, self.nn_dists.data),
            qqSquare,
            qrRect,
            self.max_search_depth,
            self.threads)

        # Update NN dist associated with model
        self.__save_sparse__(higher_rank[2], higher_rank[0], higher_rank[1],
                             self.max_search_depth, n_ref + n_query, self.nn_dists.dtype,
                             is_nn_dist = True)

        # Apply lower ranks - parallelisation within C++ code
        for rank in self.ranks:
            reduce_rank(
              rank,
              fit=self,
              higher_rank_sparse_mat=higher_rank,
              n_samples=n_ref + n_query,
              dtype=self.nn_dists.dtype
            )
        y = self.assign(min(self.ranks))
        return y

