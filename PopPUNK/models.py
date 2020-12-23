# vim: set fileencoding=<utf-8> :
# Copyright 2018-2020 John Lees and Nick Croucher

'''Classes used for model fits'''

# universal
import os
import sys
# additional
import numpy as np
import random
import operator
import pickle
import shutil
import re
from sklearn import utils
import scipy.optimize
from scipy.spatial.distance import euclidean
from scipy import stats
from scipy.sparse import coo_matrix, bmat, find

import pp_sketchlib

# BGMM
from .bgmm import fit2dMultiGaussian
from .bgmm import assign_samples
from .bgmm import findWithinLabel
from .plot import plot_results
from .plot import plot_contours

# DBSCAN
from .dbscan import fitDbScan
from .dbscan import assign_samples_dbscan
from .dbscan import findBetweenLabel
from .dbscan import evaluate_dbscan_clusters
from .plot import plot_dbscan_results

# refine
from .refine import refineFit
from .refine import likelihoodBoundary
from .refine import readManualStart
from .plot import plot_refined_results

# lineage
from .plot import distHistogram
epsilon = 1e-10

# Format for rank fits
def rankFile(rank):
    return('_rank' + str(rank) + '_fit.npz')

def loadClusterFit(pkl_file, npz_file, outPrefix = "", max_samples = 100000):
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
    '''
    with open(pkl_file, 'rb') as pickle_obj:
        fit_object, fit_type = pickle.load(pickle_obj)

    if fit_type == 'lineage':
        # Can't save multiple sparse matrices to the same file, so do some
        # file name processing
        fit_data = {}
        for rank in fit_object[0]:
            fit_file = os.path.basename(pkl_file)
            prefix = re.match(r"^(.+)_fit\.pkl$", fit_file)
            rank_file = os.path.dirname(pkl_file) + "/" + \
                        prefix.group(1) + rankFile(rank)
            fit_data[rank] = scipy.sparse.load_npz(rank_file)
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
        sys.stderr.write("Loading previously lineage cluster model\n")
        load_obj = LineageFit(outPrefix, fit_object[0])
    else:
        raise RuntimeError("Undefined model type: " + str(fit_type))

    load_obj.load(fit_data, fit_object)
    return load_obj

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

    def __init__(self, outPrefix, max_samples = 100000):
        ClusterFit.__init__(self, outPrefix)
        self.type = 'bgmm'
        self.preprocess = True
        self.max_samples = max_samples


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

        y = self.assign(X)
        self.within_label = findWithinLabel(self.means, y)
        self.between_label = findWithinLabel(self.means, y, 1)
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
        self.within_label = np.asscalar(fit_npz['within'])
        self.between_label = np.asscalar(fit_npz['between'])
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

        avg_entropy = np.mean(np.apply_along_axis(stats.entropy, 1, self.assign(self.subsampled_X, values = True)))
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
        plot_contours(y, self.weights, self.means, self.covariances, title + " assignment boundary", outfile + "_contours")


    def assign(self, X, values = False):
        '''Assign the clustering of new samples using :func:`~PopPUNK.bgmm.assign_samples`

        Args:
            X (numpy.array)
                Core and accessory distances
            values (bool)
                Return the responsibilities of assignment rather than most likely cluster
        Returns:
            y (numpy.array)
                Cluster assignments or values by samples
        '''
        if not self.fitted:
            raise RuntimeError("Trying to assign using an unfitted model")
        else:
            y = assign_samples(X, self.weights, self.means, self.covariances, self.scale, values)

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

    def __init__(self, outPrefix, max_samples = 100000):
        ClusterFit.__init__(self, outPrefix)
        self.type = 'dbscan'
        self.preprocess = True
        self.max_samples = max_samples


    def fit(self, X, max_num_clusters, min_cluster_prop):
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

        Returns:
            y (numpy.array)
                Cluster assignments of samples in X
        '''
        ClusterFit.fit(self, X)

        # DBSCAN parameters
        cache_out = "./" + self.outPrefix + "_cache"
        min_samples = max(int(min_cluster_prop * self.subsampled_X.shape[0]), 10)
        min_cluster_size = max(int(0.01 * self.subsampled_X.shape[0]), 10)

        indistinct_clustering = True
        while indistinct_clustering and min_cluster_size >= min_samples and min_samples >= 10:
            self.hdb, self.labels, self.n_clusters = fitDbScan(self.subsampled_X, min_samples, min_cluster_size, cache_out)
            self.fitted = True # needed for predict

            # Test whether model fit contains distinct clusters
            if self.n_clusters > 1 and self.n_clusters <= max_num_clusters:
                # get within strain cluster
                self.max_cluster_num = self.labels.max()
                self.cluster_means = np.full((self.n_clusters,2),0.0,dtype=float)
                self.cluster_mins = np.full((self.n_clusters,2),0.0,dtype=float)
                self.cluster_maxs = np.full((self.n_clusters,2),0.0,dtype=float)

                for i in range(self.max_cluster_num+1):
                    self.cluster_means[i,] = [np.mean(self.subsampled_X[self.labels==i,0]),np.mean(self.subsampled_X[self.labels==i,1])]
                    self.cluster_mins[i,] = [np.min(self.subsampled_X[self.labels==i,0]),np.min(self.subsampled_X[self.labels==i,1])]
                    self.cluster_maxs[i,] = [np.max(self.subsampled_X[self.labels==i,0]),np.max(self.subsampled_X[self.labels==i,1])]

                y = self.assign(self.subsampled_X, no_scale=True)
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
        else:
            shutil.rmtree(cache_out)

        y = self.assign(X)
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
             scale=self.scale)
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
        self.within_label = np.asscalar(fit_npz['within'])
        self.between_label = np.asscalar(fit_npz['between'])
        self.cluster_means = fit_npz['means']
        self.cluster_maxs = fit_npz['maxs']
        self.cluster_mins = fit_npz['mins']
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

        plot_dbscan_results(self.subsampled_X * self.scale,
                            self.assign(self.subsampled_X, no_scale=True),
                            self.n_clusters,
                            self.outPrefix + "/" + os.path.basename(self.outPrefix) + "_dbscan")


    def assign(self, X, no_scale = False):
        '''Assign the clustering of new samples using :func:`~PopPUNK.dbscan.assign_samples_dbscan`

        Args:
            X (numpy.array)
                Core and accessory distances
            no_scale (bool)
                Do not scale X

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
            y = assign_samples_dbscan(X, self.hdb, scale)

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

    def fit(self, X, sample_names, model, max_move, min_move, startFile = None, indiv_refine = False,
            no_local = False, threads = 1):
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
            indiv_refine (bool)
                Run refinement for core and accessory distances separately

                (default = False).
            no_local (bool)
                Turn off the local optimisation step.
                Quicker, but may be less well refined.
            num_processes (int)
                Number of threads to use in the global optimisation step.

                (default = 1)
        Returns:
            y (numpy.array)
                Cluster assignments of samples in X
        '''
        ClusterFit.fit(self)
        self.scale = np.copy(model.scale)
        self.max_move = max_move
        self.min_move = min_move

        # Get starting point
        assignment = model.assign(X)
        model.no_scale()
        if startFile:
            self.mean0, self.mean1, self.start_s = readManualStart(startFile)
        elif model.type == 'dbscan':
            sys.stderr.write("Initial model-based network construction based on DBSCAN fit\n")

            self.mean0 = model.cluster_means[model.within_label, :]
            self.mean1 = model.cluster_means[model.between_label, :]
            max0 = model.cluster_maxs[model.within_label, :]
            min1 = model.cluster_mins[model.between_label, :]
            core_s = (max(max0[0],min1[0]) - self.mean0[0]) / self.mean1[0]
            acc_s = (max(max0[1],min1[1]) - self.mean0[1]) / self.mean1[1]
            self.start_s = 0.5*(core_s+acc_s)

        elif model.type == 'bgmm':
            sys.stderr.write("Initial model-based network construction based on Gaussian fit\n")

            # Straight line between dist 0 centre and dist 1 centre
            # Optimize to find point of decision boundary along this line as starting point
            self.mean0 = model.means[model.within_label, :]
            self.mean1 = model.means[model.between_label, :]
            try:
                self.start_s = scipy.optimize.brentq(likelihoodBoundary, 0, euclidean(self.mean0, self.mean1),
                             args = (model, self.mean0, self.mean1, model.within_label, model.between_label))
            except ValueError:
                sys.stderr.write("Could not find start point for refinement; intial model fit likely bad\n"
                                 "Try using --manual-start\n")
                sys.exit(1)
        else:
            raise RuntimeError("Unrecognised model type")

        # Main refinement in 2D
        self.start_point, self.optimal_x, self.optimal_y, self.min_move, self.max_move = refineFit(X/self.scale,
                sample_names, self.start_s, self.mean0, self.mean1, self.max_move, self.min_move,
                slope = 2, no_local = no_local, num_processes = threads)
        self.fitted = True

        # Try and do a 1D refinement for both core and accessory
        self.core_boundary = self.optimal_x
        self.accessory_boundary = self.optimal_y
        if indiv_refine:
            try:
                sys.stderr.write("Refining core and accessory separately\n")
                # optimise core distance boundary
                start_point, self.core_boundary, core_acc, self.min_move, self.max_move = refineFit(X/self.scale,
                sample_names, self.start_s, self.mean0, self.mean1, self.max_move, self.min_move,
                slope = 0, no_local = no_local,num_processes = threads)
                # optimise accessory distance boundary
                start_point, acc_core, self.accessory_boundary, self.min_move, self.max_move = refineFit(X/self.scale,
                sample_names, self.start_s,self.mean0, self.mean1, self.max_move, self.min_move, slope = 1,
                no_local = no_local, num_processes = threads)
                self.indiv_fitted = True
            except RuntimeError as e:
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
        self.start_point = None
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
        self.optimal_x = np.asscalar(fit_npz['intercept'][0])
        self.optimal_y = np.asscalar(fit_npz['intercept'][1])
        self.core_boundary = np.asscalar(fit_npz['core_acc_intercepts'][0])
        self.accessory_boundary = np.asscalar(fit_npz['core_acc_intercepts'][1])
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
        self.start_point = None
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
            self.accessory_boundary, self.mean0, self.mean1, self.start_point, self.min_move,
            self.max_move, self.scale, self.threshold, self.indiv_fitted, "Refined fit boundary",
            self.outPrefix + "/" + os.path.basename(self.outPrefix) + "_refined_fit")


    def assign(self, X, slope=None, cpus=1):
        '''Assign the clustering of new samples

        Args:
            X (numpy.array)
                Core and accessory distances
            slope (int)
                Override self.slope. Default - use self.slope

                Set to 0 for a vertical line, 1 for a horizontal line, or
                2 to use a slope
            cpus (int)
                Number of threads to use
        Returns:
            y (numpy.array)
                Cluster assignments by samples
        '''
        if not self.fitted:
            raise RuntimeError("Trying to assign using an unfitted model")
        else:
            if slope == 2 or (slope == None and self.slope == 2):
                y = pp_sketchlib.assignThreshold(X/self.scale, 2, self.optimal_x, self.optimal_y, cpus)
            elif slope == 0 or (slope == None and self.slope == 0):
                y = pp_sketchlib.assignThreshold(X/self.scale, 0, self.core_boundary, 0, cpus)
            elif slope == 1 or (slope == None and self.slope == 1):
                y = pp_sketchlib.assignThreshold(X/self.scale, 1, 0, self.accessory_boundary, cpus)

        return y


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

    def __init__(self, outPrefix, ranks):
        ClusterFit.__init__(self, outPrefix)
        self.type = 'lineage'
        self.preprocess = False
        self.ranks = []
        for rank in sorted(ranks):
            if (rank < 1):
                sys.stderr.write("Rank must be at least 1")
                sys.exit(0)
            else:
                self.ranks.append(int(rank))


    def fit(self, X, accessory, threads):
        '''Extends :func:`~ClusterFit.fit`

        Gets assignments by using nearest neigbours.

        Args:
            X (numpy.array)
                The core and accessory distances to cluster. Must be set if
                preprocess is set.
            accessory (bool)
                Use accessory rather than core distances
            threads (int)
                Number of threads to use

        Returns:
            y (numpy.array)
                Cluster assignments of samples in X
        '''
        ClusterFit.fit(self, X)
        sample_size = int(round(0.5 * (1 + np.sqrt(1 + 8 * X.shape[0]))))
        if (max(self.ranks) >= sample_size):
            sys.stderr.write("Rank must be less than the number of samples")
            sys.exit(0)

        if accessory:
            self.dist_col = 1
        else:
            self.dist_col = 0

        self.nn_dists = {}
        for rank in self.ranks:
            row, col, data = \
                pp_sketchlib.sparsifyDists(
                    pp_sketchlib.longToSquare(X[:, [self.dist_col]], threads),
                    0,
                    rank,
                    threads
                )
            data = [epsilon if d < epsilon else d for d in data]
            self.nn_dists[rank] = coo_matrix((data, (row, col)),
                                             shape=(sample_size, sample_size),
                                             dtype = X.dtype)

        self.fitted = True

        y = self.assign(min(self.ranks))
        return y

    def save(self):
        '''Save the model to disk, as an npz and pkl (using outPrefix).'''
        if not self.fitted:
            raise RuntimeError("Trying to save unfitted model")
        else:
            for rank in self.ranks:
                scipy.sparse.save_npz(
                    self.outPrefix + "/" + os.path.basename(self.outPrefix) + \
                    rankFile(rank),
                    self.nn_dists[rank])
            with open(self.outPrefix + "/" + os.path.basename(self.outPrefix) + \
                      '_fit.pkl', 'wb') as pickle_file:
                pickle.dump([[self.ranks, self.dist_col], self.type], pickle_file)

    def load(self, fit_npz, fit_obj):
        '''Load the model from disk. Called from :func:`~loadClusterFit`

        Args:
            fit_npz (dict)
                Fit npz opened with :func:`numpy.load`
            fit_obj (sklearn.mixture.BayesianGaussianMixture)
                The saved fit object
        '''
        self.ranks, self.dist_col = fit_obj
        self.nn_dists = fit_npz
        self.fitted = True

    def plot(self, X):
        '''Extends :func:`~ClusterFit.plot`

        Write a summary of the fit, and plot the results using
        :func:`PopPUNK.plot.plot_results` and :func:`PopPUNK.plot.plot_contours`

        Args:
            X (numpy.array)
                Core and accessory distances
        '''
        ClusterFit.plot(self, X)
        for rank in self.ranks:
            distHistogram(self.nn_dists[rank].data,
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
            for row, col in zip(self.nn_dists[rank].row, self.nn_dists[rank].col):
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
            return (self.nn_dists[rank].data)

    def extend(self, qqDists, qrDists):
        # Reshape qq and qr dist matrices
        qqSquare = pp_sketchlib.longToSquare(qqDists[:, [self.dist_col]], 1)
        qqSquare[qqSquare < epsilon] = epsilon

        n_ref = self.nn_dists[self.ranks[0]].shape[0]
        n_query = qqSquare.shape[1]
        qrRect = qrDists[:, [self.dist_col]].reshape(n_query, n_ref)
        qrRect[qrRect < epsilon] = epsilon

        for rank in self.ranks:
            # Add the matrices together to make a large square matrix
            full_mat = bmat([[self.nn_dists[rank], qrRect.transpose()],
                             [qrRect,              qqSquare]],
                            format = 'csr',
                            dtype = self.nn_dists[rank].dtype)

            # Reapply the rank to each row, using sparse matrix functions
            data = []
            row = []
            col = []
            for row_idx in range(full_mat.shape[0]):
                sample_row = full_mat.getrow(row_idx)
                dist_row, dist_col, dist = find(sample_row)
                dist = [epsilon if d < epsilon else d for d in dist]
                dist_idx_sort = np.argsort(dist)

                # Identical to C++ code in matrix_ops.cpp:sparsify_dists
                neighbours = 0
                prev_val = -1
                for sort_idx in dist_idx_sort:
                    if row_idx == dist_col[sort_idx]:
                        continue
                    new_val = abs(dist[sort_idx] - prev_val) < epsilon
                    if (neighbours < rank or new_val):
                        data.append(dist[sort_idx])
                        row.append(row_idx)
                        col.append(dist_col[sort_idx])

                        if not new_val:
                            neighbours += 1
                            prev_val = data[-1]
                    else:
                        break

            self.nn_dists[rank] = coo_matrix((data, (row, col)),
                                    shape=(full_mat.shape[0], full_mat.shape[0]),
                                    dtype = self.nn_dists[rank].dtype)

        y = self.assign(min(self.ranks))
        return y

