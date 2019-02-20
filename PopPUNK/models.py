# vim: set fileencoding=<utf-8> :
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
from sklearn import utils
import scipy.optimize
from scipy.spatial.distance import euclidean
from scipy import stats

from .plot import plot_scatter

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
from .refine import withinBoundary
from .refine import readManualStart
from .plot import plot_refined_results

def loadClusterFit(pkl_file, npz_file, outPrefix = "", max_samples=100000):
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
    fit_data = np.load(npz_file)

    if fit_type == "bgmm":
        sys.stderr.write("Loading BGMM 2D Gaussian model\n")
        load_obj = BGMMFit(outPrefix, max_samples)
        load_obj.load(fit_data, fit_object)
    elif fit_type == "dbscan":
        sys.stderr.write("Loading DBSCAN model\n")
        load_obj = DBSCANFit(outPrefix, max_samples)
        load_obj.load(fit_data, fit_object)
    elif fit_type == "refine":
        sys.stderr.write("Loading previously refined model\n")
        load_obj = RefineFit(outPrefix)
        load_obj.load(fit_data, fit_object)
    else:
        raise RuntimeError("Undefined model type: " + str(fit_type))

    return load_obj

class ClusterFit:
    '''Parent class for all models used to cluster distances

    Args:
        outPrefix (str)
            The output prefix used for reading/writing
    '''

    def __init__(self, outPrefix):
        self.outPrefix = outPrefix
        self.fitted = False
        self.indiv_fitted = False


    def fit(self, X = None):
        '''Initial steps for all fit functions.

        Creates output directory. If preprocess is set then subsamples passed X
        and draws a scatter plot from result using :func:`~PopPUNK.plot.plot_scatter`.

        Args:
            X (numpy.array)
                The core and accessory distances to cluster. Must be set if
                preprocess is set.

                (default = None)
        '''
        # set output dir
        if not os.path.isdir(self.outPrefix):
            if not os.path.isfile(self.outPrefix):
                os.makedirs(self.outPrefix)
            else:
                sys.stderr.write(self.outPrefix + " already exists as a file! Use a different --output\n")
                sys.exit(1)

        # preprocess subsampling
        if self.preprocess:
            if X.shape[0] > self.max_samples:
                self.subsampled_X = utils.shuffle(X, random_state=random.randint(1,10000))[0:self.max_samples,]
            else:
                self.subsampled_X = np.copy(X)

            # perform scaling
            self.scale = np.amax(self.subsampled_X, axis = 0)
            self.subsampled_X /= self.scale

            # Show clustering
            plot_scatter(self.subsampled_X,
                         self.scale,
                         self.outPrefix + "/" + os.path.basename(self.outPrefix) + "_distanceDistribution",
                         self.outPrefix + " distances")

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
        self.scale = np.array([1, 1])


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
                                                        "\tNumber of components used\t" + str(used_components)]) + "\n")

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
        min_samples = max(int(min_cluster_prop * X.shape[0]), 10)
        min_cluster_size = max(int(0.01 * X.shape[0]), 10)

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

                y = self.assign(X)
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

        non_noise = np.sum(np.where(self.labels != -1))
        sys.stderr.write("Fit summary:\n" + "\n".join(["\tNumber of clusters\t" + str(self.n_clusters),
                                                        "\tNumber of datapoints\t" + str(self.subsampled_X.shape[0]),
                                                        "\tNumber of assignments\t" + str(non_noise)]) + "\n")

        plot_dbscan_results(self.subsampled_X * self.scale, self.assign(self.subsampled_X), self.n_clusters,
            self.outPrefix + "/" + os.path.basename(self.outPrefix) + "_dbscan")


    def assign(self, X):
        '''Assign the clustering of new samples using :func:`~PopPUNK.dbscan.assign_samples_dbscan`

        Args:
            X (numpy.array)
                Core and accessory distances
        Returns:
            y (numpy.array)
                Cluster assignments by samples
        '''
        if not self.fitted:
            raise RuntimeError("Trying to assign using an unfitted model")
        else:
            y = assign_samples_dbscan(X, self.hdb, self.scale)

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
        self.start_point, self.optimal_x, self.optimal_y = refineFit(X/self.scale,
                sample_names, self.start_s, self.mean0, self.mean1, self.max_move, self.min_move,
                slope = 2, no_local = no_local, num_processes = threads)
        self.fitted = True

        # Try and do a 1D refinement for both core and accessory
        self.core_boundary = self.optimal_x
        self.accessory_boundary = self.optimal_y
        if indiv_refine:
            try:
                sys.stderr.write("Refining core and accessory separately\n")

                start_point, self.core_boundary, core_acc = refineFit(X/self.scale, sample_names, self.start_s,
                        self.mean0, self.mean1, self.max_move, self.min_move, slope = 0, no_local = no_local,
                        num_processes = threads)
                start_point, acc_core, self.accessory_boundary = refineFit(X/self.scale, sample_names, self.start_s,
                        self.mean0, self.mean1, self.max_move, self.min_move, slope = 1, no_local = no_local,
                        num_processes = threads)
                self.indiv_fitted = True
            except RuntimeError as e:
                sys.stderr.write("Could not separately refine core and accessory boundaries. "
                                 "Using joint 2D refinement only.\n")

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
             scale=self.scale)
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
        self.indiv_fitted = False # Do not output multiple microreacts


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

        plot_refined_results(X, self.assign(X), self.optimal_x, self.optimal_y, self.core_boundary,
            self.accessory_boundary, self.mean0, self.mean1, self.start_point, self.min_move,
            self.max_move, self.scale, self.indiv_fitted, "Refined fit boundary",
            self.outPrefix + "/" + os.path.basename(self.outPrefix) + "_refined_fit")


    def assign(self, X, slope=None):
        '''Assign the clustering of new samples using :func:`~PopPUNK.refine.withinBoundary`

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
            if slope == 2 or (slope == None and self.slope == 2):
                y = withinBoundary(X/self.scale, self.optimal_x, self.optimal_y)
            elif slope == 0 or (slope == None and self.slope == 0):
                y = withinBoundary(X/self.scale, self.core_boundary, 0, slope=slope)
            elif slope == 1 or (slope == None and self.slope == 1):
                y = withinBoundary(X/self.scale, 0, self.accessory_boundary, slope=slope)

        return y


