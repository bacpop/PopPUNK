# universal
import os
import sys
# additional
import numpy as np
import random
import operator
import pickle
from sklearn import utils
import scipy.optimize
from scipy.spatial.distance import euclidean

# BGMM
from .bgmm import fit2dMultiGaussian
from .bgmm import assign_samples
from .plot import plot_results
from .plot import plot_contours

# DBSCAN
from .dbscan import fitDbScan
from .dbscan import assign_samples_dbscan
from .plot import plot_dbscan_results

# refine
from .refine import refineFit
from .refine import likelihoodBoundary
from .refine import withinBoundary
from .plot import plot_refined_results

#TODO write docstrings

def loadClusterFit(pkl_file, npz_file):
    fit_type, fit_object = pickle.load(pkl_file)
    fit_data = np.loadz(npz_file)
    if fit_type == "bgmm":
        load_obj = BGMMFit(".")
        load_obj.load(fit_data, fit_object)
    elif fit_type = "dbscan":
        load_obj = DBSCANFit(".")
        load_obj.load(fit_data, fit_object)
    elif fit_type = "refine":
        load_obj = RefineFit(".")
        load_obj.load(fit_data)

    return load_obj

class ClusterFit:
    '''all clustering methods'''

    def __init__(self, outPrefix_):
        self.outPrefix = outPrefix_
        self.fitted = False


    def fit(self):
        # set output dir
        if not os.path.isdir(self.outPrefix):
            if not os.path.isfile(self.outPrefix):
                os.makedirs(self.outPrefix)
            else:
                sys.stderr.write(self.outPrefix + " already exists as a file! Use a different --output\n")
                sys.exit(1)

        # preprocess scaling
        if self.preprocess:
            if X.shape[0] > self.max_samples:
                self.subsampled_X = utils.shuffle(X, random_state=random.randint(1,10000))[0:self.max_samples,]
            else:
                self.subsampled_X = np.copy(X)
            self.scale = np.amax(self.subsampled_X, axis = 0)
            self.subsampled_X /= self.scale

            # Show clustering
            plot_scatter(self.subsampled_X, self.outPrefix + "/" + self.outPrefix + "_distanceDistribution",
                    self.outPrefix + " distances")


class BGMMFit(ClusterFit):
    def __init__(self, outPrefix, max_samples = 100000)
        ClusterFit.__init__(self, outPrefix)
        self.type = 'bgmm'
        self.preprocess = True
        self.max_samples_ = max_samples


    def fit(self, X, max_components):
        ClusterFit.fit(self)
        self.dpgmm, self.scale, self.entropy = fit2dMultiGaussian(subsampled_X, self.outPrefix_, self.scale, max_components)
        self.weights = self.dpgmm.weights_
        self.means = self.dpgmm.means_
        self.covariances = self.dpgmm.covariances_

        y = self.assign(X)
        self.within_label = findWithinLabel(self.means, y)

        self.fitted = True
        return y


    def save(self):
        if not self.fitted:
            raise RuntimeError("Trying to save unfitted model")
        else:
            np.savez(self.outPrefix + "/" + self.outPrefix + '_fit.npz',
             weights=self.weights,
             means=self.means,
             covariances=self.covariances,
             within=self.within_label,
             scale=self.scale)
            pickle.dump((self.dpgmm, self.type), self.outPrefix + "/" + self.outPrefix + '_fit.pkl')


    def load(self, fit_npz, fit_obj):
        self.dpgmm = fit_obj
        self.weights = fit_npz['weights']
        self.means = fit_npz['means']
        self.covariances = fit_npz['covariances']
        self.scale = fit_npz['scale']
        self.within_label = np.asscalar(fit_npz['within'])
        self.fitted = True


    def plot(self, X, y):
        if not self.fitted:
            raise RuntimeError("Trying to plot unfitted model")
        else:
            used_components = np.unique(y).size
            sys.stderr.write("Fit summary:\n" + "\n".join(["\tAvg. entropy of assignment\t" +  "{:.4f}".format(self.entropy),
                                                           "\tNumber of components used\t" + str(used_components)]) + "\n")

            title = self.outPrefix + " " + str(len(np.unique(y))) + "-component DPGMM"
            outfile = self.outPrefix + "/" + self.outPrefix + "_DPGMM_fit"

            plot_results(X, y, self.means, self.covariances, self.scale, title, outfile)
            plot_contours(y, self.weights, self.means, self.covariances, title + " assignment boundary", outfile + "_contours")


    def assign(self, X, values = False):
        if not self.fitted:
            raise RuntimeError("Trying to assign using an unfitted model")
        else:
            y = assign_samples(X, self.weights, self.means, self.covariances, self.scale, values)

        return y


class DBSCANFit(ClusterFit):
    def __init__(self, outPrefix, max_samples = 1000000)
        ClusterFit.__init__(self, outPrefix)
        self.type = 'dbscan'
        self.preprocess = True
        self.max_samples_ = max_samples
        self.within_label = -1


    def fit(self, X, threads):
        ClusterFit.fit(self)
        self.hdb, self.labels, self.n_clusters = fitDbScan(self.subsampled_X, self.outPrefix_, threads)

        y = self.assign(X)

        self.fitted = True
        return y


    def save(self):
        if not self.fitted:
            raise RuntimeError("Trying to save unfitted model")
        else:
            np.savez(self.outPrefix + "/" + self.outPrefix + '_fit.npz',
             n_clusters=self.n_clusters,
             within=self.within_label,
             scale=self.scale)
            pickle.dump((self.hdb, self.type), self.outPrefix + "/" + self.outPrefix + '_fit.pkl')


    def load(self, fit_npz, fit_obj):
        self.hdb = fit_obj
        self.labels = self.hdb.labels_
        self.n_clusters = fit_npz['n_clusters']
        self.scale = fit_npz['scale']
        self.within_label = np.asscalar(fit_npz['within'])
        self.fitted = True


    def plot(self):
        if not self.fitted:
            raise RuntimeError("Trying to plot unfitted model")
        else:
            sys.stderr.write("Fit summary:\n" + "\n".join(["\tNumber of clusters\t" +  + str(self.n_clusters_),
                                                           "\tNumber of datapoints\t" + str(self.subsampled_X.shape[0]),
                                                           "\tNumber of assignments\t" + str(len(self.labels)])+ "\n")

            plot_dbscan_results(self. subsampled_X, self. labels, self.n_clusters,
                self.outPrefix + "/" + self.outPrefix + "_dbscan")


    def assign(self, X):
        if not self.fitted:
            raise RuntimeError("Trying to assign using an unfitted model")
        else:
            y = assign_samples_dbscan(X, self.hdb, self.scale)

        return y


class RefineFit(ClusterFit):
    def __init__(self, outPrefix)
        ClusterFit.__init__(self, outPrefix)
        self.type = 'refine'
        self.preprocess = False


    def fit(self, X, sample_names, model, max_move, min_move, startFile = None, no_local = False, threads = 1):
        ClusterFit.fit(self)
        self.scale = model.scale
        self.max_move = max_move
        self.min_move = min_move

        # Get starting point
        assignment = model.assign(X)
        if startFile:
            mean0, mean1, start_s = readManualStart(startFile)
        elif self.type == 'dbscan':
            sys.stderr.write("Initial model-based network construction based on DBSCAN fit\n")

            # get within strain cluster
            max_cluster_num = self.labels.max()
            cluster_means = np.full((self.n_clusters,2),0.0,dtype=float)
            cluster_mins = np.full((self.n_clusters,2),0.0,dtype=float)
            cluster_maxs = np.full((self.n_clusters,2),0.0,dtype=float)

            for i in range(self.max_cluster_num+1):
                cluster_means[i,] = [np.mean(self.subsampled_X[self.labels==i,0]),np.mean(self.subsampled_X[self.labels==i,1])]
                cluster_mins[i,] = [np.min(self.subsampled_X[self.labels==i,0]),np.min(self.subsampled_X[self.labels==i,1])]
                cluster_maxs[i,] = [np.max(self.subsampled_X[self.labels==i,0]),np.max(self.subsampled_X[self.labels==i,1])]

            within_label = findWithinLabel(cluster_means, assignment)
            between_label = findBetweenLabel(cluster_means, assignment, within_label)

            self.mean0 = cluster_means[within_label, :]
            self.mean1 = cluster_means[between_label, :]
            max0 = cluster_maxs[within_label, :]
            min1 = cluster_mins[between_label, :]
            core_s = (max(max0[0],min1[0]) - self.mean0[0]) / self.mean1[0]
            acc_s = (max(max0[1],min1[1]) - self.mean0[1]) / self.mean1[1]
            start_s = 0.5*(core_s+acc_s)

        elif self.type ='bgmm':
            sys.stderr.write("Initial model-based network construction based on Gaussian fit\n")

            within_label = findWithinLabel(self.means, assignment)
            between_label = findWithinLabel(self.means, assignment, 1)

            # Straight line between dist 0 centre and dist 1 centre
            # Optimize to find point of decision boundary along this line as starting point
            self.mean0 = means[within_label, :]
            self.mean1 = means[between_label, :]
            start_s = scipy.optimize.brentq(likelihoodBoundary, 0, euclidean(self.mean0, self.mean1),
                             args = (model, mean0, mean1, within_label, between_label))
        else:
            raise RuntimeError("Unrecognised model type")

        self.start_point, self.optimal_x, self.optimal_y =
            refineFit(X, sample_names, model.assign(X), self.max_move, self.min_move, no_local, threads)
        y = self.assign(X)

        self.fitted = True
        return y


    def save(self):
        if not self.fitted:
            raise RuntimeError("Trying to save unfitted model")
        else:
            np.savez(self.outPrefix + "/" + self.outPrefix + '_fit.npz',
             intercept=np.array([optimal_x, optimal_y]),
             scale=self.scale)
            pickle.dump((None, self.type), self.outPrefix + "/" + self.outPrefix + '_fit.pkl')


    def load(self, fit_npz, fit_obj):
        self.optimal_x = np.asscalar(fit_npz['intercept'][0])
        self.optimal_y = np.asscalar(fit_npz['intercept'][1])
        self.scale = fit_npz['scale']
        self.fitted = True


    def plot(self, X):
        if not self.fitted:
            raise RuntimeError("Trying to plot unfitted model")
        else:
            plot_refined_results(X, self.assign(X), self.optimal_x, self.optimal_y,
                self.mean0, self.mean1, self.start_point, self.min_move, self.max_move, self.scale,
                "Refined fit boundary", outPrefix + "/" + outPrefix + "_refined_fit")


    def assign(self, X):
        if not self.fitted:
            raise RuntimeError("Trying to assign using an unfitted model")
        else:
            y = withinBoundary(X, self.optimal_x, self.optimal_y)

        return y


