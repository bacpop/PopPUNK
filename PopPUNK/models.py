# universal
import os
import sys
import re
# additional
import numpy as np
import random
import operator
import pickle
from sklearn import utils

from .bgmm import fit2dMultiGaussian
from .bgmm import assign_samples
from .plot import plot_results
from .plot import plot_contours

#TODO write docstrings

def loadClusterFit(npz_file):
    fit_data = np.loadz(npz_file)
    if fit_data['model_type'] == "bgmm":
        fit_obj = BGMMFit(".", 2)
        fit_obj.load(fit_data)
    elif #TODO etc

    return fit_obj

class ClusterFit:
    '''all clustering methods'''

    def __init__(self, outPrefix):
        self.outPrefix = outPrefix
        self.fitted = False

    def fit(self):
        # set output dir
        if not os.path.isdir(outPrefix):
            if not os.path.isfile(outPrefix):
                os.makedirs(outPrefix)
            else:
                sys.stderr.write(outPrefix + " already exists as a file! Use a different --output\n")
                sys.exit(1)

    def preprocess_X(self, X):
        # preprocess scaling
        if X.shape[0] > self.max_samples:
            subsampled_X = utils.shuffle(X, random_state=random.randint(1,10000))[0:self.max_samples,]
        else:
            subsampled_X = np.copy(X)
        scale = np.amax(subsampled_X, axis = 0)
        subsampled_X /= scale

        return subsampled_X, scale

    def assignQuery(self):
        # See assignQuery in bgmm.py

class BGMMFit(ClusterFit):
    def __init__(self, outPrefix, max_components)
        ClusterFit.__init__(self, outPrefix)
        self.type = np.array('bgmm', dtype=string)
        self.K_max = max_components
        self.max_samples = 100000

    def fit(self, X):
        ClusterFit.fit(self)
        subsampled_X, self.scale = self.preprocess_X(X)
        self.dpgmm, self.scale, self.entropy = fit2dMultiGaussian(subsampled_X, outPrefix, self.K_max)
        self.weights = self.dpgmm.weights_
        self.means = self.dpgmm.means_
        self.covariances = self.dpgmm.covariances_

        y = self.assign(X)

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
             scale=self.scale,
             model_type=self.type)

    def load(self, fit_npz):
        self.weights = fit_npz['weights']
        self.means = fit_npz['means']
        self.covariances = fit_npz['covariances']
        self.scale = fit_npz['scale']
        self.fitted = True

    def plot(self, X, y):
        if not self.fitted:
            raise RuntimeError("Trying to plot unfitted model")
        else:
            used_components = np.unique(y).size
            sys.stderr.write("Fit summary:\n" + "\n".join(["\tAvg. entropy of assignment\t" +  "{:.4f}".format(self.entropy),
                                               "\tNumber of components used\t" + str(used_components)])
                                                   + "\n")

            title = self.outPrefix + " " + str(len(np.unique(y))) + "-component DPGMM"
            outfile = self.outPrefix + "/" + self.outPrefix + "_DPGMM_fit"

            plot_results(X, y, self.means, self.covariances, self.scale, title, outfile)
            plot_contours(y, self.weights, self.means, self.covariances, title + " assignment boundary", outfile + "_contours")


    def assign(self, X):
        if not self.fitted:
            raise RuntimeError("Trying to assign using an unfitted model")
        else:
            y = assign_samples(X, self.weights, self.means, self.covariances, self.scale)

        return y


class DBSCANFit(ClusterFit):
    def __init__(self):
        self.max_samples = 1000000

class RefineFit(ClusterFit):
