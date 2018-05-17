'''BGMM using sklearn'''

# universal
import os
import sys
import re
# additional
import numpy as np
import random
import operator
import pickle

from scipy import stats
from scipy import linalg
try:  # SciPy >= 0.19
    from scipy.special import logsumexp as sp_logsumexp
    from scipy.special import gammaln as sp_gammaln
except ImportError:
    from scipy.misc import logsumexp as sp_logsumexp # noqa
    from scipy.misc import gammaln as sp_gammaln
from sklearn import mixture

from .dbscan import assign_samples_dbscan

def log_multivariate_normal_density(X, means, covars, min_covar=1.e-7):
    """Log likelihood of multivariate normal density distribution

    Used to calculate per component Gaussian likelihood in
    :func:`~assign_samples`

    Args:
        X (numpy.array)
            n x 2 array of core and accessory distances for n samples
        means (numpy.array)
            Component means from :func:`~fit2dMultiGaussian`
        covars (numpy.array)
            Component covariances from :func:`~fit2dMultiGaussian`
        min_covar (float)
            Minimum covariance, added when Choleksy decomposition fails
            due to too few observations (default = 1.e-7)

    Returns:
        log_prob (numpy.array)
            An n-vector with the log-likelihoods for each sample being in
            this component
    """
    n_samples, n_dim = X.shape
    nmix = len(means)
    log_prob = np.empty((n_samples, nmix))
    for c, (mu, cv) in enumerate(zip(means, covars)):
        try:
            cv_chol = linalg.cholesky(cv, lower=True)
        except linalg.LinAlgError:
            # The model is most probably stuck in a component with too
            # few observations, we need to reinitialize this components
            try:
                cv_chol = linalg.cholesky(cv + min_covar * np.eye(n_dim),
                                          lower=True)
            except linalg.LinAlgError:
                raise ValueError("'covars' must be symmetric, "
                                 "positive-definite")

        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = linalg.solve_triangular(cv_chol, (X - mu).T, lower=True).T
        log_prob[:, c] = - .5 * (np.sum(cv_sol ** 2, axis=1) +
                                 n_dim * np.log(2 * np.pi) + cv_log_det)

    return log_prob

def assign_samples(X, weights, means, covars, scale, values = False):
    """Given distances and a fit will calculate responsibilities and return most
    likely cluster assignment

    Args:
        X (numpy.array)
            n x 2 array of core and accessory distances for n samples
        weights (numpy.array)
            Component weights from :func:`~fit2dMultiGaussian`
        means (numpy.array)
            Component means from :func:`~fit2dMultiGaussian`
        covars (numpy.array)
            Component covariances from :func:`~fit2dMultiGaussian`
        scale (numpy.array)
            Scaling of core and accessory distances from :func:`~fit2dMultiGaussian`
        values (bool)
            Whether to return the responsibilities, rather than the most
            likely assignment (used for entropy calculation).

            Default is False
    Returns:
        ret_vec (numpy.array)
            An n-vector with the most likely cluster memberships
            or an n by k matrix with the component responsibilities for each sample.
    """
    logprob, lpr = log_likelihood(X, weights, means, covars, scale)
    responsibilities = np.exp(lpr - logprob[:, np.newaxis])

    # Default to return the most likely cluster
    if values == False:
        ret_vec = responsibilities.argmax(axis=1)
    # Can return the actual responsibilities
    else:
        ret_vec = responsibilities

    return ret_vec


def log_likelihood(X, weights, means, covars, scale):
    """modified sklearn GMM function predicting distribution membership

    Returns the mixture LL for points X. Used by :func:`~assign_samples~ and
    :func:`~PopPUNK.plot.plot_contours`

    Args:
        X (numpy.array)
            n x 2 array of core and accessory distances for n samples
        weights (numpy.array)
            Component weights from :func:`~fit2dMultiGaussian`
        means (numpy.array)
            Component means from :func:`~fit2dMultiGaussian`
        covars (numpy.array)
            Component covariances from :func:`~fit2dMultiGaussian`
        scale (numpy.array)
            Scaling of core and accessory distances from :func:`~fit2dMultiGaussian`
    Returns:
        logprob (numpy.array)
            The log of the probabilities under the mixture model
        lpr (numpy.array)
            The components of the log probability from each mixture component
    """

    lpr = (log_multivariate_normal_density(X/scale, means, covars) +
                np.log(weights))
    logprob = sp_logsumexp(lpr, axis=1)

    return(logprob, lpr)


def findWithinLabel(means, assignments, rank = 0):
    """Identify within-strain links

    Finds the component with mean closest to the origin and also akes sure
    some samples are assigned to it (in the case of small weighted
    components with a Dirichlet prior some components are unused)

    Args:
        means (numpy.array)
            K x 2 array of mixture component means from :func:`~fit2dMultiGaussian` or
            :func:`~assignQuery`
        assignments (numpy.array)
            Sample cluster assignments from :func:`~assign_samples`
        rank (int)
            Which label to find, ordered by distance from origin. 0-indexed.

            (default = 0)

    Returns:
        within_label (int)
            The cluster label for the within-strain assignments
    """
    min_dists = {}
    for mixture_component, distance in enumerate(np.apply_along_axis(np.linalg.norm, 1, means)):
        if np.any(assignments == mixture_component):
            min_dists[mixture_component] = distance

    sorted_dists = sorted(min_dists.items(), key=operator.itemgetter(1))
    return(sorted_dists[rank][0])


def assignQuery(X, refPrefix, dbscan):
    """Assign component of query sequences using a previously fitted model

    Args:
        X (numpy.array)
            n x 2 array of core and accessory distances for n samples
        refPrefix (str)
            Prefix of a saved model with '.npz' suffix
    Returns:
        y (numpy.array)
            Cluster assignments for each sample in X
        weights (numpy.array)
            Component weights from :func:`~fit2dMultiGaussian`
        means (numpy.array)
            Component means from :func:`~fit2dMultiGaussian`
        covars (numpy.array)
            Component covariances from :func:`~fit2dMultiGaussian`
        scale (numpy.array)
            Scaling of core and accessory distances from :func:`~fit2dMultiGaussian`
    """
    #### THIS DOCUMENTATION NEEDS UPDATING IF THIS ALTERATION IS KEPT

    from .refine import withinBoundary

    # load model information
    weights = []
    means = []
    covariances = []
    model_type = ""
    refinedModelFileName = refPrefix + "/" + refPrefix + '_refined_fit.npz'
    bgmm_modelFileName = refPrefix + "/" + refPrefix + '_fit.npz'
    dbscan_refinedModelFileName = refPrefix + "/" + refPrefix + '_dbscan_refined_fit.npz'
    dbscan_modelFileName = refPrefix + "/" + refPrefix + '_dbscan_fit.npz'
    try:
        # structure this to use whatever file is available unless
        if os.path.isfile(refinedModelFileName) and dbscan is False:
            model_npz = np.load(dbscan_modelFileName)
            model_type = 'refined'
            sys.stderr.write("Loaded refined BGMM model\n")
        elif os.path.isfile(bgmm_modelFileName) and dbscan is False:
            model_npz = np.load(bgmm_modelFileName)
            model_type = 'bgmm'
            sys.stderr.write("Loaded BGMM model\n")
        elif os.path.isfile(dbscan_refinedModelFileName):
            model_npz = np.load(refinedModelFileName)
            model_type = 'refined'
            sys.stderr.write("Loaded refined DBSCAN model\n")
        elif os.path.isfile(dbscan_modelFileName):
            model_npz = np.load(dbscan_modelFileName)
            model_type = 'dbscan'
            sys.stderr.write("Loaded DBSCAN model\n")

    except:
        sys.stderr.write("Cannot load model from possible information files " + bgmm_modelFileName + ", " + dbscan_modelFileName + ", or " + refinedModelFileName + "\n")
        sys.exit(1)

    # extract information
    scale = model_npz['scale']
    if model_type == 'refined':
        boundary = model_npz['intercept']
        model = (scale, boundary)

        y = withinBoundary(X/scale, boundary[0], boundary[1])

    elif model_type == 'dbscan':
        means = model_npz['means']
        mins = model_npz['mins']
        maxs = model_npz['maxs']
        # load non-numpy model file
        dbscan_pickleFileName = refPrefix + "/" + refPrefix + '_dbscan_fit.pkl'
        if os.path.isfile(dbscan_pickleFileName):
            with open(dbscan_pickleFileName, 'rb') as pickle_file:
                db = pickle.load(pickle_file)
        else:
            sys.stderr.write("Cannot find DBSCAN model file name " + dbscan_pickleFileName + "\n")
            sys.exit(1)
        model = (db, scale, means, mins, maxs)

        # Get assignments using DBSCAN
        y = assign_samples_dbscan(X, db, scale)

    elif model_type == 'bgmm':
        weights = model_npz['weights']
        means = model_npz['means']
        covariances = model_npz['covariances']
        t = model_npz['t']
        model = (scale, weights, means, covariances, t)

        # Get assignments
        y = assign_samples(X, weights, means, covariances, scale, t)

    return y, model, model_type


def fit2dMultiGaussian(X, outPrefix, scale, dpgmm_max_K = 2):
    """Main function to fit model, called from :func:`~__main__.main()`

    Fits the mixture model specified, saves model parameters to a file, and assigns the samples to
    a component. Write fit summary stats to STDERR.

    By default, subsamples :math:`10^5` random distances to fit the model to.

    Args:
        X (np.array)
            n x 2 array of core and accessory distances for n samples
        outPrefix (str)
            Prefix for output files to be saved under
        dpgmm_max_K (int)
            Maximum number of components to use with the EM fit.
            (default = 2)
    Returns:
        y (numpy.array)
            Cluster assignments for each sample in X
        weights (numpy.array)
            Component weights
        means (numpy.array)
            Component means
        covars (numpy.array)
            Component covariances
        scale (numpy.array)
            Scaling of core and accessory distances
    """
    # fit bgmm model
    dpgmm = mixture.BayesianGaussianMixture(n_components = dpgmm_max_K,
                                                n_init = 5,
                                                covariance_type = 'full',
                                                weight_concentration_prior = 0.1,
                                                mean_precision_prior = 0.1,
                                                mean_prior = np.array([0,0])).fit(X)

    avg_entropy = np.mean(np.apply_along_axis(stats.entropy, 1,
        assign_samples(X, weights, means, covariances, scale, values=True)))

    return dpgmm, scale, avg_entropy

