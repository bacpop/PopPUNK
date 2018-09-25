# vim: set fileencoding=<utf-8> :
'''BGMM using sklearn'''

# universal
import os
import sys
# additional
import operator
import numpy as np

from scipy import linalg
try:  # SciPy >= 0.19
    from scipy.special import logsumexp as sp_logsumexp
except ImportError:
    from scipy.misc import logsumexp as sp_logsumexp # noqa
from sklearn import mixture


def fit2dMultiGaussian(X, dpgmm_max_K = 2):
    """Main function to fit BGMM model, called from :func:`~PopPUNK.models.BGMMFit.fit`

    Fits the mixture model specified, saves model parameters to a file, and assigns the samples to
    a component. Write fit summary stats to STDERR.

    Args:
        X (np.array)
            n x 2 array of core and accessory distances for n samples.
            This should be subsampled to 100000 samples.
        dpgmm_max_K (int)
            Maximum number of components to use with the EM fit.
            (default = 2)
    Returns:
        dpgmm (sklearn.mixture.BayesianGaussianMixture)
            Fitted bgmm model
    """
    # fit bgmm model
    dpgmm = mixture.BayesianGaussianMixture(n_components = dpgmm_max_K,
                                                n_init = 5,
                                                covariance_type = 'full',
                                                weight_concentration_prior = 0.1,
                                                mean_precision_prior = 0.1,
                                                mean_prior = np.array([0,0])).fit(X)

    return dpgmm


def assign_samples(X, weights, means, covars, scale, values = False):
    """Given distances and a fit will calculate responsibilities and return most
    likely cluster assignment

    Args:
        X (numpy.array)
            n x 2 array of core and accessory distances for n samples
        weights (numpy.array)
            Component weights from :class:`~PopPUNK.models.BGMMFit`
        means (numpy.array)
            Component means from :class:`~PopPUNK.models.BGMMFit`
        covars (numpy.array)
            Component covariances from :class:`~PopPUNK.models.BGMMFit`
        scale (numpy.array)
            Scaling of core and accessory distances from :class:`~PopPUNK.models.BGMMFit`
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


def findWithinLabel(means, assignments, rank = 0):
    """Identify within-strain links

    Finds the component with mean closest to the origin and also akes sure
    some samples are assigned to it (in the case of small weighted
    components with a Dirichlet prior some components are unused)

    Args:
        means (numpy.array)
            K x 2 array of mixture component means
        assignments (numpy.array)
            Sample cluster assignments
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


def log_likelihood(X, weights, means, covars, scale):
    """modified sklearn GMM function predicting distribution membership

    Returns the mixture LL for points X. Used by :func:`~assign_samples` and
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


