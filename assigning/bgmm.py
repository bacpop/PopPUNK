
'''BGMM using PyMC3 and sklearn'''

# imports
%env THEANO_FLAGS=device=cpu,floatX=float32

import theano
import pymc3 as pm
import theano.tensor as tt
from theano.tensor.nlinalg import det, matrix_inverse
import numpy as np
from scipy import linalg
import matplotlib as mpl
import matplotlib.pyplot as plt
from pymc3.math import logsumexp as mc3_logsumexp
from sklearn import mixture
try:  # SciPy >= 0.19
    from scipy.special import logsumexp as sp_logsumexp
except ImportError:
    from scipy.misc import logsumexp as sp_logsumexp # noqa

# modified sklearn GMM functions to predict
def assign_samples(X, weights, means, covars):
    lpr = (log_multivariate_normal_density(X, means, covars) +
               np.log(weights))
    logprob = sp_logsumexp(lpr, axis=1)
    responsibilities = np.exp(lpr - logprob[:, np.newaxis])
    return responsibilities.argmax(axis=1)

def log_multivariate_normal_density(X, means, covars, min_covar=1.e-7):
    """Log probability for full covariance matrices."""
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

# MV normal functions for theano (symbolic)

# Log likelihood of normal distribution
def logp_normal(mu, tau, value):
    # log probability of individual samples
    k = tau.shape[0]
    delta = lambda mu: value - mu
    return (-1 / 2.) * (k * tt.log(2 * np.pi) + tt.log(1./det(tau)) +
                         (delta(mu).dot(tau) * delta(mu)).sum(axis=1))

# Log likelihood of Gaussian mixture distribution
def logp_gmix(mus, pi, tau, n_samples):
    def logp_(value):
        logps = [tt.log(pi[i]) + logp_normal(mu, tau[i], value)
                 for i, mu in enumerate(mus)]

        return tt.sum(mc3_logsumexp(tt.stacklists(logps)[:, :n_samples], axis=0))

    return logp_

# 2D model with minibatches
def bgmm_model(X, model_parameters, minibatch_size = 2000, burnin_it = 25000, sampling_it = 10000, num_samples = 2000):
    # Model priors
    (proportions, strength, mu_prior, positions_belief) = model_parameters
    K = mu_prior.shape[0]

    # Minibatches
    n_samples = X.shape[0]
    X_mini = pm.Minibatch(X, minibatch_size)

    # Model definition
    with pm.Model() as mini_model:
        # Mixture prior
        pi = pm.Dirichlet('pi', a=pm.floatX(strength * proportions), shape=(K,))

        # Mean position prior
        mus = []
        mu_prior_strength = pm.HalfNormal('mu_strength', tau=positions_belief**2)
        for i in range(K):
            mus.append(pm.MvNormal('mu_%i' %i, mu=mu_prior[i,:], cov=mu_prior_strength*np.eye(2), shape=(2,)))

        # Covariance prior
        chol_packed = []
        chol = []
        cov = []
        tau = []
        sd_dist = pm.HalfCauchy.dist(beta=1, shape=2)
        for i in range(K):
            chol_packed.append(pm.LKJCholeskyCov('chol_packed_%i' %i, n=2, eta=2, sd_dist=sd_dist))
            chol.append(pm.expand_packed_triangular(2, chol_packed[i]))
            cov.append(pm.Deterministic('cov_%i' %i, tt.dot(chol[i], chol[i].T)))
            tau.append(matrix_inverse(cov[i]))

        xs = pm.DensityDist('x', logp_gmix(mus, pi, tau, n_samples), observed=X_mini, total_size=n_samples)

    # ADVI - approximate inference
    with mini_model:
        s = theano.shared(pm.floatX(1))
        inference = pm.ADVI(cost_part_grad_scale=s)
        pm.fit(n=burnin_it, method=inference)  # burn-in
        s.set_value(0)
        approx = inference.fit(n=sampling_it)    # inference

    trace = approx.sample(num_samples)
    elbos = -inference.hist

    return(trace, elbos)

# sklearn default
def dirichlet_bgmm(X, max_components = 5, weight_conc = 0.1, mean_precision = 0.1, mean_prior = np.array([0,0])):

    dpgmm = mixture.BayesianGaussianMixture(n_components=max_components,
                                            covariance_type='full',
                                            weight_concentration_prior=weight_conc,
                                            mean_precision_prior=mean_precision,
                                            mean_prior=mean_prior).fit(X)
    return(dpgmm, dpgmm.predict(X))
