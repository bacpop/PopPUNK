'''BGMM using PyMC3 and sklearn'''

# universal
import os
import sys
import argparse
import re
# additional
import numpy as np
import random
import matplotlib.pyplot as plt
try:
    import pymc3 as pm
    from pymc3.math import logsumexp as mc3_logsumexp
    import theano
    import theano.tensor as tt
    from theano.tensor.nlinalg import det, matrix_inverse
except ImportError:
    sys.stderr.write("Could not import theano, likely because python was not compiled with shared libraries\n")
    sys.stderr.write("Model fit to reference will not run unless --dpgmm is used\n")
from scipy import stats
from scipy import linalg
try:  # SciPy >= 0.19
    from scipy.special import logsumexp as sp_logsumexp
except ImportError:
    from scipy.misc import logsumexp as sp_logsumexp # noqa
from sklearn import utils
from sklearn import mixture

from .plot import plot_scatter
from .plot import plot_results

#########################################
# Log likelihood of normal distribution #
#########################################
# for pymc3 fit

def logp_normal(mu, tau, value):
    # log probability of individual samples
    k = tau.shape[0]
    delta = lambda mu: value - mu
    logp_ = (-1 / 2.) * (k * tt.log(2 * np.pi) + tt.log(1./det(tau)) + (delta(mu).dot(tau) * delta(mu)).sum(axis=1))
    return logp_

###################################################
# Log likelihood of Gaussian mixture distribution #
###################################################
# for pymc3 fit

def logp_gmix(mus, pi, tau, n_samples):
    def logp_(value):
        logps = [tt.log(pi[i]) + logp_normal(mu, tau[i], value)
                 for i, mu in enumerate(mus)]

        return tt.sum(mc3_logsumexp(tt.stacklists(logps)[:, :n_samples], axis=0))

    return logp_

# stick breaking for Dirichlet prior
def stick_breaking(beta):
    portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])
    return beta * portion_remaining

##############################################################
# Log likelihood of multivariate normal density distribution #
##############################################################
# for sample assignment below

def log_multivariate_normal_density(X, means, covars, min_covar=1.e-7):
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

#####################################################################
# modified sklearn GMM functions predicting distribution membership #
#####################################################################

def assign_samples(X, weights, means, covars):
    lpr = (log_multivariate_normal_density(X, means, covars) +
           np.log(weights))
    logprob = sp_logsumexp(lpr, axis=1)
    responsibilities = np.exp(lpr - logprob[:, np.newaxis])
    return responsibilities.argmax(axis=1)

#############################
# 2D model with minibatches #
#############################

def bgmm_model(X, model_parameters, minibatch_size = 2000, burnin_it = 25000, sampling_it = 10000, num_samples = 2000):
    # Model priors
    (proportions, strength, mu_prior, positions_belief, dirichlet) = model_parameters
    K = mu_prior.shape[0]

    # Minibatches
    n_samples = X.shape[0]
    X_mini = pm.Minibatch(X, minibatch_size)

    # Model definition
    with pm.Model() as mini_model:
        # Mixture prior
        if dirichlet == True:
            alpha = pm.Gamma('alpha', 1, 2)
            beta = pm.Beta('beta', 1, alpha, shape=K)
            pi = pm.Deterministic('pi', stick_breaking(beta))
        else:
            pi = pm.Dirichlet('pi', a=pm.floatX(proportions / strength), shape=(K,))

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
            chol_packed.append(pm.LKJCholeskyCov('chol_packed_%i' %i, n=2, eta=1, sd_dist=sd_dist))
            chol.append(pm.expand_packed_triangular(2, chol_packed[i]))
            cov.append(pm.Deterministic('cov_%i' %i, tt.dot(chol[i], chol[i].T)))
            tau.append(matrix_inverse(cov[i]))

        xs = pm.DensityDist('x', logp_gmix(mus, pi, tau, n_samples), observed=X_mini, total_size=n_samples)

    # ADVI - approximate inference
    with mini_model:
        sys.stderr.write("Running model burn-in\n")
        s = theano.shared(pm.floatX(1))
        inference = pm.ADVI(cost_part_grad_scale=s)
        pm.fit(n=burnin_it, method=inference)

        sys.stderr.write("Running model fit\n")
        s.set_value(0)
        approx = inference.fit(n=sampling_it)

    trace = approx.sample(num_samples)
    elbos = -inference.hist

    return(trace, elbos)

# Dirichlet BGMM through EM
def dirichlet_bgmm(X, max_components = 5, number_runs = 3, weight_conc = 0.1, mean_precision = 0.1, mean_prior = np.array([0,0])):

    dpgmm = mixture.BayesianGaussianMixture(n_components = max_components,
                                            n_init = number_runs,
                                            covariance_type = 'full',
                                            weight_concentration_prior = weight_conc,
                                            mean_precision_prior = mean_precision,
                                            mean_prior = mean_prior).fit(X)
    return(dpgmm)

##########################
# Assign query via model #
##########################

def assignQuery(X, refPrefix):

    # load model information
    weights = []
    means = []
    covariances = []
    modelFileName = refPrefix + "/" + refPrefix + '_fit.npz'
    try:
        model_npz = np.load(modelFileName)
    except:
        sys.stderr.write("Cannot load model information file " + modelFileName + "\n")
        sys.exit(1)

    # extract information
    weights = model_npz['weights']
    means = model_npz['means']
    covariances = model_npz['covariances']

    # Get assignments
    y = assign_samples(X, weights, means, covariances)

    return y, weights, means, covariances

# Set priors from file, or provide default
def readPriors(priorFile = None):
    # default priors
    proportions = np.array([0.001, 0.999])
    prop_strength = 1
    positions_belief = 10**4
    mu_prior = pm.floatX(np.array([[0, 0], [0.006, 0.25]]))
    dirichlet = False

    # Overwrite defaults if provided
    if priorFile is not None:
        with open(priorFile, 'r') as priors:
            for line in priors:
                (param, value) = line.rstrip().split()
                if param == 'proportions':
                    if value == "Dirichlet":
                        dirichlet = True
                        continue
                    prop_read = []
                    for pop_val in value.split(','):
                        prop_read.append(float(pop_val))
                    proportions = np.array(prop_read)

                elif param == 'prop_strength':
                    prop_strength = float(value)

                elif param == 'positions':
                    pos_read = []
                    for pos_val in value.split(';'):
                        point_read = []
                        for point_val in pos_val.split(','):
                            point_read.append(float(point_val))
                        pos_read.append(point_read)
                    mu_prior = pm.floatX(np.array(pos_read))

                elif param == 'pos_strength':
                    positions_belief = float(value)

                else:
                    sys.stderr.write('Ignoring prior line for ' + param + '\n')

    if dirichlet == False and proportions.shape[0] != mu_prior.shape[0]:
        sys.stderr.write('The number of components must be equal in the proportion and position priors\n')
        sys.exit(1)

    return proportions, prop_strength, mu_prior, positions_belief, dirichlet

#############
# Fit model #
#############

def fit2dMultiGaussian(X, outPrefix, priorFile = None, dpgmm = False, dpgmm_max_K = 2):

    # set output dir
    if not os.path.isdir(outPrefix):
        if not os.path.isfile(outPrefix):
            os.makedirs(outPrefix)
        else:
            sys.stderr.write(outPrefix + " already exists as a file! Use a different --output\n")
            sys.exit(1)

    # set the maximum sampling size
    max_samples = 100000

    # preprocess scaling
    if X.shape[0] > max_samples:
        subsampled_X = utils.shuffle(X, random_state=random.randint(1,10000))[0:max_samples,]
    else:
        subsampled_X = np.copy(X)

    # Show clustering
    plot_scatter(subsampled_X, outPrefix + "/" + outPrefix + "_distanceDistribution", outPrefix + " distances")

    # fit bgmm model
    if not dpgmm:
        scale = np.amax(subsampled_X, axis = 0)
        subsampled_X /= scale

        parameters = readPriors(priorFile)
        (trace, elbos) = bgmm_model(subsampled_X, parameters)

        # Check convergence and parameters
        plt.plot(elbos)
        plt.savefig(outPrefix + "/" + outPrefix + "_elbos.png")
        plt.close()
        pm.plot_posterior(trace, color='LightSeaGreen')
        plt.savefig(outPrefix + "/" + outPrefix + "_posterior.png")
        plt.close()

        weights = trace[:]['pi'].mean(axis=0)
        means = []
        covariances = []
        for i in range(parameters[2].shape[0]):
            means.append(trace[:]['mu_%i' %i].mean(axis=0).T)
            covariances.append(trace[:]['cov_%i' %i].mean(axis=0))
        means = np.vstack(means) * scale
        covariances = scale * np.stack(covariances) * scale
    else:
        dpgmm = dirichlet_bgmm(subsampled_X, max_components = dpgmm_max_K)
        weights = dpgmm.weights_
        means = dpgmm.means_
        covariances = dpgmm.covariances_

    # Save model fit
    np.savez(outPrefix + "/" + outPrefix + '_fit.npz', weights=weights, means=means, covariances=covariances)

    # Plot results
    y = assign_samples(X, weights, means, covariances)

    title = outPrefix + " " + str(len(np.unique(y)))
    if dpgmm:
        title += "-component DPGMM"
    else:
        title +=  "-component BGMM"
    plot_results(X, y, means, covariances, title, outPrefix + "/" + outPrefix + "_GMM_fit")

    # return output
    return y, weights, means, covariances
