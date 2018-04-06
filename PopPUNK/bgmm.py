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
    from theano import scalar
    from theano.scalar.basic_scipy import GammaLn
except ImportError:
    sys.stderr.write("Could not import theano, likely because python was not compiled with shared libraries\n")
    sys.stderr.write("Model fit to reference will not run unless --dpgmm is used\n")
from scipy import stats
from scipy import linalg
try:  # SciPy >= 0.19
    from scipy.special import logsumexp as sp_logsumexp
    from scipy.special import gammaln as sp_gammaln
except ImportError:
    from scipy.misc import logsumexp as sp_logsumexp # noqa
    from scipy.misc import gammaln as sp_gammaln
from sklearn import utils
from sklearn import mixture

from .plot import plot_scatter
from .plot import plot_results

#########################################
# Log likelihood of normal distribution #
#########################################
# for pymc3 fit

scalar_gammaln = GammaLn(scalar.upgrade_to_float, name='scalar_gammaln')
gammaln = tt.Elemwise(scalar_gammaln, name='gammaln')

def logp_t(nu, mu, tau, value):
    # log probability of individual samples
    k = tau.shape[0]

    delta = lambda mu: value - mu
    quaddist = (delta(mu).dot(tau) * delta(mu)).sum(axis=1)
    logdet = tt.log(1./det(tau))

    norm = (gammaln((nu + k) / 2.)
                - gammaln(nu / 2.)
                - 0.5 * k * pm.floatX(np.log(nu * np.pi)))
    inner = - (nu + k) / 2. * tt.log1p(quaddist / nu)
    logp = norm + inner - logdet
    return tt.switch(1 * logp, logp, -np.inf)

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

def logp_tmix(ts, pi, n_samples):
    def logp_(value):
        logps = [tt.log(pi[i]) + ts[i].logp(value)
                 for i, t in enumerate(ts)]

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

def log_multivariate_t_density(X, means, covars, nu = 1):
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

        norm = (sp_gammaln((nu + n_dim) / 2.) - sp_gammaln(nu / 2.)
                - 0.5 * n_dim * np.log(nu * np.pi))
        inner = - (nu + n_dim) / 2. * np.log1p(np.sum(cv_sol ** 2, axis=1) / nu)
        log_prob[:, c] = norm + inner - cv_log_det

    return log_prob

def assign_samples(X, weights, means, covars, scale, t_dist = False, values = False):
    """modified sklearn GMM function predicting distribution membership

    Given distances and a fit will calculate responsibilities and return most
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
    if t_dist:
        lpr = (log_multivariate_t_density(X/scale, means, covars) +
                np.log(weights))
    else:
        lpr = (log_multivariate_normal_density(X/scale, means, covars) +
                np.log(weights))
    logprob = sp_logsumexp(lpr, axis=1)
    responsibilities = np.exp(lpr - logprob[:, np.newaxis])

    # Default to return the most likely cluster
    if values == False:
        ret_vec = responsibilities.argmax(axis=1)
    # Can return the actual responsibilities
    else:
        ret_vec = responsibilities

    return ret_vec

# Identify within-strain links (closest to origin)
# Make sure some samples are assigned, in the case of small weighted components
def findWithinLabel(means, assignments):
    min_dist = None
    for mixture_component, distance in enumerate(np.apply_along_axis(np.linalg.norm, 1, means)):
        if np.any(assignments == mixture_component):
            if min_dist is None or distance < min_dist:
               min_dist = distance
               within_label = mixture_component

    return(within_label)

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

        #nu = pm.Poisson('nu', mu = 1)
        nu = 1

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
        ts = []
        sd_dist = pm.HalfCauchy.dist(beta=1, shape=2)
        for i in range(K):
            chol_packed.append(pm.LKJCholeskyCov('chol_packed_%i' %i, n=2, eta=1, sd_dist=sd_dist))
            chol.append(pm.expand_packed_triangular(2, chol_packed[i]))
            cov.append(pm.Deterministic('cov_%i' %i, tt.dot(chol[i], chol[i].T)))
            tau.append(matrix_inverse(cov[i]))
            ts.append(pm.MvStudentT.dist(nu=nu, mu=mus[i], tau=tau[i]))

        xs = pm.DensityDist('x', logp_tmix(ts, pi, n_samples), observed=X_mini, total_size=n_samples)

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
def dirichlet_bgmm(X, max_components = 5, number_runs = 5, weight_conc = 0.1, mean_precision = 0.1, mean_prior = np.array([0,0])):

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
    scale = model_npz['scale']

    # Get assignments
    y = assign_samples(X, weights, means, covariances, scale)

    return y, weights, means, covariances, scale

# Set priors from file, or provide default
def readPriors(priorFile):
    # default priors
    proportions = np.array([0.001, 0.999])
    prop_strength = 1
    positions_belief = 10**4
    mu_prior = pm.floatX(np.array([[0, 0], [0.7, 0.7]]))
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
    scale = np.amax(subsampled_X, axis = 0)
    subsampled_X /= scale

    # Show clustering
    plot_scatter(subsampled_X, outPrefix + "/" + outPrefix + "_distanceDistribution", outPrefix + " distances")

    # fit bgmm model
    if not dpgmm:
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
        means = np.vstack(means)
        covariances = np.stack(covariances)
    else:
        dpgmm = dirichlet_bgmm(subsampled_X, max_components = dpgmm_max_K)
        weights = dpgmm.weights_
        means = dpgmm.means_
        covariances = dpgmm.covariances_

    # Save model fit
    np.savez(outPrefix + "/" + outPrefix + '_fit.npz', weights=weights, means=means, covariances=covariances, scale=scale)

    # Plot results
    y = assign_samples(X, weights, means, covariances, scale, t_dist = True)
    avg_entropy = np.mean(np.apply_along_axis(stats.entropy, 1,
        assign_samples(subsampled_X, weights, means, covariances, scale, t_dist = True, values=True)))
    used_components = np.unique(y).size

    title = outPrefix + " " + str(len(np.unique(y)))
    if dpgmm:
        title += "-component DPGMM"
    else:
        title +=  "-component BGMM"
    plot_results(X, y, means, covariances, scale, title, outPrefix + "/" + outPrefix + "_GMM_fit")

    sys.stderr.write("Fit summary:\n" + "\n".join(["\tAvg. entropy of assignment\t" +  "{:.4f}".format(avg_entropy),
                                                   "\tNumber of components used\t" + str(used_components)])
                                                   + "\n")

    # return output
    return y, weights, means, covariances, scale
