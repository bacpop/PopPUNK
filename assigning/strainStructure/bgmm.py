'''BGMM using PyMC3 and sklearn'''

# universal
import os
import sys
import argparse
import re
# additional
import networkx as nx
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
from pymc3.math import logsumexp as mc3_logsumexp
import theano
import theano.tensor as tt
from theano.tensor.nlinalg import det, matrix_inverse
from scipy import stats
from scipy import linalg
try:  # SciPy >= 0.19
    from scipy.special import logsumexp as sp_logsumexp
except ImportError:
    from scipy.misc import logsumexp as sp_logsumexp # noqa
from sklearn import preprocessing, utils
# import strainStructure package
import strainStructure

#########################################
# Log likelihood of normal distribution #
#########################################

def logp_normal(mu, tau, value):
    # log probability of individual samples
    k = tau.shape[0]
    delta = lambda mu: value - mu
    logp_ = (-1 / 2.) * (k * tt.log(2 * np.pi) + tt.log(1./det(tau)) + (delta(mu).dot(tau) * delta(mu)).sum(axis=1))
    return logp_

###################################################
# Log likelihood of Gaussian mixture distribution #
###################################################

def logp_gmix(mus, pi, tau, n_samples):
    def logp_(value):
        logps = [tt.log(pi[i]) + logp_normal(mu, tau[i], value)
                 for i, mu in enumerate(mus)]
                 
        return tt.sum(mc3_logsumexp(tt.stacklists(logps)[:, :n_samples], axis=0))
    
    return logp_

##############################################################
# Log likelihood of multivariate normal density distribution #
##############################################################

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

##########################
# Assign query via model #
##########################

def assignQuery(X,refPrefix):
    
    # load model information
    weights = []
    means = []
    covariances = []
    modelFileName = refPrefix+"/"+refPrefix+'_fit.npz'
    try:
        model_npz = np.load(modelFileName)
    except:
        sys.exit("Cannot load model information file "+modelFileName)
    
    # extract information
    weights = model_npz['weights']
    means = model_npz['means']
    covariances = model_npz['covariances']

    # Plot results
    y = assign_samples(X, weights, means, covariances)
    
    return y,weights,means,covariances

#############
# Fit model #   # needs work still
#############

def fit2dMultiGaussian(X,outPrefix):
    
    # set the maximum sampling size
    max_samples = 100000
    
    # Proportion within/between prior
    proportions = np.array([0.001, 0.999])
    strength = 1
    # Location of Gaussians
    within_core = 0
    within_accessory = 0
    between_core = 0.1 # changed from 0.7
    between_accessory = 0.1 # changed from 0.7
    positions_belief = 10**4
    mu_prior = pm.floatX(np.array([[0, 0], [0.006, 0.25]]))
    
    # preprocess scaling
    scaler = preprocessing.MinMaxScaler().fit(X)
    if X.shape[0] > max_samples:
        #X = utils.shuffle(X, random_state=datetime.now())[0:max_samples,]
        subsampled_X = utils.shuffle(X, random_state=random.randint(1,10000))[0:max_samples,]
        scaled_X = scaler.fit_transform(subsampled_X)
    else:
        scaled_X = scaler.fit_transform(X)
    
    # Show clustering
    plt.ioff()
    #    plt.scatter(scaled_X[:, 0], scaled_X[:, 1])
    #    print(X[:,0])
    plt.scatter(X[:,0].flat, X[:,1].flat)
    plt.savefig(outPrefix+"_distanceDistribution.png")
    plt.close()
    
    # fit bgmm model
    #    parameters = (proportions, strength, within_core, within_accessory, between_core, between_accessory, positions_belief)
    parameters = (proportions, strength, mu_prior, positions_belief)
    #    (trace, elbos) = bgmm_model(scaled_X, parameters)
    (trace, elbos) = bgmm_model(X, parameters)  # Nick messing with stuff
    
    # Check convergence and parameters
    plt.plot(elbos)
    plt.savefig(outPrefix+"_elbos.png")
    plt.close()
    pm.plot_posterior(trace, color='LightSeaGreen')
    plt.savefig(outPrefix+"_trace.png")
    plt.close()
    
    # Save model fit
    weights = trace[:]['pi'].mean(axis=0)
    means = []
    covariances = []
    for i in range(mu_prior.shape[0]):
        means.append(trace[:]['mu_%i' %i].mean(axis=0).T)
        covariances.append(trace[:]['cov_%i' %i].mean(axis=0))
    means = np.vstack(means)
    covariances = np.stack(covariances)
    outFileName = outPrefix+"/"+outPrefix+'_fit.npz'
    np.savez(outFileName, weights=weights, means=means, covariances=covariances)

    # Plot results
    y = strainStructure.assign_samples(X, weights, means, covariances)
    strainStructure.plot_results(X, y, means, covariances, 0, outPrefix)
    
    # return output
    return y,weights,means,covariances
