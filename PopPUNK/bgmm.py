'''BGMM using PyMC3 and sklearn'''

# universal
import os
import sys
import argparse
import re
# additional
import numpy as np
import random
import operator
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
from .plot import plot_contours

# for pymc3 fit
scalar_gammaln = GammaLn(scalar.upgrade_to_float, name='scalar_gammaln')
gammaln = tt.Elemwise(scalar_gammaln, name='gammaln')

def logp_t(nu, mu, tau, value):
    """log probability of individual samples for a multivariate t-distribution

    Theano compiled, used for ADVI fitting. Try to return -inf rather than NaN
    for divergent values of logp.

    Not currently used, preferring logp function from pymc3 multivariate t.
    Would be called from :func:`~logp_tmix`

    Args:
        nu (int)
            Degrees of freedom for t-distribution
        mu (pm.MvNormal)
            Position of mean mu
        tau (pm.Matrix)
            Precision matrix tau (inverse of Sigma)
        value (pm.tensor)
            Points to evaluate log- likelihood at

    Returns:
        logp (tt.tensor)
            Log-likelihood of values
    """
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


def logp_tmix(ts, pi, n_samples):
    """log probability of a mixture of t-distributions

    Theano compiled, used for ADVI fitting. Uses passed t-dists
    to calculate logp for each point for each component

    Args:
        ts (list)
            List of component t-distributions
        pi (pm.Dirichlet)
            Weights of each mixture component
        n_samples (int)
            Number of samples being fitted

    Returns:
        logp (tt.tensor)
            Mixture log-likelihood with given parameters
    """

    def logp_(value):
        logps = [tt.log(pi[i]) + ts[i].logp(value)
                 for i, t in enumerate(ts)]

        return tt.sum(mc3_logsumexp(tt.stacklists(logps)[:, :n_samples], axis=0))

    return logp_


def logp_normal(mu, tau, value):
    """log probability of individual samples for a multivariate Gaussian

    Theano compiled, used for ADVI fitting. Used in :func:`~logp_gmix`

    Args:
        mu (pm.MvNormal)
            Position of mean mu
        tau (pm.Matrix)
            Precision matrix tau (inverse of Sigma)
        value (pm.tensor)
            Points to evaluate log- likelihood at

    Returns:
        logp (tt.tensor)
            Log-likelihood of values
    """
    k = tau.shape[0]
    delta = lambda mu: value - mu
    logp_ = (-1 / 2.) * (k * tt.log(2 * np.pi) + tt.log(1./det(tau)) + (delta(mu).dot(tau) * delta(mu)).sum(axis=1))
    return logp_


def logp_gmix(mus, pi, tau, n_samples):
    """log probability of a mixture of Gaussians

    Theano compiled, used for ADVI fitting. Uses a closure of
    :func:`~logp_normal` to calculate logp for each point for each component

    Args:
        mus (list)
            List of means for each component
        pi (pm.Dirichlet)
            Weights of each mixture component
        tau (list)
            List of precision matrix tau (inverse of Sigma) for each component
        n_samples (int)
            Number of samples being fitted

    Returns:
        logp (tt.tensor)
            Mixture log-likelihood with given parameters
    """
    def logp_(value):
        logps = [tt.log(pi[i]) + logp_normal(mu, tau[i], value)
                 for i, mu in enumerate(mus)]

        return tt.sum(mc3_logsumexp(tt.stacklists(logps)[:, :n_samples], axis=0))

    return logp_

def stick_breaking(beta):
    """stick breaking function to define Dirichlet process prior

    Theano compiled, used for ADVI fitting. Used in :func:`~bgmm_model` if
    weight prior = Dirichlet

    Args:
        beta (pm.Beta)
            :math:`\\beta \sim \mathrm{Beta}(1, \\alpha)`.
            Larger :math:`\\alpha` gives more concentrated weights

    Returns:
        weights (tt.tensor)
            Weights distribution (sums to 1)
    """
    portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])
    return beta * portion_remaining

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

def log_multivariate_t_density(X, means, covars, nu = 1, min_covar=1.e-7):
    """Log likelihood of multivariate t-distribution

    Used to calculate per component t-dist likelihood in
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

        norm = (sp_gammaln((nu + n_dim) / 2.) - sp_gammaln(nu / 2.)
                - 0.5 * n_dim * np.log(nu * np.pi))
        inner = - (nu + n_dim) / 2. * np.log1p(np.sum(cv_sol ** 2, axis=1) / nu)
        log_prob[:, c] = norm + inner - cv_log_det

    return log_prob

def assign_samples(X, weights, means, covars, scale, t_dist = False, values = False):
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
        t_dist (bool)
            Indicates the fit was with a mixture of t-distributions
            (default = False).
        values (bool)
            Whether to return the responsibilities, rather than the most
            likely assignment (used for entropy calculation).

            Default is False
    Returns:
        ret_vec (numpy.array)
            An n-vector with the most likely cluster memberships
            or an n by k matrix with the component responsibilities for each sample.
    """
    logprob, lpr = log_likelihood(X, weights, means, covars, scale, t_dist)
    responsibilities = np.exp(lpr - logprob[:, np.newaxis])

    # Default to return the most likely cluster
    if values == False:
        ret_vec = responsibilities.argmax(axis=1)
    # Can return the actual responsibilities
    else:
        ret_vec = responsibilities

    return ret_vec


def log_likelihood(X, weights, means, covars, scale, t_dist = False):
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
        t_dist (bool)
            Indicates the fit was with a mixture of t-distributions
            (default = False).
    Returns:
        logprob (numpy.array)
            The log of the probabilities under the mixture model
        lpr (numpy.array)
            The components of the log probability from each mixture component
    """
    if t_dist:
        lpr = (log_multivariate_t_density(X/scale, means, covars) +
                np.log(weights))
    else:
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

def bgmm_model(X, model_parameters, t_dist = False, minibatch_size = 2000, burnin_it = 25000, sampling_it = 10000, num_samples = 2000):
    """Fits the 2D mixture model using ADVI with minibatches (BGMM)

    Args:
        X (numpy.array)
            n x 2 array of core and accessory distances for n samples
        model_parameters (np.array, float, np.array, float, np.array)
            Priors from :func:`~readPriors`
        t_dist (bool)
            If True, fit a mixture of t-distributions rather than a mixture of Gaussians
            (default = False)
        minibatch_size (int)
            Size of minibatch sample at each iteration
            (default = 2000)
        burnin_it (int)
            Number of burn-in iterations with learning rate = 1
            (default = 25000)
        sampling_it (bool)
            Number of sampling iterations with learning rate = 0
            (default = 1000)
        sampling_it (bool)
            Number of posterior samples to draw from the sampling section of the chain
            (default = 2000)

    Returns:
        trace (pm.trace)
            Trace of the posterior sample
        elbos (pm.history)
            ELBO values across the entire chain (loss function)
    """
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
            pi = pm.Dirichlet('pi', a=pm.floatX(proportions * strength), shape=(K,))

        if t_dist:
            nu = 1
            #nu_raw = pm.Poisson('nu_raw', mu = 0.5, shape = K)
            #nu = pm.Deterministic('nu', nu_raw + 1)

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

            # likelihoods for t-distributions
            if t_dist:
                ts.append(pm.MvStudentT.dist(nu=nu, mu=mus[i], tau=tau[i]))

        if t_dist:
            xs = pm.DensityDist('x', logp_tmix(ts, pi, n_samples), observed=X_mini, total_size=n_samples)
        else:
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

def dirichlet_bgmm(X, max_components = 5, number_runs = 5, weight_conc = 0.1, mean_precision = 0.1, mean_prior = np.array([0,0])):
    """Fits the 2D mixture model using EM (DPGMM)

    A wrapper to the sklearn function :func:`~sklearn.mixture.BayesianGaussianMixture`

    Args:
        X (numpy.array)
            n x 2 array of core and accessory distances for n samples
        max_components (int)
            Maximum number of mixture components to fit.
            (default = 5)
        number_runs (int)
            Number of runs with different starts to try. The run with the best likelihood
            is returned.
            (default = 5)
        weight_conc (float)
            Weight concentration prior (c.f. alpha in :func:`~stick_breaking`)
            (default = 0.1)
        mean_precision (float)
            Mean precision prior (precision of confidence in mean_prior)
            (default = 0.1)
        mean_prior (np.array)
            Prior on mean positions
            (default = [0, 0])

    Returns:
        dpgmm (mixture.BayesianGaussianMixture)
            sklearn BayesianGaussianMixture fitted to X
    """
    dpgmm = mixture.BayesianGaussianMixture(n_components = max_components,
                                            n_init = number_runs,
                                            covariance_type = 'full',
                                            weight_concentration_prior = weight_conc,
                                            mean_precision_prior = mean_precision,
                                            mean_prior = mean_prior).fit(X)
    return(dpgmm)

def assignQuery(X, refPrefix):
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
        t (bool)
            Indicates the fit was with a mixture of t-distributions
            (default = False).
    """
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
    t = model_npz['t']

    # Get assignments
    y = assign_samples(X, weights, means, covariances, scale, t)

    return y, weights, means, covariances, scale, t

def readPriors(priorFile):
    """Read priors for :func:`~bgmm_model` from file

    If no file, or incorrectly formatted, will use default priors (see docs)

    Args:
        priorFile (str)
            Location of file to read from
    Returns:
        proportions (numpy.array)
            Prior on weights
        prop_strength (float)
            Stength of proportions prior
        mu_prior (numpy.array)
            Prior on means
        positions_belief (numpy.array)
            Strength of mu_prior
        dirichlet (bool)
            If True, use Dirichlet process weight prior
    """
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

def fit2dMultiGaussian(X, outPrefix, t_dist = False, priorFile = None, bgmm = False, dpgmm_max_K = 2):
    """Main function to fit model, called from :func:`~__main__.main()`

    Fits the mixture model specified, saves model parameters to a file, and assigns the samples to
    a component. Write fit summary stats to STDERR.

    By default, subsamples :math:`10^6` random distances to fit the model to.

    Args:
        X (np.array)
            n x 2 array of core and accessory distances for n samples
        outPrefix (str)
            Prefix for output files to be saved under
        t_dist (bool)
            If used with bgmm, fit t-distribution mixture rather than Gaussian
        priorFile (str)
            Location of a prior file, used with bgmm
        bgmm (bool)
            Use the ADVI fit rather than EM.
            (default = False)
        dpgmm_max_K (int)
            Maximum number of components to use with the EM fit.
            (default = 2)
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
        t (bool)
            Indicates the fit was with a mixture of t-distributions
            (default = False).
    """
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
    if bgmm:
        parameters = readPriors(priorFile)
        (trace, elbos) = bgmm_model(subsampled_X, parameters, t_dist)

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
        t_dist = False

    # Save model fit
    np.savez(outPrefix + "/" + outPrefix + '_fit.npz',
             weights=weights,
             means=means,
             covariances=covariances,
             scale=scale,
             t=np.array(t_dist, dtype=np.bool_))

    # Plot results
    y = assign_samples(X, weights, means, covariances, scale, t_dist)
    avg_entropy = np.mean(np.apply_along_axis(stats.entropy, 1,
        assign_samples(subsampled_X, weights, means, covariances, scale, t_dist, values=True)))
    used_components = np.unique(y).size

    title = outPrefix + " " + str(len(np.unique(y)))
    outfile = outPrefix + "/" + outPrefix
    if not bgmm:
        title += "-component DPGMM"
        outfile += "_DPGMM_fit"
    elif t_dist:
        title += "-component BtMM"
        outfile += "_BtMM_fit"
    else:
        title +=  "-component BGMM"
        outfile += "_BGMM_fit"
    plot_results(X, y, means, covariances, scale, title, outfile)
    plot_contours(y, weights, means, covariances, title + " assignment boundary", outfile + "_contours", t_dist)

    sys.stderr.write("Fit summary:\n" + "\n".join(["\tAvg. entropy of assignment\t" +  "{:.4f}".format(avg_entropy),
                                                   "\tNumber of components used\t" + str(used_components)])
                                                   + "\n")

    # return output
    return y, weights, means, covariances, scale, t_dist

