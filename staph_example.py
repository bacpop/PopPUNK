#!/usr/bin/env python

import matplotlib as mpl
import pickle
import numpy as np
import networkx as nx
from datetime import datetime
from sklearn import preprocessing, utils

from .bgmm import bgmm_model
from .bgmm import assign_samples
from .plot import plot_results
from .network import build_network
from .network import print_clusters
from .network import print_representatives

max_samples = 100000

################
# Read in data #
################

# (needs to be generalised)
X = np.loadtxt("staph_distances.txt")
scaler = preprocessing.MinMaxScaler().fit(X)
pickle.dump(scaler, open("staph_scaler.pkl", 'wb')) # save scaling for use with possible new data

if X.shape[0] > max_samples:
    X = utils.shuffle(X, random_state=datetime.now())[0:max_samples,]

scaled_X = scaler.fit_transform(X)

# Show clustering
plt.scatter(scaled_X[:, 0], scaled_X[:, 1])
plt.show()

####################
# Input parameters #
####################

# Proportion within/between prior
proportions = np.array([0.1, 0.9])
strength = 1

# Location of Gaussians
within_core = 0
within_accessory = 0
between_core = 0.7
between_accessory = 0.7
positions_belief = 10**3

mu_prior = pm.floatX(np.array([[within_core, within_accessory], [between_core, between_accessory]]))
parameters = (proportions, strength, within_core, within_accessory, between_core, between_accessory, positions_belief)

# for a higher number of componenets, set a higher number of mu priors e.g. for five
# mu_prior = pm.floatX(np.array([[0.1, 0.1], [0.6, 0.4], [0.7, 0.5], [0.67, 0.7], [0.75, 0.9]]))

#############
# Fit model #
#############
(trace, elbos) = bgmm_model(scaled_X, parameters)

# Check convergence and parameters
plt.plot(elbos)
pm.plot_posterior(trace, color='LightSeaGreen')

# Save model fit
weights = trace[:]['pi'].mean(axis=0)
means = []
covariances = []
for i in range(mu_prior.shape[0]):
    means.append(trace[:]['mu_%i' %i].mean(axis=0).T)
    covariances.append(trace[:]['cov_%i' %i].mean(axis=0))
means = np.vstack(means)
covariances = np.stack(covariances)

np.savez('staph_fit.npz', weights=weights, means=means, covariances=covariances)

# Plot results
y = assign_samples(subsampled_X, weights, means, covariances)
plot_results(subsampled_X, y, means, covariances, 0, "Staph 2-component BGMM")

###########
# Network #
###########

# Build network, and choose clusters and references
G = build_network("staph_distances.txt.gz", "staph_scaler.pkl", "staph_fit.npz")
sys.stderr.write("Transitivity: " + str(nx.transitvity(G)))
sys.stderr.write("Desnity: " + str(nx.density(G)))

print_clusters(G, "staph_clusters.txt")
print_representatives(G, "staph_references.txt")
