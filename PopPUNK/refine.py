'''Refine mixture model using network properties'''

# universal
import os
import sys
import argparse
import re
# additional
import numpy as np
import math
import networkx as nx
from scipy.optimize import brentq
from scipy.spatial.distance import euclidean

from .mash import iterDistRows
from .bgmm import assign_samples
from .bgmm import findWithinLabel
from .network import constructNetwork

def refineFit(distMat, sample_names, assignment, weights, means, covariances, scale, t_dist):
    """Try to refine a fit by maximising a network score based on transitivity and density.

    Iteratively move the decision boundary to do this, using starting point from existing model.

    Args:
        distMat (numpy.array)
            n x 2 array of core and accessory distances for n samples
        sample_names (list)
            List of query sequence labels
        assignment (numpy.array)
            Labels of most likely cluster assignment from :func:`~PopPUNK.bgmm.assign_samples`
        weights (numpy.array)
            Component weights from :func:`~PopPUNK.bgmm.fit2dMultiGaussian`
        means (numpy.array)
            Component means from :func:`~PopPUNK.bgmm.fit2dMultiGaussian`
        covariances (numpy.array)
            Component covariances from :func:`~PopPUNK.bgmm.fit2dMultiGaussian`
        scale (numpy.array)
            Scaling of core and accessory distances from :func:`~PopPUNK.bgmm.fit2dMultiGaussian`
        t_dist (bool)
            Indicates the fit was with a mixture of t-distributions
    Returns:
        G (networkx.Graph)
            The resulting refined network
    """
    sys.stderr.write("Initial model-based network construction\n")
    distMat /= scale # Deal with scale at start
    within_label = findWithinLabel(means, assignment)
    between_label = findWithinLabel(means, assignment, 1)
    G = constructNetwork(sample_names, sample_names, assignment, within_label)

    # Straight line between dist 0 centre and dist 1 centre
    # Optimize to find point of decision boundary along this line as starting point
    sys.stderr.write("Initial boundary based network construction\n")
    mean0 = means[within_label, :]
    mean1 = means[between_label, :]
    start_s = brentq(likelihoodBoundary, 0, euclidean(mean0, mean1),
                     args = (weights, means, covariances, np.array([1, 1]), t_dist, mean0, mean1, within_label, between_label))
    start_point = transformLine(start_s, mean0, mean1)

    # Boundary is left on line normal to this point and first line
    gradient = (mean1[1] - mean0[1]) / (mean1[0] - mean0[0])
    x_max, y_max = decisionBoundary(start_point, gradient)
    boundary_assignments = withinBoundary(distMat, x_max, y_max)
    G = constructNetwork(sample_names, sample_names, boundary_assignments, -1)

    # Move boundary along line
    new_intercept = transformLine(-0.4, start_point, mean1)
    x_max, y_max = decisionBoundary(new_intercept, gradient)
    updated_assignments = withinBoundary(distMat, x_max, y_max)
    change_connections = []
    for old, new, (ref, query) in zip(boundary_assignments, updated_assignments, iterDistRows(sample_names, sample_names, self=True)):
        if new == 1 and old == -1:
            change_connections.append((ref, query))
    G.remove_edges_from(change_connections)

    for s in np.linspace(-0.4, 0.1, 200):
        new_intercept = transformLine(s, start_point, mean1)
        x_max, y_max = decisionBoundary(new_intercept, gradient)
        updated_assignments = withinBoundary(distMat, x_max, y_max)
        change_connections = []
        for old, new, (ref, query) in zip(boundary_assignments, updated_assignments, iterDistRows(sample_names, sample_names, self=True)):
            if new == -1 and old == 1:
                change_connections.append((ref, query))
        G.add_edges_from(change_connections)
        boundary_assignments = updated_assignments
        print("\t".join([str(s), str(nx.number_connected_components(G)), str(nx.density(G)), str(nx.transitivity(G))]))

    # to optimize, could just reconstruct network each time?

    #Use interval bisection to maximize score
        #Need to ensure score is monotonic (try plotting first)
    #Use different save mode for boundary assignment

    # also return new fit
    return G, x_max, y_max

def likelihoodBoundary(s, weights, means, covars, scale, t_dist, start, end, within, between):
    """Wrapper function around :func:`~PopPUNK.bgmm.fit2dMultiGaussian` so that it can
    go into a root-finding function for probabilities between components

    Args:
        s (float)
            Distance along line from mean0
        weights (numpy.array)
            Component weights from :func:`~PopPUNK.bgmm.fit2dMultiGaussian`
        means (numpy.array)
            Component means from :func:`~PopPUNK.bgmm.fit2dMultiGaussian`
        covariances (numpy.array)
            Component covariances from :func:`~PopPUNK.bgmm.fit2dMultiGaussian`
        scale (numpy.array)
            Scaling of core and accessory distances from :func:`~PopPUNK.bgmm.fit2dMultiGaussian`
        t_dist (bool)
            Indicates the fit was with a mixture of t-distributions
        start (numpy.array)
            The co-ordinates of the centre of the within-strain distribution
        end (numpy.array)
            The co-ordinates of the centre of the between-strain distribution
    Returns:
        responsibility (float)
            The difference between responsibilities of assignment to the within component
            and the between assignment
    """
    X = transformLine(s, start, end).reshape(1, -1)
    responsibilities = assign_samples(X, weights, means, covars, scale, t_dist, values = True)
    return(responsibilities[0, within] - responsibilities[0, between])

def transformLine(s, mean0, mean1):
    """Return x and y co-ordinates for traversing along a line between mean0 and mean1, parameterised by
    a single scalar distance s from the start point mean0.

    Args:
        s (float)
            Distance along line from mean0
        mean0 (numpy.array)
            Start position of line (x0, y0)
        mean1 (numpy.array)
            End position of line (x1, y1)
    Returns:
        x (float)
            The Cartesian x-coordinate
        y (float)
            The Cartesian y-coordinate
    """
    tan_theta = (mean1[1] - mean0[1]) / (mean1[0] - mean0[0])
    x = mean0[0] + s * (1/math.sqrt(1+tan_theta))
    y = mean0[1] + s * (tan_theta/math.sqrt(1+tan_theta))

    return np.array([x, y])

def decisionBoundary(intercept, gradient):
    # Returns the co-ordinates of the triangle the decision boundary forms
    x = intercept[0] + intercept[1] * gradient
    y = intercept[1] + intercept[0] / gradient
    return(x, y)

def withinBoundary(dists, x_max, y_max):
    # See https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
    # x_max and y_max from decisionBoundary
    in_tri = lambda row: row[0]*row[1] - (x_max-row[0])*(y_max-row[1])
    boundary_test = np.apply_along_axis(in_tri, 1, dists)
    return(np.sign(boundary_test))


