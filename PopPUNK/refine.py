'''Refine mixture model using network properties'''

# universal
import os
import sys
import argparse
import re
# additional
import numpy as np
import sharedmem
from functools import partial
import networkx as nx
import scipy.optimize
from scipy.spatial.distance import euclidean

from .mash import iterDistRows
from .bgmm import assign_samples
from .bgmm import findWithinLabel
from .network import constructNetwork
from .network import networkSummary
from .plot import plot_refined_results

def refineFit(distMat, outPrefix, sample_names, assignment, weights, means,
        covariances, scale, t_dist, max_move, min_move, no_local = False, num_processes = 1):
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
    start_s = scipy.optimize.brentq(likelihoodBoundary, 0, euclidean(mean0, mean1),
                     args = (weights, means, covariances, np.array([1, 1]), t_dist, mean0, mean1, within_label, between_label))
    start_point = transformLine(start_s, mean0, mean1)
    sys.stderr.write("Decision boundary starts at (" + "{:.2f}".format(start_point[0])
                      + "," + "{:.2f}".format(start_point[1]) + ")\n")

    # Boundary is left of line normal to this point and first line
    gradient = (mean1[1] - mean0[1]) / (mean1[0] - mean0[0])
    x_max, y_max = decisionBoundary(start_point, gradient)
    boundary_assignments = withinBoundary(distMat, x_max, y_max)
    G = constructNetwork(sample_names, sample_names, boundary_assignments, -1)

    # Optimize boundary - grid search for global minimum
    sys.stderr.write("Trying to optimise score globally\n")
    shared_dists = sharedmem.copy(distMat)
    s_range = np.linspace(-min_move, max_move, num = 40)
    with sharedmem.MapReduce(np = num_processes) as pool:
        global_s = pool.map(partial(newNetwork,
            sample_names=sample_names, distMat=shared_dists, start_point=start_point, mean1=mean1, gradient=gradient),
            s_range)

    # Local optimisation around global optimum
    min_idx = np.argmin(np.array(global_s))
    if min_idx > 0 and min_idx < len(s_range) - 1 and not no_local:
        sys.stderr.write("Trying to optimise score locally\n")
        local_s = scipy.optimize.minimize_scalar(newNetwork,
                        bounds=[s_range[min_idx-1], s_range[min_idx+1]],
                        method='Bounded', options={'disp': True},
                        args = (sample_names, distMat, start_point, mean1, gradient))
        optimised_s = local_s.x
    else:
        optimised_s = s_range[min_idx]

    optimal_x, optimal_y = decisionBoundary(transformLine(optimised_s, start_point, mean1), gradient)
    if optimal_x <= 0 or optimal_x >= 1 or optimal_y <= 0 or optimal_y >= 1:
        sys.stderr.write("Optimisation failed: produced a boundary outside of allowed range\n")
        sys.exit(1)

    # Make network from new optimal boundary
    boundary_assignments = withinBoundary(distMat, optimal_x, optimal_y)
    G = constructNetwork(sample_names, sample_names, boundary_assignments, -1)

    # ALTERNATIVE - use a single network
    # Move boundary along in steps, and find those samples which have changed
    # Use remove_edges/add_edges with index k lookup (n total) to find sample IDs
    # https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
    # i = n - 2 - int(sqrt(-8*k + 4*n*(n-1)-7)/2.0 - 0.5)
    # j = k + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2

    # Save new fit
    if not os.path.isdir(outPrefix):
        os.makedirs(outPrefix)
    plot_refined_results(distMat, boundary_assignments, optimal_x, optimal_y, [1, 1],
            "Refined fit boundary", outPrefix + "/" + outPrefix + "_refined_fit")
    np.savez(outPrefix + "/" + outPrefix + '_refined_fit.npz',
             intercept=np.array([optimal_x, optimal_y]),
             scale=scale,
             boundary=np.array(True, dtype=np.bool_))

    # return new network
    return G

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
    x = mean0[0] + s * (1/np.sqrt(1+tan_theta))
    y = mean0[1] + s * (tan_theta/np.sqrt(1+tan_theta))

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

def newNetwork(s, sample_names, distMat, start_point, mean1, gradient):
    new_intercept = transformLine(s, start_point, mean1)
    x_max, y_max = decisionBoundary(new_intercept, gradient)
    boundary_assignments = withinBoundary(distMat, x_max, y_max)
    G = constructNetwork(sample_names, sample_names, boundary_assignments, -1, summarise = False)
    (components, density, transitivity, score) = networkSummary(G)
    return(-score)
