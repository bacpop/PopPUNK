# vim: set fileencoding=<utf-8> :
# Copyright 2018-2020 John Lees and Nick Croucher

'''Refine mixture model using network properties'''

# universal
import os
import sys
# additional
from functools import partial
import numpy as np
import scipy.optimize
import collections
try:
    from multiprocessing import Pool, shared_memory
    from multiprocessing.managers import SharedMemoryManager
    NumpyShared = collections.namedtuple('NumpyShared', ('name', 'shape', 'dtype'))
except ImportError as e:
    sys.stderr.write("This version of PopPUNK requires python v3.8 or higher\n")
    sys.exit(0)
import pp_sketchlib
import graph_tool.all as gt

from .network import constructNetwork
from .network import networkSummary

def refineFit(distMat, sample_names, start_s, mean0, mean1,
        max_move, min_move, slope = 2, no_local = False, num_processes = 1):
    """Try to refine a fit by maximising a network score based on transitivity and density.

    Iteratively move the decision boundary to do this, using starting point from existing model.

    Args:
        distMat (numpy.array)
            n x 2 array of core and accessory distances for n samples
        sample_names (list)
            List of query sequence labels
        start_s (float)
            Point along line to start search
        mean0 (numpy.array)
            Start point to define search line
        mean1 (numpy.array)
            End point to define search line
        max_move (float)
            Maximum distance to move away from start point
        min_move (float)
            Minimum distance to move away from start point
        slope (int)
            Set to 0 for a vertical line, 1 for a horizontal line, or
            2 to use a slope
        no_local (bool)
            Turn off the local optimisation step.
            Quicker, but may be less well refined.
        num_processes (int)
            Number of threads to use in the global optimisation step.

            (default = 1)
    Returns:
        start_point (tuple)
            (x, y) co-ordinates of starting point
        optimal_x (float)
            x-coordinate of refined fit
        optimal_y (float)
            y-coordinate of refined fit
    """
    sys.stderr.write("Initial boundary based network construction\n")
    start_point = transformLine(start_s, mean0, mean1)
    sys.stderr.write("Decision boundary starts at (" + "{:.2f}".format(start_point[0])
                      + "," + "{:.2f}".format(start_point[1]) + ")\n")

    # calculate distance between start point and means if none is supplied
    if min_move is None:
        min_move = ((mean0[0] - start_point[0])**2 + (mean0[1] - start_point[1])**2)**0.5
    if max_move is None:
        max_move = ((mean1[0] - start_point[0])**2 + (mean1[1] - start_point[1])**2)**0.5

    # Boundary is left of line normal to this point and first line
    gradient = (mean1[1] - mean0[1]) / (mean1[0] - mean0[0])

    # ALTERNATIVE - use a single network
    # Move boundary along in steps, and find those samples which have changed
    # Use remove_edges/add_edges with index k lookup (n total) to find sample IDs
    # https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
    # i = n - 2 - int(sqrt(-8*k + 4*n*(n-1)-7)/2.0 - 0.5)
    # j = k + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2

    # Optimize boundary - grid search for global minimum
    sys.stderr.write("Trying to optimise score globally\n")
    global_grid_resolution = 40 # Seems to work
    s_range = np.linspace(-min_move, max_move, num = global_grid_resolution)

    # Move distMat into shared memory
    with SharedMemoryManager() as smm:
        shm_distMat = smm.SharedMemory(size = distMat.nbytes)
        distances_shared_array = np.ndarray(distMat.shape, dtype = distMat.dtype, buffer = shm_distMat.buf)
        distances_shared_array[:] = distMat[:]
        distances_shared = NumpyShared(name = shm_distMat.name, shape = distMat.shape, dtype = distMat.dtype)

        with Pool(processes = num_processes) as pool:
            global_s = pool.map(partial(newNetwork,
                                        sample_names = sample_names,
                                        distMat = distances_shared,
                                        start_point = start_point,
                                        mean1 = mean1,
                                        gradient = gradient,
                                        slope = slope),
                                s_range)

    # Local optimisation around global optimum
    min_idx = np.argmin(np.array(global_s))
    if min_idx > 0 and min_idx < len(s_range) - 1 and not no_local:
        sys.stderr.write("Trying to optimise score locally\n")
        local_s = scipy.optimize.minimize_scalar(newNetwork,
                        bounds=[s_range[min_idx-1], s_range[min_idx+1]],
                        method='Bounded', options={'disp': True},
                        args = (sample_names, distMat, start_point, mean1, gradient, slope))
        optimised_s = local_s.x
    else:
        optimised_s = s_range[min_idx]

    optimised_coor = transformLine(optimised_s, start_point, mean1)
    if slope == 2:
        optimal_x, optimal_y = decisionBoundary(optimised_coor, gradient)
    else:
        optimal_x = optimised_coor[0]
        optimal_y = optimised_coor[1]

    if optimal_x < 0 or optimal_y < 0:
        raise RuntimeError("Optimisation failed: produced a boundary outside of allowed range\n")

    return start_point, optimal_x, optimal_y, min_move, max_move


def newNetwork(s, sample_names, distMat, start_point, mean1, gradient, slope=2, cpus = 1):
    """Wrapper function for :func:`~PopPUNK.network.constructNetwork` which is called
    by optimisation functions moving a triangular decision boundary.

    Given the boundary parameterisation, constructs the network and returns
    its score, to be minimised.

    Args:
        s (float)
            Distance along line between start_point and mean1 from start_point
        sample_names (list)
            Sample names corresponding to distMat (accessed by iterator)
        distMat (numpy.array or NumpyShared)
            Core and accessory distances or NumpyShared describing these in sharedmem
        start_point (numpy.array)
            Initial boundary cutoff
        mean1 (numpy.array)
            Defines line direction from start_point
        gradient (float)
            Gradient of line to move along
        slope (int)
            Set to 0 for a vertical line, 1 for a horizontal line, or
            2 to use a slope
        cpus (int)
            Number of CPUs to use for calculating assignment
    Returns:
        score (float)
            -1 * network score. Where network score is from :func:`~PopPUNK.network.networkSummary`
    """
    if isinstance(distMat, NumpyShared):
        distMat_shm = shared_memory.SharedMemory(name = distMat.name)
        distMat = np.ndarray(distMat.shape, dtype = distMat.dtype, buffer = distMat_shm.buf)

    # Set up boundary
    new_intercept = transformLine(s, start_point, mean1)
    if slope == 2:
        x_max, y_max = decisionBoundary(new_intercept, gradient)
    elif slope == 0:
        x_max = new_intercept[0]
        y_max = 0
    elif slope == 1:
        x_max = 0
        y_max = new_intercept[1]

    # Make network
    boundary_assignments = pp_sketchlib.assignThreshold(distMat, slope, x_max, y_max, cpus)
    G = constructNetwork(sample_names, sample_names, boundary_assignments, -1, summarise = False)

    # Return score
    score = networkSummary(G)[3]
    return(-score)

def readManualStart(startFile):
    """Reads a file to define a manual start point, rather than using ``--fit-model``

    Throws and exits if incorrectly formatted.

    Args:
        startFile (str)
            Name of file with values to read
    Returns:
        mean0 (numpy.array)
            Centre of within-strain distribution
        mean1 (numpy.array)
            Centre of between-strain distribution
        start_s (float)
            Distance along line between mean0 and mean1 to start at
    """
    start_s = None
    mean0 = None
    mean1 = None

    with open(startFile, 'r') as start:
        for line in start:
            (param, value) = line.rstrip().split()
            if param == 'mean0':
                mean_read = []
                for mean_val in value.split(','):
                    mean_read.append(float(mean_val))
                mean0 = np.array(mean_read)
            elif param == 'mean1':
                mean_read = []
                for mean_val in value.split(','):
                    mean_read.append(float(mean_val))
                mean1 = np.array(mean_read)
            elif param == 'start_point':
                start_s = float(value)
    try:
        if not isinstance(mean0, np.ndarray) or not isinstance(mean1, np.ndarray) or start_s == None:
            raise RuntimeError('All of mean0, mean1 and start_s must all be set')
        if mean0.shape != (2,) or mean1.shape != (2,):
            raise RuntimeError('Wrong size for values')
        check_vals = np.hstack([mean0, mean1, start_s])
        for val in np.nditer(check_vals):
            if val > 1 or val < 0:
                raise RuntimeError('Value out of range (between 0 and 1)')
    except RuntimeError as e:
        sys.stderr.write("Could not read manual start file " + startFile + "\n")
        sys.stderr.write(e)
        sys.exit(1)

    return mean0, mean1, start_s


def likelihoodBoundary(s, model, start, end, within, between):
    """Wrapper function around :func:`~PopPUNK.bgmm.fit2dMultiGaussian` so that it can
    go into a root-finding function for probabilities between components

    Args:
        s (float)
            Distance along line from mean0
        model (BGMMFit)
            Fitted mixture model
        start (numpy.array)
            The co-ordinates of the centre of the within-strain distribution
        end (numpy.array)
            The co-ordinates of the centre of the between-strain distribution
        within (int)
            Label of the within-strain distribution
        between (int)
            Label of the between-strain distribution
    Returns:
        responsibility (float)
            The difference between responsibilities of assignment to the within component
            and the between assignment
    """
    X = transformLine(s, start, end).reshape(1, -1)
    responsibilities = model.assign(X, values = True)
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
    """Returns the co-ordinates where the triangle the decision boundary forms
    meets the x- and y-axes.

    Args:
        intercept (numpy.array)
            Cartesian co-ordinates of point along line (:func:`~transformLine`)
            which intercepts the boundary
        gradient (float)
            Gradient of the line
    Returns:
        x (float)
            The x-axis intercept
        y (float)
            The y-axis intercept
    """
    x = intercept[0] + intercept[1] * gradient
    y = intercept[1] + intercept[0] / gradient
    return(x, y)

