# vim: set fileencoding=<utf-8> :
# Copyright 2018-2020 John Lees and Nick Croucher

'''Refine mixture model using network properties'''

# universal
import os
import sys
# additional
from itertools import chain
from functools import partial
import numpy as np
import scipy.optimize
import collections
from tqdm import tqdm
try:
    from multiprocessing import Pool, shared_memory
    from multiprocessing.managers import SharedMemoryManager
    NumpyShared = collections.namedtuple('NumpyShared', ('name', 'shape', 'dtype'))
except ImportError as e:
    sys.stderr.write("This version of PopPUNK requires python v3.8 or higher\n")
    sys.exit(0)
import pp_sketchlib
import poppunk_refine
import graph_tool.all as gt

# GPU support
try:
    import cugraph
    import cudf
    gpu_lib = True
except ImportError as e:
    gpu_lib = False

from .network import constructNetwork
from .network import networkSummary

from .utils import transformLine
from .utils import decisionBoundary

def refineFit(distMat, sample_names, start_s, mean0, mean1,
              max_move, min_move, slope = 2, score_idx = 0,
              unconstrained = False, no_local = False, num_processes = 1, use_gpu = False):
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
        score_idx (int)
            Index of score from :func:`~PopPUNK.network.networkSummary` to use
            [default = 0]
        unconstrained (bool)
            If True, search in 2D and change the slope of the boundary
        no_local (bool)
            Turn off the local optimisation step.
            Quicker, but may be less well refined.
        num_processes (int)
            Number of threads to use in the global optimisation step.
            (default = 1)
        use_gpu (bool)
            Whether to use cugraph for graph analyses

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

    # Optimize boundary - grid search for global minimum
    sys.stderr.write("Trying to optimise score globally\n")

    if unconstrained:
        if slope != 2:
            raise RuntimeError("Unconstrained optimization and indiv-refine incompatible")

        global_grid_resolution = 20
        x_max_start, y_max_start = decisionBoundary(mean0, gradient)
        x_max_end, y_max_end = decisionBoundary(mean1, gradient)
        x_max = np.linspace(x_max_start, x_max_end, global_grid_resolution, dtype=np.float32)
        y_max = np.linspace(y_max_start, y_max_end, global_grid_resolution, dtype=np.float32)

        if use_gpu:
            global_s = map(partial(newNetwork2D,
                                   sample_names = sample_names,
                                   distMat = distMat,
                                   x_range = x_max,
                                   y_range = y_max,
                                   score_idx = score_idx,
                                   use_gpu = True),
                           range(global_grid_resolution))
        else:
            if gt.openmp_enabled():
                gt.openmp_set_num_threads(1)

            with SharedMemoryManager() as smm:
                shm_distMat = smm.SharedMemory(size = distMat.nbytes)
                distances_shared_array = np.ndarray(distMat.shape, dtype = distMat.dtype, buffer = shm_distMat.buf)
                distances_shared_array[:] = distMat[:]
                distances_shared = NumpyShared(name = shm_distMat.name, shape = distMat.shape, dtype = distMat.dtype)

                with Pool(processes = num_processes) as pool:
                    global_s = pool.map(partial(newNetwork2D,
                                                sample_names = sample_names,
                                                distMat = distances_shared,
                                                x_range = x_max,
                                                y_range = y_max,
                                                score_idx = score_idx,
                                                use_gpu = False),
                                        range(global_grid_resolution))

            if gt.openmp_enabled():
                gt.openmp_set_num_threads(num_processes)

        global_s = np.array(list(chain.from_iterable(global_s)))
        global_s[np.isnan(global_s)] = 1
        min_idx = np.argmin(global_s)
        optimal_x = x_max[min_idx % global_grid_resolution]
        optimal_y = y_max[min_idx // global_grid_resolution]

        if not (optimal_x > x_max_start and optimal_x < x_max_end and \
                optimal_y > y_max_start and optimal_y < y_max_end):
            no_local = True
        elif not no_local:
            # We have a fixed gradient and want to optimised the intercept
            # This parameterisation is a little awkward to match the 1D case:
            # Make two points along the right slope
            gradient = optimal_x / optimal_y # of 1D search
            delta = x_max[1] - x_max[0]
            bounds = [-delta, delta]
            start_point = (optimal_x, 0)
            mean1 = (optimal_x + delta, delta * gradient)

    else:
        global_grid_resolution = 40 # Seems to work
        s_range = np.linspace(-min_move, max_move, num = global_grid_resolution)
        i_vec, j_vec, idx_vec = \
            poppunk_refine.thresholdIterate1D(distMat, s_range, slope,
                                                  start_point[0], start_point[1],
                                                  mean1[0], mean1[1], num_processes)
        global_s = np.array(growNetwork(sample_names, i_vec, j_vec, idx_vec, s_range, score_idx, use_gpu = use_gpu))
        global_s[np.isnan(global_s)] = 1
        min_idx = np.argmin(np.array(global_s))
        if min_idx > 0 and min_idx < len(s_range) - 1:
            bounds = [s_range[min_idx-1], s_range[min_idx+1]]
        else:
            no_local = True
            optimised_s = s_range[min_idx]

    # Local optimisation around global optimum
    if not no_local:
        sys.stderr.write("Trying to optimise score locally\n")
        local_s = scipy.optimize.minimize_scalar(newNetwork,
                        bounds=bounds,
                        method='Bounded', options={'disp': True},
                        args = (sample_names, distMat, start_point, mean1, gradient,
                                slope, score_idx, num_processes, use_gpu),
                        )
        optimised_s = local_s.x

    # Convert to x_max, y_max if needed
    if not unconstrained or not no_local:
        optimised_coor = transformLine(optimised_s, start_point, mean1)
        if slope == 2:
            optimal_x, optimal_y = decisionBoundary(optimised_coor, gradient)
        else:
            optimal_x = optimised_coor[0]
            optimal_y = optimised_coor[1]

    if optimal_x < 0 or optimal_y < 0:
        raise RuntimeError("Optimisation failed: produced a boundary outside of allowed range\n")

    return start_point, optimal_x, optimal_y, min_move, max_move


def growNetwork(sample_names, i_vec, j_vec, idx_vec, s_range, score_idx, thread_idx = 0, use_gpu = False):
    """Construct a network, then add edges to it iteratively.
    Input is from ``pp_sketchlib.iterateBoundary1D`` or``pp_sketchlib.iterateBoundary2D``

    Args:
        sample_names (list)
            Sample names corresponding to distMat (accessed by iterator)
        i_vec (list)
            Ordered ref vertex index to add
        j_vec (list)
            Ordered query (==ref) vertex index to add
        idx_vec (list)
            For each i, j tuple, the index of the intercept at which these enter
            the network. These are sorted and increasing
        s_range (list)
            Offsets which correspond to idx_vec entries
        score_idx (int)
            Index of score from :func:`~PopPUNK.network.networkSummary` to use
            [default = 0]
        thread_idx (int)
            Optional thread idx (if multithreaded) to offset progress bar by
        use_gpu (bool)
            Whether to use cugraph for graph analyses

    Returns:
        scores (list)
            -1 * network score for each of x_range.
            Where network score is from :func:`~PopPUNK.network.networkSummary`
    """
    # load CUDA libraries
    if use_gpu and not gpu_lib:
        sys.stderr.write('Unable to load GPU libraries; exiting\n')
        sys.exit(1)

    scores = []
    edge_list = []
    prev_idx = 0
    # Grow a network
    with tqdm(total=(idx_vec[-1] + 1),
              bar_format="{bar}| {n_fmt}/{total_fmt}",
              ncols=40,
              position=thread_idx) as pbar:
        for i, j, idx in zip(i_vec, j_vec, idx_vec):
            if idx > prev_idx:
                # At first offset, make a new network, otherwise just add the new edges
                if prev_idx == 0:
                    G = constructNetwork(sample_names, sample_names, edge_list, -1,
                                         summarise=False, edge_list=True, use_gpu = use_gpu)
                else:
                    if use_gpu:
                        G_current_df = G.view_edge_list()
                        G_current_df.columns = ['source','destination']
                        G_extra_df = cudf.DataFrame(edge_list, columns =['source','destination'])
                        G_df = cudf.concat([G_current_df,G_extra_df], ignore_index = True)
                        G = cugraph.Graph()
                        G.from_cudf_edgelist(G_df)
                    else:
                        # Adding edges to network not currently possible with GPU - https://github.com/rapidsai/cugraph/issues/805
                        # We add to the cuDF, and then reconstruct the network instead
                        G.add_edge_list(edge_list)
                # Add score into vector for any offsets passed (should usually just be one)
                for s in range(prev_idx, idx):
                    scores.append(-networkSummary(G, score_idx > 0, use_gpu = use_gpu)[1][score_idx])
                    pbar.update(1)
                prev_idx = idx
                edge_list = []
            edge_list.append((i, j))

        # Add score for final offset(s) at end of loop
        if prev_idx == 0:
            G = constructNetwork(sample_names, sample_names, edge_list, -1,
                                 summarise=False, edge_list=True, use_gpu = use_gpu)
        else:
            if use_gpu:
                G = constructNetwork(sample_names, sample_names, edge_list, -1,
                                        summarise=False, edge_list=True, use_gpu = use_gpu)
            else:
                # Not currently possible with GPU - https://github.com/rapidsai/cugraph/issues/805
                G.add_edge_list(edge_list)
        for s in range(prev_idx, len(s_range)):
            scores.append(-networkSummary(G, score_idx > 0, use_gpu = use_gpu)[1][score_idx])
            pbar.update(1)

    return(scores)


def newNetwork(s, sample_names, distMat, start_point, mean1, gradient,
               slope=2, score_idx=0, cpus=1, use_gpu = False):
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
            [default = 2]
        score_idx (int)
            Index of score from :func:`~PopPUNK.network.networkSummary` to use
            [default = 0]
        cpus (int)
            Number of CPUs to use for calculating assignment
        use_gpu (bool)
            Whether to use cugraph for graph analysis

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
    boundary_assignments = poppunk_refine.assignThreshold(distMat, slope, x_max, y_max, cpus)
    G = constructNetwork(sample_names, sample_names, boundary_assignments, -1, summarise = False,
                            use_gpu = use_gpu)

    # Return score
    score = networkSummary(G, score_idx > 0, use_gpu = use_gpu)[1][score_idx]
    return(-score)

def newNetwork2D(y_idx, sample_names, distMat, x_range, y_range, score_idx=0, use_gpu = False):
    """Wrapper function for thresholdIterate2D and :func:`growNetwork`.

    For a given y_max, constructs networks across x_range and returns a list
    of scores

    Args:
        y_idx (float)
            Maximum y-intercept of boundary, as index into y_range
        sample_names (list)
            Sample names corresponding to distMat (accessed by iterator)
        distMat (numpy.array or NumpyShared)
            Core and accessory distances or NumpyShared describing these in sharedmem
        x_range (list)
            Sorted list of x-intercepts to search
        y_range (list)
            Sorted list of y-intercepts to search
        score_idx (int)
            Index of score from :func:`~PopPUNK.network.networkSummary` to use
            [default = 0]
        use_gpu (bool)
            Whether to use cugraph for graph analysis

    Returns:
        scores (list)
            -1 * network score for each of x_range.
            Where network score is from :func:`~PopPUNK.network.networkSummary`
    """
    if gt.openmp_enabled():
        gt.openmp_set_num_threads(1)
    if isinstance(distMat, NumpyShared):
        distMat_shm = shared_memory.SharedMemory(name = distMat.name)
        distMat = np.ndarray(distMat.shape, dtype = distMat.dtype, buffer = distMat_shm.buf)

    y_max = y_range[y_idx]
    i_vec, j_vec, idx_vec = \
            poppunk_refine.thresholdIterate2D(distMat, x_range, y_max)
    scores = growNetwork(sample_names, i_vec, j_vec, idx_vec, x_range, score_idx, y_idx, use_gpu = use_gpu)
    return(scores)

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
    responsibilities = model.assign(X, progress = False, values = True)
    return(responsibilities[0, within] - responsibilities[0, between])
