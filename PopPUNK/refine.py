# vim: set fileencoding=<utf-8> :
# Copyright 2018-2023 John Lees and Nick Croucher

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
from math import sqrt
from tqdm import tqdm
try:
    from multiprocessing import Pool, shared_memory
    from multiprocessing.managers import SharedMemoryManager
    NumpyShared = collections.namedtuple('NumpyShared', ('name', 'shape', 'dtype'))
except ImportError as e:
    sys.stderr.write("This version of PopPUNK requires python v3.8 or higher\n")
    sys.exit(0)
import graph_tool.all as gt
import pandas as pd

# Load GPU libraries
try:
    import cupyx
    import cugraph
    import cudf
    import cupy as cp
    from numba import cuda
    import rmm
except ImportError:
    pass

import poppunk_refine

from .__main__ import betweenness_sample_default

from .network import construct_network_from_df, printClusters
from .network import construct_network_from_edge_list
from .network import networkSummary
from .network import add_self_loop

from .utils import transformLine
from .utils import decisionBoundary
from .utils import check_and_set_gpu

def refineFit(distMat, sample_names, mean0, mean1, scale,
              max_move, min_move, slope = 2, score_idx = 0,
              unconstrained = False, no_local = False, num_processes = 1,
              betweenness_sample = betweenness_sample_default, sample_size = None,
              use_gpu = False):
    """Try to refine a fit by maximising a network score based on transitivity and density.

    Iteratively move the decision boundary to do this, using starting point from existing model.

    Args:
        distMat (numpy.array)
            n x 2 array of core and accessory distances for n samples
        sample_names (list)
            List of query sequence labels
        mean0 (numpy.array)
            Start point to define search line
        mean1 (numpy.array)
            End point to define search line
        scale (numpy.array)
            Scaling factor of distMat
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
        betweenness_sample (int)
            Number of sequences per component used to estimate betweenness using
            a GPU. Smaller numbers are faster but less precise [default = 100]
        sample_size (int)
            Number of nodes to subsample for graph statistic calculation
        use_gpu (bool)
            Whether to use cugraph for graph analyses

    Returns:
        optimal_x (float)
            x-coordinate of refined fit
        optimal_y (float)
            y-coordinate of refined fit
        optimised_s (float)
            Position along search range of refined fit
    """
    # Optimize boundary - grid search for global minimum
    sys.stderr.write("Trying to optimise score globally\n")


    # Boundary is left of line normal to this point and first line
    gradient = (mean1[1] - mean0[1]) / (mean1[0] - mean0[0])

    if unconstrained:
        if slope != 2:
            raise RuntimeError("Unconstrained optimization and indiv-refine incompatible")

        global_grid_resolution = 20
        x_max_start, y_max_start = decisionBoundary(mean0, gradient)
        x_max_end, y_max_end = decisionBoundary(mean1, gradient)

        if x_max_start < 0 or y_max_start < 0:
            raise RuntimeError("Boundary range below zero")

        x_max = np.linspace(x_max_start, x_max_end, global_grid_resolution, dtype=np.float32)
        y_max = np.linspace(y_max_start, y_max_end, global_grid_resolution, dtype=np.float32)
        sys.stderr.write("Searching core intercept from " +
                         "{:.3f}".format(x_max_start * scale[0]) +
                         " to " + "{:.3f}".format(x_max_end * scale[0]) + "\n")
        sys.stderr.write("Searching accessory intercept from " +
                         "{:.3f}".format(y_max_start * scale[1]) +
                         " to " + "{:.3f}".format(y_max_end * scale[1]) + "\n")

        if use_gpu:
            global_s = map(partial(newNetwork2D,
                                   sample_names = sample_names,
                                   distMat = distMat,
                                   x_range = x_max,
                                   y_range = y_max,
                                   score_idx = score_idx,
                                   betweenness_sample = betweenness_sample,
                                   sample_size = sample_size,
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
                                                betweenness_sample = betweenness_sample,
                                                sample_size = sample_size,
                                                use_gpu = False),
                                        range(global_grid_resolution))

            if gt.openmp_enabled():
                gt.openmp_set_num_threads(num_processes)

        global_s = np.array(list(chain.from_iterable(global_s)))
        global_s[np.isnan(global_s)] = 1
        min_idx = np.argmin(global_s)
        optimal_x = x_max[min_idx % global_grid_resolution]
        optimal_y = y_max[min_idx // global_grid_resolution]
        optimised_s = global_s[min_idx]

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
            mean1 = (optimal_x + delta, delta * gradient)

    else:
        # Set the range of points to search
        search_length = max_move + ((mean1[0] - mean0[0])**2 + (mean1[1] - mean0[1])**2)**0.5
        global_grid_resolution = 40 # Seems to work
        s_range = np.linspace(-min_move, search_length, num = global_grid_resolution)
        (min_x, max_x), (min_y, max_y) = \
            check_search_range(scale, mean0, mean1, s_range[0], s_range[-1])
        if min_x < 0 or min_y < 0:
            raise RuntimeError("Boundary range below zero")

        i_vec, j_vec, idx_vec = \
            poppunk_refine.thresholdIterate1D(distMat, s_range, slope,
                                              mean0[0], mean0[1],
                                              mean1[0], mean1[1], num_processes)
        if len(idx_vec) == distMat.shape[0]:
            raise RuntimeError("Boundary range includes all points")
        global_s = np.array(growNetwork(sample_names,
                                        i_vec,
                                        j_vec,
                                        idx_vec,
                                        s_range,
                                        score_idx,
                                        betweenness_sample = betweenness_sample,
                                        sample_size = sample_size,
                                        use_gpu = use_gpu))
        global_s[np.isnan(global_s)] = 1
        min_idx = np.argmin(np.array(global_s))
        if min_idx > 0 and min_idx < len(s_range) - 1:
            bounds = [s_range[min_idx-1], s_range[min_idx+1]]
        else:
            no_local = True
        if no_local:
            optimised_s = s_range[min_idx]

    # Local optimisation around global optimum
    if not no_local:
        sys.stderr.write("Trying to optimise score locally\n")
        local_s = scipy.optimize.minimize_scalar(
                    newNetwork,
                    bounds = bounds,
                    method = 'Bounded', options={'disp': True},
                    args = (sample_names, distMat, mean0, mean1, gradient,
                            slope, score_idx, num_processes,
                            betweenness_sample, sample_size, use_gpu)
                )
        optimised_s = local_s.x

    # Convert to x_max, y_max if needed
    if not unconstrained or not no_local:
        optimised_coor = transformLine(optimised_s, mean0, mean1)
        if slope == 2:
            optimal_x, optimal_y = decisionBoundary(optimised_coor, gradient)
            if optimal_x < 0 or optimal_y < 0:
                raise RuntimeError("Optimisation failed: produced a boundary outside of allowed range\n")
        else:
            optimal_x = optimised_coor[0]
            optimal_y = optimised_coor[1]
            if (slope == 0 and optimal_x < 0) or (slope == 1 and optimal_y < 0):
               raise RuntimeError("Optimisation failed: produced a boundary outside of allowed range\n")

    return optimal_x, optimal_y, optimised_s

def multi_refine(distMat, sample_names, mean0, mean1, scale, s_max,
                 n_boundary_points, output_prefix,
                 num_processes = 1, use_gpu = False):
    """Move the refinement boundary between the optimum and where it meets an
    axis. Discrete steps, output the clusers at each step

    Args:
        distMat (numpy.array)
            n x 2 array of core and accessory distances for n samples
        sample_names (list)
            List of query sequence labels
        mean0 (numpy.array)
            Start point to define search line
        mean1 (numpy.array)
            End point to define search line
        scale (numpy.array)
            Scaling factor of distMat
        s_max (float)
            The optimal s position from refinement (:func:`~PopPUNK.refine.refineFit`)
        n_boundary_points (int)
            Number of positions to try drawing the boundary at
        num_processes (int)
            Number of threads to use in the global optimisation step.
            (default = 1)
        use_gpu (bool)
            Whether to use cugraph for graph analyses
    """

    # Set the range
    # Between optimised s and where line meets an axis
    gradient = (mean1[1] - mean0[1]) / (mean1[0] - mean0[0])
    if mean0[1] >= gradient * mean0[0]:
        s_min = -mean0[0] * sqrt(1 + gradient * gradient)
    else:
        s_min = -mean0[1] * sqrt(1 + 1 / (gradient * gradient))
    s_range = np.linspace(s_min, s_max, num = n_boundary_points)

    (min_x, max_x), (min_y, max_y) = \
        check_search_range(scale, mean0, mean1, s_range[0], s_range[-1])
    if min_x < 0 or min_y < 0:
        sys.stderr.write("Boundary range below zero")

    i_vec, j_vec, idx_vec = \
        poppunk_refine.thresholdIterate1D(distMat, s_range, 2,
                                          mean0[0], mean0[1],
                                          mean1[0], mean1[1],
                                          num_processes)

    growNetwork(sample_names,
                i_vec,
                j_vec,
                idx_vec,
                s_range,
                0,
                write_clusters = output_prefix,
                sample_size = sample_size,
                use_gpu = use_gpu)

def check_search_range(scale, mean0, mean1, lower_s, upper_s):
    """Checks a search range is within a valid range

    Args:
        scale (np.array)
            Rescaling factor to [0, 1] for each axis
        mean0 (np.array)
            (x, y) of starting point defining line
        mean1 (np.array)
            (x, y) of end point defining line
        lower_s (float)
            distance along line to start search
        upper_s (float)
            distance along line to end search

    Returns:
        min_x, max_x
            minimum and maximum x-intercepts of the search range
        min_y, max_y
            minimum and maximum x-intercepts of the search range
    """
    gradient = (mean1[1] - mean0[1]) / (mean1[0] - mean0[0])
    bottom_end = transformLine(lower_s, mean0, mean1)
    top_end = transformLine(upper_s, mean0, mean1)
    min_x, min_y = decisionBoundary(bottom_end, gradient)
    max_x, max_y = decisionBoundary(top_end, gradient)

    sys.stderr.write("Search range (" +
                        ",".join(["{:.3f}".format(x) for x in bottom_end * scale]) +
                        ") to (" +
                        ",".join(["{:.3f}".format(x) for x in top_end * scale]) + ")\n")
    sys.stderr.write("Searching core intercept from " +
                        "{:.3f}".format(min_x * scale[0]) +
                        " to " + "{:.3f}".format(max_x * scale[0]) + "\n")
    sys.stderr.write("Searching accessory intercept from " +
                        "{:.3f}".format(min_y * scale[1]) +
                        " to " + "{:.3f}".format(max_y * scale[1]) + "\n")

    return((min_x, max_x), (min_y, max_y))

def expand_cugraph_network(G, G_extra_df):
    """Reconstruct a cugraph network with additional edges.

    Args:
        G (cugraph network)
            Original cugraph network
        extra_edges (cudf dataframe)
            Data frame of edges to add

    Returns:
        G (cugraph network)
            Expanded cugraph network
    """
    G_vertex_count = G.number_of_vertices()-1
    G_original_df = G.view_edge_list()
    if 'src' in G_original_df.columns:
        G_original_df.columns = ['source','destination']
    G_df = cudf.concat([G_original_df,G_extra_df])
    G = add_self_loop(G_df, G_vertex_count, weights = False, renumber = False)
    return G

def growNetwork(sample_names, i_vec, j_vec, idx_vec, s_range, score_idx = 0,
                thread_idx = 0, betweenness_sample = betweenness_sample_default,
                write_clusters = None, sample_size = None, use_gpu = False):
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
        betweenness_sample (int)
            Number of sequences per component used to estimate betweenness using
            a GPU. Smaller numbers are faster but less precise [default = 100]
        write_clusters (str)
            Set to a prefix to write the clusters from each position to files
            [default = None]
        sample_size (int)
            Number of nodes to subsample for graph statistic calculation
        use_gpu (bool)
            Whether to use cugraph for graph analyses

    Returns:
        scores (list)
            -1 * network score for each of x_range.
            Where network score is from :func:`~PopPUNK.network.networkSummary`
    """
    scores = []
    prev_idx = -1

    # create data frame
    if use_gpu:
        edge_list_df = cudf.DataFrame()
    else:
        edge_list_df = pd.DataFrame()
    edge_list_df['source'] = i_vec
    edge_list_df['destination'] = j_vec
    edge_list_df['idx_list'] = idx_vec
    if use_gpu:
        idx_values = edge_list_df.to_pandas().idx_list.unique()
    else:
        idx_values = edge_list_df.idx_list.unique()

    # Grow a network
    with tqdm(total = max(idx_values) + 1,
              bar_format = "{bar}| {n_fmt}/{total_fmt}",
              ncols = 40,
              position = thread_idx) as pbar:
        for idx in idx_values:
            # Create DF
            edge_df = edge_list_df.loc[(edge_list_df['idx_list']==idx),['source','destination']]
            # At first offset, make a new network, otherwise just add the new edges
            if prev_idx == -1:
                G = construct_network_from_df(sample_names, sample_names,
                                              edge_df,
                                              summarise = False,
                                              use_gpu = use_gpu)
            else:
                if use_gpu:
                    G = expand_cugraph_network(G, edge_df)
                else:
                    edge_list = list(edge_df[['source','destination']].itertuples(index=False, name=None))
                    G.add_edge_list(edge_list)
                    edge_list = []
            # Add score into vector for any offsets passed (should usually just be one)
            G_summary = networkSummary(G,
                                score_idx > 0,
                                betweenness_sample = betweenness_sample,
                                subsample = sample_size,
                                use_gpu = use_gpu)
            latest_score = -G_summary[1][score_idx]
            for s in range(prev_idx, idx):
                scores.append(latest_score)
                pbar.update(1)
                # Write the cluster output as long as there is at least one
                # non-trivial cluster
                if write_clusters and G_summary[0][0] < len(sample_names):
                    o_prefix = \
                        f"{write_clusters}/{os.path.basename(write_clusters)}_boundary{s + 1}"
                    printClusters(G,
                                  sample_names,
                                  outPrefix=o_prefix,
                                  write_unwords=False,
                                  use_gpu=use_gpu)

            prev_idx = idx

    return(scores)

def newNetwork(s, sample_names, distMat, mean0, mean1, gradient,
               slope=2, score_idx=0, cpus=1, betweenness_sample = betweenness_sample_default,
               sample_size = None, use_gpu = False):
    """Wrapper function for :func:`~PopPUNK.network.construct_network_from_edge_list` which is called
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
        mean0 (numpy.array)
            Start point
        mean1 (numpy.array)
            End point
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
        betweenness_sample (int)
            Number of sequences per component used to estimate betweenness using
            a GPU. Smaller numbers are faster but less precise [default = 100]
        sample_size (int)
            Number of nodes to subsample for graph statistic calculation
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
    new_intercept = transformLine(s, mean0, mean1)
    if slope == 2:
        x_max, y_max = decisionBoundary(new_intercept, gradient)
    elif slope == 0:
        x_max = new_intercept[0]
        y_max = 0
    elif slope == 1:
        x_max = 0
        y_max = new_intercept[1]

    # Make network
    connections = poppunk_refine.edgeThreshold(distMat, slope, x_max, y_max)
    G = construct_network_from_edge_list(sample_names,
                                        sample_names,
                                        connections,
                                        summarise = False,
                                        use_gpu = use_gpu)

    # Return score
    score = networkSummary(G,
                            score_idx > 0,
                            subsample = sample_size,
                            betweenness_sample = betweenness_sample,
                            use_gpu = use_gpu)[1][score_idx]
    return(-score)

def newNetwork2D(y_idx, sample_names, distMat, x_range, y_range, score_idx=0,
                 betweenness_sample = betweenness_sample_default, sample_size = None,
                 use_gpu = False):
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
        betweenness_sample (int)
            Number of sequences per component used to estimate betweenness using
            a GPU. Smaller numbers are faster but less precise [default = 100]
        sample_size (int)
            Number of nodes to subsample for graph statistic calculation
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

    # If everything is in the network, skip this boundary
    if len(idx_vec) == distMat.shape[0]:
        scores = [0] * len(x_range)
    else:
        scores = growNetwork(sample_names,
                                i_vec,
                                j_vec,
                                idx_vec,
                                x_range,
                                score_idx,
                                y_idx,
                                betweenness_sample,
                                sample_size = sample_size,
                                use_gpu = use_gpu)

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
        scaled (bool)
            True if means are scaled between [0,1]
    """
    mean0 = None
    mean1 = None
    scaled = True

    with open(startFile, 'r') as start:
        for line in start:
            (param, value) = line.rstrip().split()
            if param == 'start':
                mean_read = []
                for mean_val in value.split(','):
                    mean_read.append(float(mean_val))
                mean0 = np.array(mean_read)
            elif param == 'end':
                mean_read = []
                for mean_val in value.split(','):
                    mean_read.append(float(mean_val))
                mean1 = np.array(mean_read)
            elif param == 'scaled':
                if value == "False" or value == "false":
                    scaled = False
            else:
                raise RuntimeError("Incorrectly formatted manual start file")
    try:
        if not isinstance(mean0, np.ndarray) or not isinstance(mean1, np.ndarray):
            raise RuntimeError('Must set both start and end')
        if mean0.shape != (2,) or mean1.shape != (2,):
            raise RuntimeError('Wrong size for values')
        check_vals = np.hstack([mean0, mean1])
        for val in np.nditer(check_vals):
            if val > 1 or val < 0:
                raise RuntimeError('Value out of range (between 0 and 1)')
    except RuntimeError as e:
        sys.stderr.write("Could not read manual start file " + startFile + "\n")
        sys.stderr.write(str(e) + "\n")
        sys.exit(1)

    return mean0, mean1, scaled

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
