# vim: set fileencoding=<utf-8> :
# Copyright 2018-2023 John Lees and Nick Croucher

'''Plots of GMM results, k-mer fits, and microreact output'''

import sys
import os
import subprocess
import random
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams.update({'font.size': 18})
import matplotlib.pyplot as plt
import itertools
# for other outputs
import pandas as pd
from pandas.errors import DataError
from collections import defaultdict
from sklearn import utils
try:  # sklearn >= 0.22
    from sklearn.neighbors import KernelDensity
except ImportError:
    from sklearn.neighbors.kde import KernelDensity

from .trees import write_tree

from .utils import isolateNameToLabel
from .utils import decisionBoundary

def plot_scatter(X, out_prefix, title, kde = True):
    """Draws a 2D scatter plot (png) of the core and accessory distances

    Also draws contours of kernel density estimare

    Args:
        X (numpy.array)
            n x 2 array of core and accessory distances for n samples.
        out_prefix (str)
            Prefix for output plot file (.png will be appended)
        title (str)
            The title to display above the plot
        kde (bool)
            Whether to draw kernel density estimate contours

            (default = True)
    """
    # Plot results - max 1M for speed
    max_plot_samples = 1000000
    if X.shape[0] > max_plot_samples:
        X = utils.shuffle(X, random_state=random.randint(1,10000))[0:max_plot_samples,]

    # Kernel estimate uses scaled data 0-1 on each axis
    scale = np.amax(X, axis = 0)
    X /= scale

    plt.figure(figsize=(11, 8), dpi= 160, facecolor='w', edgecolor='k')
    if kde:
        xx, yy, xy = get_grid(0, 1, 100)

        # KDE estimate
        kde = KernelDensity(bandwidth=0.03, metric='euclidean',
                            kernel='epanechnikov', algorithm='ball_tree')
        kde.fit(X)
        z = np.exp(kde.score_samples(xy))
        z = z.reshape(xx.shape).T

        levels = np.linspace(z.min(), z.max(), 10)
        # Rescale contours
        plt.contour(xx*scale[0], yy*scale[1], z, levels=levels[1:], cmap='plasma')
        scatter_alpha = 1
    else:
        scatter_alpha = 0.1

    # Plot on correct scale
    plt.scatter(X[:,0]*scale[0].flat, X[:,1]*scale[1].flat, s=1, alpha=scatter_alpha)

    plt.title(title)
    plt.xlabel('Core distance (' + r'$\pi$' + ')')
    plt.ylabel('Accessory distance (' + r'$a$' + ')')
    plt.savefig(os.path.join(out_prefix, os.path.basename(out_prefix) + '_distanceDistribution.png'))
    plt.close()

def plot_database_evaluations(genome_lengths, ambiguous_bases):
    """Plot histograms of sequence characteristics for database evaluation.

    Args:
        genome_lengths (list)
            Lengths of genomes in database
        ambiguous_bases (list)
            Counts of ambiguous bases in genomes in database
    """
    plot_evaluation_histogram(genome_lengths,
                              n_bins = 100,
                              prefix = prefix,
                              suffix = 'genome_lengths',
                              plt_title = 'Distribution of sequence lengths',
                              xlab = 'Sequence length (nt)')
    plot_evaluation_histogram(ambiguous_bases,
                              n_bins = 100,
                              prefix = prefix,
                              suffix = 'ambiguous_base_counts',
                              plt_title = 'Distribution of ambiguous base counts',
                              xlab = 'Number of ambiguous bases')

def plot_evaluation_histogram(input_data, n_bins = 100, prefix = 'hist',
    suffix = '', plt_title = 'histogram', xlab = 'x'):
    """Plot histograms of sequence characteristics for database evaluation.

    Args:
        input_data (list)
            Input data (list of numbers)
        n_bins (int)
            Number of bins to use for the histogram
        prefix (str)
            Prefix of database
        suffix (str)
            Suffix specifying plot type
        plt_title (str)
            Title for plot
        xlab (str)
            Title for the horizontal axis
    """
    plt.figure(figsize=(8, 8), dpi=160, facecolor='w', edgecolor='k')
    counts, bins = np.histogram(input_data, bins = n_bins)
    plt.stairs(counts, bins, fill = True)
    plt.title(plt_title)
    plt.xlabel(xlab)
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(prefix, os.path.basename(prefix) + '_' + suffix + '.png'))
    plt.savefig(os.path.join(prefix,prefix + '.png'))
    plt.close()

def plot_fit(klist, raw_matching, raw_fit, corrected_matching, corrected_fit, out_prefix, title):
    """Draw a scatter plot (pdf) of k-mer sizes vs match probability, and the
    fit used to assign core and accessory distance

    K-mer sizes on x-axis, log(pr(match)) on y - expect a straight line fit
    with intercept representing accessory distance and slope core distance

    Args:
        klist (list)
            List of k-mer sizes
        raw_matching (list)
            Proportion of matching k-mers at each klist value
        raw_fit (numpy.array)
            Fit to klist and raw_matching from :func:`~PopPUNK.sketchlib.fitKmerCurve`
        corrected_matching (list)
            Corrected proportion of matching k-mers at each klist value
        corrected_fit (numpy.array)
            Fit to klist and corrected_matching from :func:`~PopPUNK.sketchlib.fitKmerCurve`
        out_prefix (str)
            Prefix for output plot file (.pdf will be appended)
        title (str)
            The title to display above the plot
    """
    k_fit = np.linspace(0, klist[-1], num = 100)
    raw_matching_fit = (1 - raw_fit[1]) * np.power((1 - raw_fit[0]), k_fit)
    corrected_matching_fit = (1 - corrected_fit[1]) * np.power((1 - corrected_fit[0]), k_fit)

    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_xlabel('k-mer length', fontsize = 9)
    ax.set_ylabel('Proportion of matches', fontsize = 9)
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.tick_params(axis='both', which='minor', labelsize=9)

    plt.tight_layout()
    plt.plot(klist, raw_matching, 'o', label= 'Raw matching k-mer proportion')
    plt.plot(k_fit, raw_matching_fit, 'b-', label= 'Fit to raw matches')
    plt.plot(klist, corrected_matching, 'mx', label= 'Corrected matching k-mer proportion')
    plt.plot(k_fit, corrected_matching_fit, 'm--', label= 'Fit to corrected matches')

    plt.legend(loc='upper right', prop={'size': 8})

    plt.title(title, fontsize = 10)
    plt.savefig(out_prefix + ".pdf",
                bbox_inches='tight')
    plt.close()

def plot_results(X, Y, means, covariances, scale, title, out_prefix):
    """Draw a scatter plot (png) to show the BGMM model fit

    A scatter plot of core and accessory distances, coloured by component
    membership. Also shown are ellipses for each component (centre: means
    axes: covariances).

    This is based on the example in the sklearn documentation.

    Args:
        X (numpy.array)
             n x 2 array of core and accessory distances for n samples.
        Y (numpy.array)
             n x 1 array of cluster assignments for n samples.
        means (numpy.array)
            Component means from :class:`~PopPUNK.models.BGMMFit`
        covars (numpy.array)
            Component covariances from :class:`~PopPUNK.models.BGMMFit`
        scale (numpy.array)
            Scaling factor from :class:`~PopPUNK.models.BGMMFit`
        out_prefix (str)
            Prefix for output plot file (.png will be appended)
        title (str)
            The title to display above the plot
    """
    color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold','darkorange'])

    fig=plt.figure(figsize=(11, 8), dpi= 160, facecolor='w', edgecolor='k')
    splot = plt.subplot(1, 1, 1)
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        scaled_covar = np.matmul(np.matmul(np.diag(scale), covar), np.diag(scale).T)
        v, w = np.linalg.eigh(scaled_covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y == i):
            continue
        plt.scatter([(X)[Y == i, 0]], [(X)[Y == i, 1]], .4, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean*scale, v[0], v[1], angle=180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.title(title)
    plt.xlabel('Core distance (' + r'$\pi$' + ')')
    plt.ylabel('Accessory distance (' + r'$a$' + ')')
    plt.savefig(out_prefix + ".png")
    plt.close()

def plot_dbscan_results(X, y, n_clusters, out_prefix, use_gpu):
    """Draw a scatter plot (png) to show the DBSCAN model fit

    A scatter plot of core and accessory distances, coloured by component
    membership. Black is noise

    Args:
        X (numpy.array)
             n x 2 array of core and accessory distances for n samples.
        Y (numpy.array)
             n x 1 array of cluster assignments for n samples.
        n_clusters (int)
            Number of clusters used (excluding noise)
        out_prefix (str)
            Prefix for output file (.png will be appended)
        use_gpu (bool)
            Whether model was fitted with GPU-enabled code
    """
    # Convert data if from GPU
    if use_gpu:
        # Convert to numpy for plotting
        import cupy as cp
        X = cp.asnumpy(X)
    
    # Black removed and is used for noise instead.
    unique_labels = set(y)
    colours = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]  # changed to work with two clusters

    fig=plt.figure(figsize=(11, 8), dpi= 160, facecolor='w', edgecolor='k')
    for k in unique_labels:
        if k == -1:
            ptsize = 1
            col = 'k'
        else:
            ptsize = 2
            col = tuple(colours.pop())
        class_member_mask = (y == k)
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], '.', color=col, markersize=ptsize)

    # plot output
    plt_filename = out_prefix + ".png"
    plt.title('HDBSCAN – estimated number of spatial clusters: %d' % n_clusters)
    plt.xlabel('Core distance (' + r'$\pi$' + ')')
    plt.ylabel('Accessory distance (' + r'$a$' + ')')
    plt.savefig(out_prefix + ".png")
    plt.close()

def plot_refined_results(X, Y, x_boundary, y_boundary, core_boundary, accessory_boundary,
        mean0, mean1, min_move, max_move, scale, threshold, indiv_boundaries,
        unconstrained, title, out_prefix):
    """Draw a scatter plot (png) to show the refined model fit

    A scatter plot of core and accessory distances, coloured by component
    membership. The triangular decision boundary is also shown

    Args:
        X (numpy.array)
            n x 2 array of core and accessory distances for n samples.
        Y (numpy.array)
            n x 1 array of cluster assignments for n samples.
        x_boundary (float)
            Intercept of boundary with x-axis, from :class:`~PopPUNK.models.RefineFit`
        y_boundary (float)
            Intercept of boundary with y-axis, from :class:`~PopPUNK.models.RefineFit`
        core_boundary (float)
            Intercept of 1D (core) boundary with x-axis, from :class:`~PopPUNK.models.RefineFit`
        accessory_boundary (float)
            Intercept of 1D (core) boundary with y-axis, from :class:`~PopPUNK.models.RefineFit`
        mean0 (numpy.array)
            Centre of within-strain distribution
        mean1 (numpy.array)
            Centre of between-strain distribution
        min_move (float)
            Minimum s range
        max_move (float)
            Maximum s range
        scale (numpy.array)
            Scaling factor from :class:`~PopPUNK.models.RefineFit`
        threshold (bool)
            If fit was just from a simple thresholding
        indiv_boundaries (bool)
            Whether to draw lines for core and accessory refinement
        title (str)
            The title to display above the plot
        out_prefix (str)
            Prefix for output plot file (.png will be appended)
    """
    from .refine import transformLine

    fig=plt.figure(figsize=(11, 8), dpi= 160, facecolor='w', edgecolor='k')

    # Draw points
    plt.scatter([(X)[Y == -1, 0]], [(X)[Y == -1, 1]], .4, color='cornflowerblue')
    plt.scatter([(X)[Y == 1, 0]], [(X)[Y == 1, 1]], .4, color='c')

    # Draw fit lines
    if not threshold:
        plt.plot([x_boundary*scale[0], 0], [0, y_boundary*scale[1]], color='red', linewidth=2, linestyle='--',
                label='Combined decision boundary')
        if indiv_boundaries:
            plt.plot([core_boundary*scale[0], core_boundary*scale[0]], [0, np.amax(X[:,1])], color='darkgray', linewidth=1,
                    linestyle='-.', label='Individual decision boundaries')
            plt.plot([0, np.amax(X[:,0])], [accessory_boundary*scale[1], accessory_boundary*scale[1]], color='darkgray', linewidth=1,
                    linestyle='-.')

        # Draw boundary search range
        if mean0 is not None and mean1 is not None and min_move is not None and max_move is not None:
            if unconstrained:
                gradient = (mean1[1] - mean0[1]) / (mean1[0] - mean0[0])
                opt_start = decisionBoundary(mean0, gradient) * scale
                opt_end = decisionBoundary(mean1, gradient) * scale
                plt.fill([opt_start[0], opt_end[0], 0, 0],
                         [0, 0, opt_end[1], opt_start[1]],
                         fill=True, facecolor='lightcoral', alpha = 0.2,
                         label='Search range')
            else:
                search_length = max_move + ((mean1[0] - mean0[0])**2 + (mean1[1] - mean0[1])**2)**0.5
                minimum_xy = transformLine(-min_move, mean0, mean1) * scale
                maximum_xy = transformLine(search_length, mean0, mean1) * scale
                plt.plot([minimum_xy[0], maximum_xy[0]], [minimum_xy[1], maximum_xy[1]],
                         color='k', linewidth=1, linestyle=':', label='Search range')

            mean0 *= scale
            mean1 *= scale
            plt.plot(mean0[0], mean0[1], 'rx', label='Within-strain mean')
            plt.plot(mean1[0], mean1[1], 'r+', label='Between-strain mean')
    else:
        plt.plot([core_boundary*scale[0], core_boundary*scale[0]], [0, np.amax(X[:,1])], color='red', linewidth=2, linestyle='--',
                label='Threshold boundary')

    plt.legend(loc='lower right')
    plt.title(title)
    plt.xlabel('Core distance (' + r'$\pi$' + ')')
    plt.ylabel('Accessory distance (' + r'$a$' + ')')
    plt.savefig(out_prefix + ".png")
    plt.close()

def plot_contours(model, assignments, title, out_prefix):
    """Draw contours of mixture model assignments

    Will draw the decision boundary for between/within in red

    Args:
        model (BGMMFit)
            Model we are plotting from
        assignments (numpy.array)
            n-vectors of cluster assignments for model
        title (str)
            The title to display above the plot
        out_prefix (str)
            Prefix for output plot file (.pdf will be appended)
    """
    # avoid recursive import
    from .bgmm import log_likelihood
    from .bgmm import findWithinLabel
    from .bgmm import findBetweenLabel_bgmm

    xx, yy, xy = get_grid(0, 1, 100)

    # for likelihood boundary
    z = model.assign(xy, values=True, progress=False)
    z_diff = z[:,findWithinLabel(model.means, assignments, 0)] - z[:,findBetweenLabel_bgmm(model.means, assignments)]
    z = z_diff.reshape(xx.shape).T

    # For full likelihood surface
    z_ll, lpr = log_likelihood(xy, model.weights, model.means, model.covariances, np.array([1,1]))
    z_ll = z_ll.reshape(xx.shape).T

    plt.figure(figsize=(11, 8), dpi= 160, facecolor='w', edgecolor='k')
    plt.contour(xx, yy, z_ll, levels=np.linspace(z_ll.min(), z_ll.max(), 25))
    plt.contour(xx, yy, z, levels=[0], colors='r', linewidths=3)

    plt.title(title)
    plt.xlabel('Scaled core distance')
    plt.ylabel('Scaled accessory distance')
    plt.savefig(out_prefix + ".pdf")
    plt.close()

def get_grid(minimum, maximum, resolution):
    """Get a square grid of points to evaluate a function across

    Used for :func:`~plot_scatter` and :func:`~plot_contours`

    Args:
        minimum (float)
            Minimum value for grid
        maximum (float)
            Maximum value for grid
        resolution (int)
            Number of points along each axis
    Returns:
        xx (numpy.array)
            x values across n x n grid
        yy (numpy.array)
            y values across n x n grid
        xy (numpy.array)
            n x 2 pairs of x, y values grid is over
    """
    x = np.linspace(minimum, maximum, resolution)
    y = np.linspace(minimum, maximum, resolution)
    xx, yy = np.meshgrid(x, y)
    xy = np.vstack([yy.ravel(), xx.ravel()]).T

    return(xx, yy, xy)

def distHistogram(dists, rank, outPrefix):
    """Plot a histogram of distances (1D)

    Args:
        dists (np.array)
            Distance vector
        rank (int)
            Rank (used for name and title)
        outPrefix (int)
            Full path prefix for plot file
    """
    plt.figure(figsize=(11, 8), dpi= 160, facecolor='w', edgecolor='k')
    plt.hist(dists,
             50,
             facecolor='b',
             alpha=0.75)

    plt.title('Included nearest neighbour distances for rank ' + str(rank))
    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.grid(True)
    plt.savefig(outPrefix + \
                "_rank_" + str(rank) + "_histogram.png")
    plt.close()

def drawMST(mst, outPrefix, isolate_clustering, clustering_name, overwrite):
    """Plot a layout of the minimum spanning tree

    Args:
        mst (graph_tool.Graph)
            A minimum spanning tree
        outPrefix (str)
            Output prefix for save files
        isolate_clustering (dict)
            Dictionary of ID: cluster, used for colouring vertices
        clustering_name (str)
            Name of clustering scheme to be used for colouring
        overwrite (bool)
            Overwrite existing output files
    """
    import graph_tool.all as gt
    graph1_file_name = outPrefix + "/" + os.path.basename(outPrefix) + "_mst_stress_plot.png"
    graph2_file_name = outPrefix + "/" + os.path.basename(outPrefix) + "_mst_cluster_plot.png"
    if overwrite or not os.path.isfile(graph1_file_name) or not os.path.isfile(graph2_file_name):
        sys.stderr.write("Drawing MST\n")
        pos = gt.sfdp_layout(mst)
        if overwrite or not os.path.isfile(graph1_file_name):
            deg = mst.degree_property_map("total")
            deg.a = 4 * (np.sqrt(deg.a) * 0.5 + 0.4)
            ebet = gt.betweenness(mst)[1]
            ebet.a /= ebet.a.max() / 50.
            eorder = ebet.copy()
            eorder.a *= -1
            gt.graph_draw(mst, pos=pos, vertex_size=gt.prop_to_size(deg, mi=20, ma=50),
                            vertex_fill_color=deg, vorder=deg,
                            edge_color=ebet, eorder=eorder, edge_pen_width=ebet,
                            output=graph1_file_name, output_size=(3000, 3000))
        if overwrite or not os.path.isfile(graph2_file_name):
            cluster_fill = {}
            for cluster in set(isolate_clustering[clustering_name].values()):
                cluster_fill[cluster] = list(np.random.rand(3)) + [0.9]
            plot_color = mst.new_vertex_property('vector<double>')
            mst.vertex_properties['plot_color'] = plot_color
            for v in mst.vertices():
                plot_color[v] = cluster_fill[isolate_clustering[clustering_name][mst.vp.id[v]]]

            gt.graph_draw(mst, pos=pos, vertex_fill_color=mst.vertex_properties['plot_color'],
                    output=graph2_file_name, output_size=(3000, 3000))

def outputsForCytoscape(G, G_mst, isolate_names, clustering, outPrefix, epiCsv, queryList = None,
                        suffix = None, writeCsv = True):
    """Write outputs for cytoscape. A graphml of the network, and CSV with metadata

    Args:
        G (graph)
            The network to write
        G_mst (graph)
            The minimum spanning tree of G
        isolate_names (list)
            Ordered list of sequence names
        clustering (dict)
            Dictionary of cluster assignments (keys are nodeNames).
        outPrefix (str)
            Prefix for files to be written
        epiCsv (str)
            Optional CSV of epi data to paste in the output in addition to
            the clusters.
        queryList (list)
            Optional list of isolates that have been added as a query.
            (default = None)
        suffix (string)
            String to append to network file name.
            (default = None)
        writeCsv (bool)
            Whether to print CSV file to accompany network
    """

    # Avoid circular import
    from .network import save_network
    import graph_tool.all as gt

    # edit names
    seqLabels = isolateNameToLabel(isolate_names)
    vid = G.new_vertex_property('string',
                                vals = seqLabels)
    G.vp.id = vid

    # write graph file
    if suffix is None:
        suffix = '_cytoscape'
    else:
        suffix = suffix + '_cytoscape'
    save_network(G, prefix = outPrefix, suffix = suffix, use_graphml = True)

    # Save each component too (useful for very large graphs)
    component_assignments, component_hist = gt.label_components(G)
    for component_idx in range(len(component_hist)):
        remove_list = []
        for vidx, v_component in enumerate(component_assignments.a):
            if v_component != component_idx:
                remove_list.append(vidx)
        G_copy = G.copy()
        G_copy.remove_vertex(remove_list)
        save_network(G_copy, prefix = outPrefix, suffix = "_component_" + str(component_idx + 1), use_graphml = True)
        del G_copy

    if G_mst != None:
        isolate_labels = isolateNameToLabel(G_mst.vp.id)
        for n,v in enumerate(G_mst.vertices()):
            G_mst.vp.id[v] = isolate_labels[n]
        suffix = suffix + '_mst'
        save_network(G_mst, prefix = outPrefix, suffix = suffix, use_graphml = True)

    # Write CSV of metadata
    if writeCsv:
        writeClusterCsv(outPrefix + "/" + os.path.basename(outPrefix) + "_cytoscape.csv",
                        isolate_names,
                        seqLabels,
                        clustering,
                        'cytoscape',
                        epiCsv,
                        queryList)

def writeClusterCsv(outfile, nodeNames, nodeLabels, clustering,
                    output_format = 'microreact', epiCsv = None,
                    queryNames = None, suffix = '_Cluster'):
    """Print CSV file of clustering and optionally epi data

    Writes CSV output of clusters which can be used as input to microreact and cytoscape.
    Uses pandas to deal with CSV reading and writing nicely.

    The epiCsv, if provided, should have the node labels in the first column.

    Args:
        outfile (str)
            File to write the CSV to.
        nodeNames (list)
            Names of sequences in clustering (includes path).
        nodeLabels (list)
            Names of sequences to write in CSV (usually has path removed).
        clustering (dict or dict of dicts)
            Dictionary of cluster assignments (keys are nodeNames). Pass a dict with depth two
            to include multiple possible clusterings.
        output_format (str)
            Software for which CSV should be formatted
            (microreact, phandango, grapetree and cytoscape are accepted)
        epiCsv (str)
            Optional CSV of epi data to paste in the output in addition to
            the clusters (default = None).
        queryNames (list)
            Optional list of isolates that have been added as a query.

            (default = None)
    """
    # set order of column names
    colnames = []
    if output_format == 'microreact':
        colnames = ['id']
        for cluster_type in clustering:
            col_name = cluster_type + suffix + '__autocolour'
            colnames.append(col_name)
        if queryNames is not None:
            colnames.append('Status')
            colnames.append('Status__colour')
    elif output_format == 'phandango':
        colnames = ['id']
        for cluster_type in clustering:
            col_name = cluster_type + suffix
            colnames.append(col_name)
        if queryNames is not None:
            colnames.append('Status')
            colnames.append('Status:colour')
    elif output_format == 'grapetree':
        colnames = ['ID']
        for cluster_type in clustering:
            col_name = cluster_type + suffix
            colnames.append(col_name)
        if queryNames is not None:
            colnames.append('Status')
    elif output_format == 'cytoscape':
        colnames = ['id']
        for cluster_type in clustering:
            col_name = cluster_type + suffix
            colnames.append(col_name)
        if queryNames is not None:
            colnames.append('Status')
    else:
        sys.stderr.write("Do not recognise format for CSV writing\n")
        exit(1)

    # process epidemiological data
    d = defaultdict(list)

    # process epidemiological data without duplicating names
    # used by PopPUNK
    if epiCsv is not None:
        columns_to_be_omitted = ['id', 'Id', 'ID', 'combined_Cluster__autocolour',
        'core_Cluster__autocolour', 'accessory_Cluster__autocolour',
        'overall_Lineage']
        epiData = pd.read_csv(epiCsv, index_col = False, quotechar='"')
        epiData.index = isolateNameToLabel(epiData.iloc[:,0])
        for e in epiData.columns.values:
            if e not in columns_to_be_omitted:
                colnames.append(str(e))

    # get example clustering name for validation
    example_cluster_title = list(clustering.keys())[0]

    for name, label in zip(nodeNames, isolateNameToLabel(nodeLabels)):
        if name in clustering[example_cluster_title]:
            if output_format == 'microreact':
                d['id'].append(label)
                for cluster_type in clustering:
                    col_name = cluster_type + suffix + "__autocolour"
                    d[col_name].append(clustering[cluster_type][name])
                if queryNames is not None:
                    if name in queryNames:
                        d['Status'].append("Query")
                        d['Status__colour'].append("red")
                    else:
                        d['Status'].append("Reference")
                        d['Status__colour'].append("black")
            elif output_format == 'phandango':
                d['id'].append(label)
                for cluster_type in clustering:
                    col_name = cluster_type + suffix
                    d[col_name].append(clustering[cluster_type][name])
                if queryNames is not None:
                    if name in queryNames:
                        d['Status'].append("Query")
                        d['Status:colour'].append("#ff0000")
                    else:
                        d['Status'].append("Reference")
                        d['Status:colour'].append("#000000")
            elif output_format == 'grapetree':
                d['ID'].append(label)
                for cluster_type in clustering:
                    col_name = cluster_type + suffix
                    d[col_name].append(clustering[cluster_type][name])
                if queryNames is not None:
                    if name in queryNames:
                        d['Status'].append("Query")
                    else:
                        d['Status'].append("Reference")
            elif output_format == 'cytoscape':
                d['id'].append(label)
                for cluster_type in clustering:
                    col_name = cluster_type + suffix
                    d[col_name].append(clustering[cluster_type][name])
                if queryNames is not None:
                    if name in queryNames:
                        d['Status'].append("Query")
                    else:
                        d['Status'].append("Reference")
            if epiCsv is not None:
                if label in epiData.index:
                    if label in epiData.index:
                        for col, value in zip(epiData.columns.values, epiData.loc[[label]].iloc[0].values):
                            if col not in columns_to_be_omitted:
                                d[col].append(str(value))
                    else:
                        for col in epiData.columns.values:
                            if col not in columns_to_be_omitted:
                                d[col].append('nan')

        else:
            sys.stderr.write("Cannot find " + name + " in clustering\n")
            sys.exit(1)

    # print CSV
    sys.stderr.write("Parsed data, now writing to CSV\n")
    try:
        pd.DataFrame(data=d).to_csv(outfile, columns = colnames, index = False)
    except (ValueError,DataError) as e:
        sys.stderr.write("Problem with epidemiological data CSV; returned code: " + str(e) + "\n")
        # check CSV
        prev_col_items = -1
        prev_col_name = "unknown"
        for col in d:
            this_col_items = len(d[col])
            if prev_col_items > -1 and prev_col_items != this_col_items:
                sys.stderr.write("Discrepant length between " + prev_col_name + \
                                 " (length of " + str(prev_col_items) + ") and " + \
                                 col + "(length of " + str(this_col_items) + ")\n")
            prev_col_items = this_col_items
        sys.exit(1)

def outputsForMicroreact(combined_list, clustering, nj_tree, mst_tree, accMat, perplexity, maxIter,
                         outPrefix, epiCsv, queryList = None, overwrite = False, n_threads = 1,
                         use_gpu = False, device_id = 0):
    """Generate files for microreact

    Output a neighbour joining tree (.nwk) from core distances, a plot of t-SNE clustering
    of accessory distances (.dot) and cluster assignment (.csv)

    Args:
        combined_list (list)
            Name of sequences being analysed. The part of the name before the first '.' will
            be shown in the output
        clustering (dict or dict of dicts)
            List of cluster assignments from :func:`~PopPUNK.network.printClusters`.
            Further clusterings (e.g. 1D core only) can be included by passing these as a dict.
        nj_tree (str or None)
            String representation of a Newick-formatted NJ tree
        mst_tree (str or None)
            String representation of a Newick-formatted minimum-spanning tree
        accMat (numpy.array)
            n x n array of accessory distances for n samples.
        perplexity (int)
            Perplexity parameter passed to mandrake
        maxIter (int)
            Maximum iterations for mandrake
        outPrefix (str)
            Prefix for all generated output files, which will be placed in `outPrefix` subdirectory
        epiCsv (str)
            A CSV containing other information, to include with the CSV of clusters
        queryList (list)
            Optional list of isolates that have been added as a query for colouring in the CSV.
            (default = None)
        overwrite (bool)
            Overwrite existing output if present (default = False)
        n_threads (int)
            Number of CPU threads to use
            (default = 1)
        use_gpu (bool)
            Whether to use a GPU for t-SNE generation
        device_id (int)
            Device ID of GPU to be used
            (default = 0)
    Returns:
        outfiles (list)
            List of output files create
    """
    # Avoid recursive import
    from .mandrake import generate_embedding

    # generate sequence labels
    seqLabels = isolateNameToLabel(combined_list)

    # check CSV before calculating other outputs
    outfiles = [outPrefix + "/" + os.path.basename(outPrefix) + "_microreact_clusters.csv"]
    writeClusterCsv(outPrefix + "/" + os.path.basename(outPrefix) + "_microreact_clusters.csv",
                        combined_list, combined_list, clustering, 'microreact', epiCsv, queryList)

    # write the phylogeny .nwk; t-SNE network .dot; clusters + data .csv
    embedding_file = generate_embedding(seqLabels, accMat, perplexity, outPrefix, overwrite,
                       kNN=100, maxIter=maxIter, n_threads=n_threads,
                       use_gpu=use_gpu, device_id=device_id)
    outfiles.append(embedding_file)

    # write NJ tree
    if nj_tree is not None:
        write_tree(nj_tree, outPrefix, "_core_NJ.nwk", overwrite)
        outfiles.append(outPrefix + "/" + os.path.basename(outPrefix) + "_core_NJ.nwk")

    # write MST
    if mst_tree is not None:
        write_tree(mst_tree, outPrefix, "_MST.nwk", overwrite)
        outfiles.append(outPrefix + "/" + os.path.basename(outPrefix) + "_MST.nwk")

    return outfiles

def createMicroreact(prefix, microreact_files, api_key=None):
    """Creates a .microreact file, and instance via the API

    Args:
        prefix (str)
            Prefix for output file
        microreact_files (str)
            List of Microreact files [clusters, dot, tree, mst_tree]
        api_key (str)
            API key for your account
    """
    import pkg_resources
    import pickle
    import requests
    import json
    from datetime import datetime

    microreact_api_new_url = "https://microreact.org/api/projects/create"
    description_string = "PopPUNK run on " + datetime.now().strftime("%Y-%b-%d %H:%M")
    # Load example JSON to be modified
    with pkg_resources.resource_stream(__name__, 'data/microreact_example.pkl') as example_pickle:
        json_pickle = pickle.load(example_pickle)
    json_pickle["meta"]["name"] = description_string

    # Read data in
    with open(microreact_files[0]) as cluster_file:
        csv_string = cluster_file.read()
        json_pickle["files"]["data-file-1"]["blob"] = csv_string
    with open(microreact_files[1], 'r') as dot_file:
        dot_string = dot_file.read()
        json_pickle["files"]["network-file-1"] = {"id": "network-file-1",
                                                  "name": "network.dot",
                                                  "format": "text/vnd.graphviz",
                                                  "blob": dot_string}
        json_pickle["networks"]["network-1"] = {"title": "Network",
                                                "file": "network-file-1",
                                                "nodeField": "id"}
    if len(microreact_files) > 2:
        with open(microreact_files[2], 'r') as tree_file:
            tree_string = tree_file.read()
            json_pickle["files"]["tree-file-1"]["blob"] = tree_string
    else:
        del json_pickle["files"]["tree-file-1"]

    with open(prefix + "/" + os.path.basename(prefix) + ".microreact", 'w') as json_file:
        json.dump(json_pickle, json_file)

    url = None
    if api_key != None:
        headers = {"Content-type": "application/json; charset=UTF-8",
                   "Access-Token": api_key}
        r = requests.post(microreact_api_new_url, data=json.dumps(json_pickle), headers=headers)
        if not r.ok:
            if r.status_code == 400:
                sys.stderr.write("Microreact API call failed with response " + r.text + "\n")
            else:
                sys.stderr.write("Microreact API call failed with unknown response code " + str(r.status_code) + "\n")
        else:
            url = r.json()['url']

    return url

def outputsForPhandango(combined_list, clustering, nj_tree, mst_tree, outPrefix, epiCsv,
                        queryList = None, overwrite = False):
    """Generate files for Phandango

    Write a neighbour joining tree (.tree) from core distances
    and cluster assignment (.csv)

    Args:
        combined_list (list)
            Name of sequences being analysed. The part of the name before the first '.' will
            be shown in the output
        clustering (dict or dict of dicts)
            List of cluster assignments from :func:`~PopPUNK.network.printClusters`.
            Further clusterings (e.g. 1D core only) can be included by passing these as a dict.
        nj_tree (str or None)
            String representation of a Newick-formatted NJ tree
        mst_tree (str or None)
            String representation of a Newick-formatted minimum-spanning tree
        outPrefix (str)
            Prefix for all generated output files, which will be placed in `outPrefix` subdirectory
        epiCsv (str)
            A CSV containing other information, to include with the CSV of clusters
        queryList (list)
            Optional list of isolates that have been added as a query for colouring in the CSV.
            (default = None)
        overwrite (bool)
            Overwrite existing output if present (default = False)
        threads (int)
            Number of threads to use with rapidnj
    """
    # print clustering file
    writeClusterCsv(outPrefix + "/" + os.path.basename(outPrefix) + "_phandango_clusters.csv",
                    combined_list, combined_list, clustering, 'phandango', epiCsv, queryList)

    # write NJ tree
    if nj_tree is not None:
        write_tree(nj_tree, outPrefix, "_core_NJ.tree", overwrite)
    else:
        sys.stderr.write("Need an NJ tree for a Phandango output")

def outputsForGrapetree(combined_list, clustering, nj_tree, mst_tree, outPrefix, epiCsv,
                        queryList = None, overwrite = False):
    """Generate files for Grapetree

    Write a neighbour joining tree (.nwk) from core distances
    and cluster assignment (.csv)

    Args:
        combined_list (list)
            Name of sequences being analysed. The part of the name before the
            first '.' will be shown in the output
        clustering (dict or dict of dicts)
            List of cluster assignments from :func:`~PopPUNK.network.printClusters`.
            Further clusterings (e.g. 1D core only) can be included by passing these
            as a dict.
        nj_tree (str or None)
            String representation of a Newick-formatted NJ tree
        mst_tree (str or None)
            String representation of a Newick-formatted minimum-spanning tree
        outPrefix (str)
            Prefix for all generated output files, which will be placed in `outPrefix`
            subdirectory.
        epiCsv (str)
            A CSV containing other information, to include with the CSV of clusters
        queryList (list)
            Optional list of isolates that have been added as a query for colouring
            in the CSV. (default = None)
        overwrite (bool)
            Overwrite existing output if present (default = False).
    """
    # print clustering file
    writeClusterCsv(outPrefix + "/" + os.path.basename(outPrefix) + "_grapetree_clusters.csv",
                    combined_list, combined_list, clustering, 'grapetree', epiCsv, queryList)

    # calculate phylogeny, or copy existing microreact file
    # write NJ tree
    if nj_tree is not None:
        write_tree(nj_tree, outPrefix, "_core_NJ.nwk", overwrite)

    # write MST
    if mst_tree is not None:
        write_tree(mst_tree, outPrefix, "_core_MST.nwk", overwrite)
