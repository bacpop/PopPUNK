# vim: set fileencoding=<utf-8> :
'''Plots of GMM results, k-mer fits, and microreact output'''

import sys
import os
import subprocess
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams.update({'font.size': 18})
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import itertools
# for other outputs
from shutil import copyfile
import pandas as pd
from collections import defaultdict
from scipy import spatial
from sklearn import manifold
from sklearn.neighbors.kde import KernelDensity
import dendropy
import networkx as nx

def outputsForCytoscape(G, clustering, outPrefix, epiCsv, queryList = None, suffix = None, writeCsv = True):
    """Write outputs for cytoscape. A graphml of the network, and CSV with metadata

    Args:
        G (networkx.Graph)
            The network to write from :func:`~PopPUNK.network.constructNetwork`
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
    # write graph file
    if suffix is None:
        graph_file_name = os.path.basename(outPrefix) + "_cytoscape.graphml"
    else:
        graph_file_name = os.path.basename(outPrefix) + "_" + suffix + "_cytoscape.graphml"
    nx.write_graphml(G, outPrefix + "/" + graph_file_name)

    # Write CSV of metadata
    if writeCsv:
        refNames = G.nodes(data=False)
        seqLabels = [r.split('/')[-1].split('.')[0] for r in refNames]
        writeClusterCsv(outPrefix + "/" + outPrefix + "_cytoscape.csv",
                        refNames,
                        seqLabels,
                        clustering,
                        'cytoscape',
                        epiCsv,
                        queryList)

def writeClusterCsv(outfile, nodeNames, nodeLabels, clustering, output_format = 'microreact', epiCsv = None, queryNames = None):
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
            col_name = cluster_type + '_Cluster__autocolour'
            colnames.append(col_name)
        if queryNames is not None:
            colnames.append('Status')
            colnames.append('Status__colour')
    elif output_format == 'phandango':
        colnames = ['id']
        for cluster_type in clustering:
            col_name = cluster_type + '_Cluster'
            colnames.append(col_name)
        if queryNames is not None:
            colnames.append('Status')
            colnames.append('Status:colour')
    elif output_format == 'grapetree':
        colnames = ['ID']
        for cluster_type in clustering:
            col_name = cluster_type + '_Cluster'
            colnames.append(col_name)
        if queryNames is not None:
            colnames.append('Status')
    elif output_format == 'cytoscape':
        colnames = ['id']
        for cluster_type in clustering:
            col_name = cluster_type + '_Cluster'
            colnames.append(col_name)
        if queryNames is not None:
            colnames.append('Status')
    else:
        sys.stderr.write("Do not recognise format for CSV writing")
        exit(1)

    # process epidemiological data
    if epiCsv is not None:
        epiData = pd.read_csv(epiCsv, index_col = 0, quotechar='"')

    d = defaultdict(list)
    if epiCsv is not None:
        for e in epiData.columns.values:
            colnames.append(str(e))

    columns_to_be_omitted = []

    # process clustering data
    nodeLabels = [r.split('/')[-1].split('.')[0] for r in nodeNames]

    for name, label in zip(nodeNames, nodeLabels):
        if name in clustering['combined']:
            if output_format == 'microreact':
                d['id'].append(label)
                for cluster_type in clustering:
                    col_name = cluster_type + "_Cluster__autocolour"
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
                    col_name = cluster_type + "_Cluster"
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
                    col_name = cluster_type + "_Cluster"
                    d[col_name].append(clustering[cluster_type][name])
                if queryNames is not None:
                    if name in queryNames:
                        d['Status'].append("Query")
                    else:
                        d['Status'].append("Reference")
            elif output_format == 'cytoscape':
                d['id'].append(name)
                for cluster_type in clustering:
                    col_name = cluster_type + "_Cluster"
                    d[col_name].append(clustering[cluster_type][name])
                if queryNames is not None:
                    if name in queryNames:
                        d['Status'].append("Query")
                    else:
                        d['Status'].append("Reference")
            if epiCsv is not None:
                # avoid adding
                if len(columns_to_be_omitted) == 0:
                    columns_to_be_omitted = ['id', 'Id', 'ID', 'combined_Cluster__autocolour',
                                             'core_Cluster__autocolour', 'accessory_Cluster__autocolour']
                    for c in d:
                        columns_to_be_omitted.append(c)
                if label in epiData.index:
                    for col, value in zip(epiData.columns.values, epiData.loc[label].values):
                        if col not in columns_to_be_omitted:
                            d[col].append(str(value))
                else:
                    for col in colnames:
                        d[col].append('nan')
        else:
            sys.stderr.write("Cannot find " + name + " in clustering\n")
            sys.exit(1)

    # print CSV
    pd.DataFrame(data=d).to_csv(outfile, columns = colnames, index = False)

def buildRapidNJ(rapidnj, refList, coreMat, outPrefix, tree_filename):
    """Use rapidNJ for more rapid tree building

    Creates a phylip of core distances, system call to rapidnj executable, loads tree as
    dendropy object (cleaning quotes in node names), removes temporary files.

    Args:
        rapidnj (str)
            Location of rapidnj executable
        refList (list)
            Names of sequences in coreMat (same order)
        coreMat (numpy.array)
            NxN core distance matrix produced in :func:`~outputsForMicroreact`
        outPrefix (int)
            Output prefix for temporary files
        outPrefix (str)
            Prefix for all generated output files, which will be placed in `outPrefix` subdirectory
        tree_filename (str)
            Filename for output tree (saved to disk)

    Returns:
        tree (dendropy.Tree)
            NJ tree from core distances
    """
    # generate phylip matrix
    phylip_name = outPrefix + "/" + os.path.basename(outPrefix) + "_core_distances.phylip"
    with open(phylip_name, 'w') as pFile:
        pFile.write(str(len(refList))+"\n")
        for coreDist, ref in zip(coreMat, refList):
            pFile.write(ref)
            pFile.write(' '+' '.join(map(str, coreDist)))
            pFile.write("\n")

    # construct tree
    rapidnj_cmd = rapidnj + " " + phylip_name + " -i pd -o t -x " + tree_filename + ".raw"
    try:
        # run command
        subprocess.run(rapidnj_cmd, shell=True, check=True)

        # remove quotation marks for microreact
        with open(tree_filename + ".raw", 'r') as f, open(tree_filename, 'w') as fo:
            for line in f:
                fo.write(line.replace("'", ''))
        # tidy unnecessary files
        os.remove(tree_filename+".raw")
        os.remove(phylip_name)

    # record errors
    except subprocess.CalledProcessError as e:
        sys.stderr.write("Could not run command " + rapidnj_cmd + "; returned code: " + str(e.returncode) + "\n")
        sys.exit(1)

    # read tree and return
    tree = dendropy.Tree.get(path=tree_filename, schema="newick")
    return tree

def plot_scatter(X, scale, out_prefix, title, kde = True):
    """Draws a 2D scatter plot (png) of the core and accessory distances

    Also draws contours of kernel density estimare

    Args:
        X (numpy.array)
            n x 2 array of core and accessory distances for n samples.
        scale (numpy.array)
            Scaling factor from :class:`~PopPUNK.models.BGMMFit`
        out_prefix (str)
            Prefix for output plot file (.png will be appended)
        title (str)
            The title to display above the plot
        kde (bool)
            Whether to draw kernel density estimate contours

            (default = True)
    """
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
        plt.contour(xx*scale[0], yy*scale[1], z, levels=levels[1:], cmap='plasma')
        scatter_alpha = 1
    else:
        scatter_alpha = 0.1

    plt.scatter(X[:,0]*scale[0].flat, X[:,1]*scale[1].flat, s=1, alpha=scatter_alpha)

    plt.title(title)
    plt.xlabel('Core distance (' + r'$\pi$' + ')')
    plt.ylabel('Accessory distance (' + r'$a$' + ')')
    plt.savefig(out_prefix + ".png")
    plt.close()

def plot_fit(klist, matching, fit, out_prefix, title):
    """Draw a scatter plot (pdf) of k-mer sizes vs match probability, and the
    fit used to assign core and accessory distance

    K-mer sizes on x-axis, log(pr(match)) on y - expect a straight line fit
    with intercept representing accessory distance and slope core distance

    Args:
        klist (list)
            List of k-mer sizes
        matching (list)
            Proportion of matching k-mers at each klist value
        kfit (numpy.array)
            Fit to klist and matching from :func:`~PopPUNK.mash.fitKmerCurve`
        out_prefix (str)
            Prefix for output plot file (.pdf will be appended)
        title (str)
            The title to display above the plot
    """
    k_fit = np.linspace(0, klist[-1], num = 100)
    matching_fit = (1 - fit[1]) * np.power((1 - fit[0]), k_fit)

    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_xlabel('k-mer length')
    ax.set_ylabel('Proportion of matches')

    plt.tight_layout()
    plt.plot(klist, matching, 'o')
    plt.plot(k_fit, matching_fit, 'r-')

    plt.title(title)
    plt.savefig(out_prefix + ".pdf", bbox_inches='tight')
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
        ell = mpl.patches.Ellipse(mean*scale, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.title(title)
    plt.xlabel('Core distance (' + r'$\pi$' + ')')
    plt.ylabel('Accessory distance (' + r'$a$' + ')')
    plt.savefig(out_prefix + ".png")
    plt.close()

def plot_dbscan_results(X, y, n_clusters, out_prefix):
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
    """
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
        mean0, mean1, start_point, min_move, max_move, scale, indiv_boundaries, title, out_prefix):
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
        start_point (numpy.array)
            Start point of optimisation
        min_move (float)
            Minimum s range
        max_move (float)
            Maximum s range
        scale (numpy.array)
            Scaling factor from :class:`~PopPUNK.models.RefineFit`
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
    plt.plot([x_boundary*scale[0], 0], [0, y_boundary*scale[1]], color='red', linewidth=2, linestyle='--',
              label='Combined decision boundary')
    if indiv_boundaries:
        plt.plot([core_boundary*scale[0], core_boundary*scale[0]], [0, np.amax(X[:,1])], color='darkgray', linewidth=1,
                linestyle='-.', label='Individual decision boundaries')
        plt.plot([0, np.amax(X[:,0])], [accessory_boundary*scale[1], accessory_boundary*scale[1]], color='darkgray', linewidth=1,
                linestyle='-.')

    minimum_xy = transformLine(-min_move, start_point, mean1) * scale
    maximum_xy = transformLine(max_move, start_point, mean1) * scale
    plt.plot([minimum_xy[0], maximum_xy[0]], [minimum_xy[1], maximum_xy[1]],
              color='k', linewidth=1, linestyle=':', label='Search range')
    start_point *= scale
    plt.plot(start_point[0], start_point[1], 'ro', label='Initial boundary')

    mean0 *= scale
    mean1 *= scale
    plt.plot(mean0[0], mean0[1], 'rx', label='Within-strain mean')
    plt.plot(mean1[0], mean1[1], 'r+', label='Between-strain mean')

    plt.legend()
    plt.title(title)
    plt.xlabel('Core distance (' + r'$\pi$' + ')')
    plt.ylabel('Accessory distance (' + r'$a$' + ')')
    plt.savefig(out_prefix + ".png")
    plt.close()

def plot_contours(assignments, weights, means, covariances, title, out_prefix):
    """Draw contours of mixture model assignments

    Will draw the decision boundary for between/within in red

    Args:
        assignments (numpy.array)
             n-vectors of cluster assignments for model
        weights (numpy.array)
            Component weights from :class:`~PopPUNK.models.BGMMFit`
        means (numpy.array)
            Component means from :class:`~PopPUNK.models.BGMMFit`
        covars (numpy.array)
            Component covariances from :class:`~PopPUNK.models.BGMMFit`
        title (str)
            The title to display above the plot
        out_prefix (str)
            Prefix for output plot file (.pdf will be appended)
    """
    # avoid recursive import
    from .bgmm import assign_samples
    from .bgmm import log_likelihood
    from .bgmm import findWithinLabel

    xx, yy, xy = get_grid(0, 1, 100)

    # for likelihood boundary
    z = assign_samples(xy, weights, means, covariances, np.array([1,1]), True)
    z_diff = z[:,findWithinLabel(means, assignments, 0)] - z[:,findWithinLabel(means, assignments, 1)]
    z = z_diff.reshape(xx.shape).T

    # For full likelihood surface
    z_ll, lpr = log_likelihood(xy, weights, means, covariances, np.array([1,1]))
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


def outputsForMicroreact(combined_list, coreMat, accMat, clustering, perplexity, outPrefix, epiCsv,
                         rapidnj, queryList = None, overwrite = False):
    """Generate files for microreact

    Output a neighbour joining tree (.nwk) from core distances, a plot of t-SNE clustering
    of accessory distances (.dot) and cluster assignment (.csv)

    Args:
        combined_list (list)
            Name of sequences being analysed. The part of the name before the first '.' will
            be shown in the output
        coreMat (numpy.array)
            n x n array of core distances for n samples.
        accMat (numpy.array)
            n x n array of accessory distances for n samples.
        clustering (dict or dict of dicts)
            List of cluster assignments from :func:`~PopPUNK.network.printClusters`.
            Further clusterings (e.g. 1D core only) can be included by passing these as a dict.
        perplexity (int)
            Perplexity parameter passed to t-SNE
        outPrefix (str)
            Prefix for all generated output files, which will be placed in `outPrefix` subdirectory
        epiCsv (str)
            A CSV containing other information, to include with the CSV of clusters
        rapidnj (str)
            A string with the location of the rapidnj executable for tree-building. If None, will
            use dendropy by default
        queryList (list)
            Optional list of isolates that have been added as a query for colouring in the CSV.
            (default = None)
        overwrite (bool)
            Overwrite existing output if present (default = False)
    """
    # Avoid recursive import
    from .tsne import generate_tsne

    # generate sequence labels
    seqLabels = [r.split('/')[-1].split('.')[0] for r in combined_list]

    # write the phylogeny .nwk; t-SNE network .dot; clusters + data .csv
    generate_phylogeny(coreMat, seqLabels, outPrefix, "_core_NJ.nwk", rapidnj, overwrite)
    generate_tsne(seqLabels, accMat, perplexity, outPrefix, overwrite)
    writeClusterCsv(outPrefix + "/" + os.path.basename(outPrefix) + "_microreact_clusters.csv",
                    combined_list, combined_list, clustering, 'microreact', epiCsv, queryList)

def generate_phylogeny(coreMat, seqLabels, outPrefix, tree_suffix, rapidnj, overwrite):
    """Generate phylogeny using dendropy or RapidNJ

    Writes a neighbour joining tree (.nwk) from core distances.

    Args:
        coreMat (numpy.array)
            n x n array of core distances for n samples.
        seqLabels (list)
            Processed names of sequences being analysed.
        outPrefix (str)
            Prefix for all generated output files, which will be placed in `outPrefix` subdirectory
        tree_suffix (str)
            String to append to tree file name
        rapidnj (str)
            A string with the location of the rapidnj executable for tree-building. If None, will
            use dendropy by default
        overwrite (bool)
            Overwrite existing output if present (default = False)
    """
    # Save distances to file
    core_dist_file = outPrefix + "/" + os.path.basename(outPrefix) + "_core_dists.csv"
    np.savetxt(core_dist_file, coreMat, delimiter=",", header = ",".join(seqLabels), comments="")

    # calculate phylogeny
    tree_filename = outPrefix + "/" + os.path.basename(outPrefix) + tree_suffix
    if overwrite or not os.path.isfile(tree_filename):
        sys.stderr.write("Building phylogeny\n")
        if rapidnj is not None:
            tree = buildRapidNJ(rapidnj, seqLabels, coreMat, outPrefix, tree_filename)
        else:
            pdm = dendropy.PhylogeneticDistanceMatrix.from_csv(src=open(core_dist_file),
                                                               delimiter=",",
                                                               is_first_row_column_names=True,
                                                               is_first_column_row_names=False)
            tree = pdm.nj_tree()

        # Not sure why, but seems that this needs to be run twice to get
        # what I would think of as a midpoint rooted tree
        tree.reroot_at_midpoint(update_bipartitions=True, suppress_unifurcations=False)
        tree.reroot_at_midpoint(update_bipartitions=True, suppress_unifurcations=False)
        tree.write(path=tree_filename,
                   schema="newick",
                   suppress_rooting=True,
                   unquoted_underscores=True)
    else:
        sys.stderr.write("NJ phylogeny already exists; add --overwrite to replace\n")

    # remove file as it can be large
    os.remove(core_dist_file)

def outputsForPhandango(combined_list, coreMat, clustering, outPrefix, epiCsv, rapidnj,
                        queryList = None, overwrite = False, microreact = False):
    """Generate files for Phandango

    Write a neighbour joining tree (.tree) from core distances
    and cluster assignment (.csv)

    Args:
        combined_list (list)
            Name of sequences being analysed. The part of the name before the first '.' will
            be shown in the output
        coreMat (numpy.array)
            n x n array of core distances for n samples.
        clustering (dict or dict of dicts)
            List of cluster assignments from :func:`~PopPUNK.network.printClusters`.
            Further clusterings (e.g. 1D core only) can be included by passing these as a dict.
        outPrefix (str)
            Prefix for all generated output files, which will be placed in `outPrefix` subdirectory
        epiCsv (str)
            A CSV containing other information, to include with the CSV of clusters
        rapidnj (str)
            A string with the location of the rapidnj executable for tree-building. If None, will
            use dendropy by default
        queryList (list)
            Optional list of isolates that have been added as a query for colouring in the CSV.
            (default = None)
        overwrite (bool)
            Overwrite existing output if present (default = False)
        microreact (bool)
            Avoid regenerating tree if already built for microreact (default = False)
    """
    # generate sequence labels
    seqLabels = [r.split('/')[-1].split('.')[0] for r in combined_list]

    # calculate phylogeny, or copy existing microreact file
    microreact_tree_filename = outPrefix + "/" + os.path.basename(outPrefix) + "_core_NJ.nwk"
    phandango_tree_filename = outPrefix + "/" + os.path.basename(outPrefix) + "_core_NJ.tree"
    if microreact and os.path.isfile(microreact_tree_filename):
        sys.stderr.write('Copying microreact tree')
        copyfile(microreact_tree_filename, phandango_tree_filename)
    else:
        generate_phylogeny(coreMat, seqLabels, outPrefix, "_core_NJ.tree", rapidnj, overwrite)

    # print clustering file
    writeClusterCsv(outPrefix + "/" + os.path.basename(outPrefix) + "_phandango_clusters.csv",
                    combined_list, combined_list, clustering, 'phandango', epiCsv, queryList)

def outputsForGrapetree(combined_list, coreMat, clustering, outPrefix, epiCsv, rapidnj,
                        queryList = None, overwrite = False, microreact = False):
    """Generate files for Grapetree

    Write a neighbour joining tree (.nwk) from core distances
    and cluster assignment (.csv)

    Args:
        combined_list (list)
            Name of sequences being analysed. The part of the name before the
            first '.' will be shown in the output
        coreMat (numpy.array)
            n x n array of core distances for n samples.
        clustering (dict or dict of dicts)
            List of cluster assignments from :func:`~PopPUNK.network.printClusters`.
            Further clusterings (e.g. 1D core only) can be included by passing these
            as a dict.
        outPrefix (str)
            Prefix for all generated output files, which will be placed in `outPrefix`
            subdirectory.
        epiCsv (str)
            A CSV containing other information, to include with the CSV of clusters
        rapidnj (str)
            A string with the location of the rapidnj executable for tree-building.
            If None, will use dendropy by default
        queryList (list)
            Optional list of isolates that have been added as a query for colouring
            in the CSV.

            (default = None)
        overwrite (bool)
            Overwrite existing output if present (default = False).
        microreact (bool)
            Avoid regenerating tree if already built for microreact (default = False).
    """
    # generate sequence labels
    seqLabels = [r.split('/')[-1].split('.')[0] for r in combined_list]

    # calculate phylogeny, or copy existing microreact file
    microreact_tree_filename = outPrefix + "/" + os.path.basename(outPrefix) + "_core_NJ.nwk"
    if microreact and os.path.isfile(microreact_tree_filename):
        sys.stderr.write('Using microreact tree')
    else:
        generate_phylogeny(coreMat, seqLabels, outPrefix, "_core_NJ.nwk", rapidnj, overwrite)

    # print clustering file
    writeClusterCsv(outPrefix + "/" + os.path.basename(outPrefix) + "_grapetree_clusters.csv",
                    combined_list, combined_list, clustering, 'grapetree', epiCsv, queryList)

