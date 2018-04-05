'''Plots of GMM results, k-mer fits, and microreact output'''

import sys
import os
import subprocess
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
# for microreact
from scipy import spatial
from sklearn import manifold
import dendropy
import networkx as nx

#####################################
# Parse epidemiological information #
#####################################

def readEpiFile(epiCsv):
    # data structures
    epi = {}
    epiHeader = []
    # parse file
    with open(epiCsv.rstrip(), 'r') as eFile:
        for line in eFile:
            data = line.rstrip().split(',')
            id = data.pop(0)
            if id == "id":
                epiHeader = data
            else:
                if len(data) == len(epiHeader):
                    epi[id] = data
                else:
                    sys.stderr.write("Incorrect number of fields in CSV for line with id " \
                                     + id + " - header must start with 'id'")
                    sys.exit(1)

    if len(epiHeader) == 0:
        sys.stderr.write("Unable to find header line starting with 'Id'")
        sys.exit(1)
    missingString = (','*len(epiHeader))[:len(epiHeader)]

    return epi, epiHeader, missingString

##############################
# Write output for Cytoscape #
##############################

def outputsForCytoscape(G, clustering, outPrefix, epiCsv):

    # write graph file
    nx.write_graphml(G, "./" + outPrefix + "/" + outPrefix + "_cytoscape.graphml")

    # read epi info if provided
    if epiCsv is not None:
        epi, epiHeader, missingString = readEpiFile(epiCsv)

    # write annotation file for Cytoscape
    with open(outPrefix + "/" + outPrefix + "_cytoscape.csv", 'w') as cFile:
        cFile.write("id,Cluster")
        if epiCsv is not None:
            cFile.write(','+','.join(str(e) for e in epiHeader))
        cFile.write("\n")
        for unique in G.nodes(data=False):
            if unique in clustering:
                cFile.write(unique + ',' + str(clustering[unique]))
                if epiCsv is not None:
                    if unique in epi.keys():
                        cFile.write(','+','.join(str(e) for e in epi[unique]))
                    else:
                        cFile.write(missingString)
                cFile.write("\n")
            else:
                sys.stderr.write("Cannot find " + unique + " in clustering\n")
                sys.exit(1)

############################################
# Use rapidNJ for more rapid tree building #
############################################

def buildRapidNJ(rapidnj, refList, coreMat, outPrefix, tree_filename):

    # generate phylip matrix
    phylip_name = "./" + outPrefix + "/" + outPrefix + "_core_distances.phylip"
    with open(phylip_name, 'w') as pFile:
        pFile.write(str(len(refList))+"\n")
        for r in range(len(refList)):
            namePrefix = refList[r].split('.')[0]
            pFile.write(namePrefix)
            pFile.write(' '+' '.join(map(str,coreMat[r,])))
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
        sys.stderr.write("Could not run command " + tree_info + "; returned: " + e.message + "\n")
        sys.exit(1)

    # read tree and return
    tree = dendropy.Tree.get(path=tree_filename, schema="newick")
    return tree

def outputsForMicroreact(refList, distMat, clustering, perplexity, outPrefix, epiCsv, rapidnj, overwrite = False):
    """Generate files for microreact

    Output a neighbour joining tree (.nwk) from core distances, a plot of t-SNE clustering
    of accessory distances (.dot) and cluster assignment (.csv)

    Args:
        refList (list)
            Name of reference sequences. The part of the name before the first '.' will
            be shown in the output
        distMat (numpy.array)
            n x 2 array of core and accessory distances for n samples.
        clustering (list)
            List of cluster assignments from :func:`~PopPUNK.network.printClusters`
        perplexity (int)
            Perplexity parameter passed to t-SNE
        outPrefix (str)
            Prefix for all generated output files, which will be placed in `outPrefix` subdirectory
        epiCsv (str)
            A CSV containing other information, to include with the CSV of clusters
        rapidnj (str)
            A string with the location of the rapidnj executable for tree-building. If None, will
            use dendropy by default
        overwrite (bool)
            Overwrite existing output if present (default = False)
    """

    # avoid recursive import
    from .mash import iterDistRows

    sys.stderr.write("writing microreact output:\n")
    seqLabels = [r.split('.')[0] for r in refList]

    coreMat = np.zeros((len(refList), len(refList)))
    accMat = np.zeros((len(refList), len(refList)))

    # Fill in symmetric matrices
    i = 0
    j = 1
    for row, (ref, query) in enumerate(iterDistRows(refList, refList, self=True)):
        coreMat[i, j] = distMat[row, 0]
        coreMat[j, i] = coreMat[i, j]
        accMat[i, j] = distMat[row, 1]
        accMat[j, i] = accMat[i, j]

        if j == len(refList) - 1:
            i += 1
            j = i + 1
        else:
            j += 1

    core_dist_file = outPrefix + "/" + outPrefix + "_core_dists.csv"
    np.savetxt(core_dist_file, coreMat, delimiter=",", header = ",".join(seqLabels), comments="")
    acc_dist_file = outPrefix + "/" + outPrefix + "_acc_dists.csv"
    np.savetxt(acc_dist_file, accMat, delimiter=",", header = ",".join(seqLabels), comments="")

    # calculate phylogeny
    tree_filename = outPrefix + "/" + outPrefix + "_core_NJ_microreact.nwk"
    if overwrite or not os.path.isfile(tree_filename):
        sys.stderr.write("Building phylogeny\n")
        pdm = dendropy.PhylogeneticDistanceMatrix.from_csv(src=open(core_dist_file),
                                                           delimiter=",",
                                                           is_first_row_column_names=True,
                                                           is_first_column_row_names=False)
        if rapidnj is not None:
            tree = buildRapidNJ(rapidnj, refList, coreMat, outPrefix, tree_filename)
        else:
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


    # generate accessory genome distance representation
    tsne_filename = outPrefix + "/" + outPrefix + "_perplexity" + str(perplexity) + "_accessory_tsne.dot"
    if overwrite or not os.path.isfile(tsne_filename):
        sys.stderr.write("Running t-SNE\n")
        accArray_embedded = manifold.TSNE(n_components=2, perplexity=perplexity).fit_transform(np.array(accMat))

        # print dot file
        with open(tsne_filename, 'w') as nFile:
            nFile.write("graph G { ")
            for s, seqLabel in enumerate(seqLabels):
                nFile.write('"' + seqLabel + '"' +
                        '[x='+str(5*float(accArray_embedded[s][0]))+',y='+str(5*float(accArray_embedded[s][1]))+']; ')
            nFile.write("}\n")
    else:
        sys.stderr.write("t-SNE analysis already exists; add --overwrite to replace\n")

    # read epidemiological information if provided
    rawepi = {}
    epiHeader = []
    missingString = ""
    epi = {}
    if epiCsv is not None:
        rawepi, epiHeader, missingString = readEpiFile(epiCsv)
        # prune off suffix
        for label in rawepi.keys():
            newlabel = label.split('.')[0]
            epi[newlabel] = rawepi[label]
        del rawepi

    # print clustering file
    with open(outPrefix + "/" + outPrefix + "_microreact_clusters.csv", 'w') as cFile:
        cFile.write("id,Cluster__autocolour")
        if epiCsv is not None:
            cFile.write(','+','.join(str(e) for e in epiHeader))
        cFile.write("\n")
        for label, unique in zip(seqLabels, refList):
            if unique in clustering:
                cFile.write(label + ',' + str(clustering[unique]))
                if epiCsv is not None:
                    if label in epi.keys():
                        cFile.write(','+','.join(str(e) for e in epi[label]))
                    else:
                        cFile.write(missingString)
                cFile.write("\n")
            else:
                sys.stderr.write("Cannot find " + unique + " in clustering\n")
                sys.exit(1)

def plot_scatter(X, out_prefix, title):
    """Draws a 2D scatter plot (png) of the core and accessory distances

    Args:
        X (numpy.array)
            n x 2 array of core and accessory distances for n samples.
        out_prefix (str)
            Prefix for output plot file (.png will be appended)
        title (str)
            The title to display above the plot
    """
    plt.ioff()
    plt.title(title)
    plt.scatter(X[:,0].flat, X[:,1].flat, s=0.8)
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
            Component means from :func:`~PopPUNK.bgmm.fit2dMultiGaussian`
        covars (numpy.array)
            Component covariances from :func:`~PopPUNK.bgmm.fit2dMultiGaussian`
        scale (numpy.array)
            Scaling factor from :func:`~PopPUNK.bgmm.fit2dMultiGaussian`
        out_prefix (str)
            Prefix for output plot file (.png will be appended)
        title (str)
            The title to display above the plot
    """
    color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold','darkorange'])
    fig=plt.figure(figsize=(22, 16), dpi= 160, facecolor='w', edgecolor='k')
    splot = plt.subplot(1, 1, 1)
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = np.linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y == i):
            continue
        plt.scatter([(X/scale)[Y == i, 0]], [(X/scale)[Y == i, 1]], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.title(title)
    plt.savefig(out_prefix + ".png")
    plt.close()
