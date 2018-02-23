'''Plot GMM results'''

import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
# for microreact
from scipy import spatial
from sklearn import manifold
import Bio
from Bio.Phylo import TreeConstruction
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
from Bio.Phylo.TreeConstruction import _Matrix
from Bio.Phylo.TreeConstruction import _DistanceMatrix as DM

#################################
# Generate files for microreact #
#################################

def outputsForMicroreact(refList, queryList, distMat, clustering, outPrefix):

    sys.stderr.write("Getting unique sequences\n")
    uniqueSeq = list(set(refList))
    seqLabels = [r.split('.')[0] for r in uniqueSeq]

    sys.stderr.write("Converting to matrix\n")
    coreMat = np.zeros((len(uniqueSeq), len(uniqueSeq)))
    accMat = np.zeros((len(uniqueSeq), len(uniqueSeq)))

    for row, (ref, query) in enumerate(zip(refList, queryList)):
        i = uniqueSeq.index(ref)
        j = uniqueSeq.index(query)
        if i != j:
            coreMat[i,j] = distMat[row, 0]
            accMat[i,j] = distMat[row, 1]

    sys.stderr.write("Making triangular\n")
    core_dist_tri = []
    for row in range(len(uniqueSeq)):
        core_dist_tri.append([])
        for col in range(row + 1):
            core_dist_tri[row].append(coreMat[row, col])
    core_dist_matrix = _Matrix(uniqueSeq, core_dist_tri)
    new_matrix = DM(names=seqLabels, matrix=core_dist_tri)

    sys.stderr.write("Building phylogeny\n")
    # calculate phylogeny
    constructor = DistanceTreeConstructor()
    tree = constructor.nj(new_matrix)
    Bio.Phylo.write(tree, outPrefix + ".nwk", "newick")

    sys.stderr.write("Running t-SNE\n")
    # generate accessory genome distance representation
    accArray = np.array(accMat)
    accArray_embedded = manifold.TSNE(n_components=2, perplexity=25.0).fit_transform(accArray)

    # print dot file
    sys.stderr.write("Printing t-SNE\n")
    with open(outPrefix + ".dot", 'w') as nFile:
        nFile.write("graph G { ")
        for s, seqLabel in enumerate(seqLabels):
            nFile.write('"' + seqLabel + '"' +
                    '[x='+str(5*float(accArray_embedded[s][0]))+',y='+str(5*float(accArray_embedded[s][1]))+']; ')
        nFile.write("}\n")

    sys.stderr.write("Printing clustering\n")
    with open(outPrefix + ".csv", 'w') as cFile:
        cFile.write("id,Cluster__autocolour\n")
        for label, unique in zip(seqLabels, uniqueSeq):
            if unique in clustering:
                cFile.write(label + ',' + str(clustering[unique]) + '\n')
            else:
                sys.stderr.write("Cannot find " + unique + " in clustering\n")
                sys.exit(1)

    sys.stderr.write("Done\n")

###################
# Plot model fits #
###################

def plot_results(X, Y_, means, covariances, index, outPrefix):
    title = outPrefix + " 2-component BGMM"
    color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold','darkorange'])
    fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = np.linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter([X[Y_ == i, 0]], [X[Y_ == i, 1]], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.title(title)
    plt.savefig(outPrefix + "_twoComponentBGMM.png")
    plt.close()
