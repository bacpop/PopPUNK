'''Plot GMM results'''

import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
# for microreact
from scipy import spatial
from sklearn import manifold
import dendropy

#################################
# Generate files for microreact #
#################################

def outputsForMicroreact(refList, queryList, distMat, clustering, perplexity, outPrefix):

    sys.stderr.write("Writing Microreact output:\n")
    #sys.stderr.write("Getting unique sequences\n")
    uniqueSeq = list(set(refList))
    seqLabels = [r.split('.')[0] for r in uniqueSeq]

    #sys.stderr.write("Converting to matrix\n")
    coreMat = np.zeros((len(uniqueSeq), len(uniqueSeq)))
    accMat = np.zeros((len(uniqueSeq), len(uniqueSeq)))

    for row, (ref, query) in enumerate(zip(refList, queryList)):
        i = uniqueSeq.index(ref)
        j = uniqueSeq.index(query)
        if i != j:
            coreMat[i,j] = distMat[row, 0]
            accMat[i,j] = distMat[row, 1]

    core_dist_file = outPrefix + "/" + outPrefix + "_core_dists.csv"
    np.savetxt(core_dist_file, coreMat, delimiter=",", header = ",".join(seqLabels), comments="")
    acc_dist_file = outPrefix + "/" + outPrefix + "_acc_dists.csv"
    np.savetxt(acc_dist_file, accMat, delimiter=",", header = ",".join(seqLabels), comments="")

    # calculate phylogeny
    sys.stderr.write("Building phylogeny\n")
    pdm = dendropy.PhylogeneticDistanceMatrix.from_csv(src=open(core_dist_file),
                                                       delimiter=",",
                                                       is_first_row_column_names=True,
                                                       is_first_column_row_names=False)
    tree = pdm.nj_tree()
    tree.reroot_at_midpoint(update_bipartitions=False)
    tree.write(path=outPrefix + "/" + outPrefix + "_core_NJ.nwk",
               schema="newick",
               suppress_rooting=True,
               unquoted_underscores=True)

    # generate accessory genome distance representation
    sys.stderr.write("Running t-SNE\n")
    accArray_embedded = manifold.TSNE(n_components=2, perplexity=int(perplexity)).fit_transform(np.array(accMat))

    # print dot file
    #sys.stderr.write("Printing t-SNE\n")
    with open(outPrefix + "/" + outPrefix + "_accessory_tsne.dot", 'w') as nFile:
        nFile.write("graph G { ")
        for s, seqLabel in enumerate(seqLabels):
            nFile.write('"' + seqLabel + '"' +
                    '[x='+str(5*float(accArray_embedded[s][0]))+',y='+str(5*float(accArray_embedded[s][1]))+']; ')
        nFile.write("}\n")

    #sys.stderr.write("Printing clustering\n")
    with open(outPrefix + "/" + outPrefix + "_microreact_clusters.csv", 'w') as cFile:
        cFile.write("id,Cluster__autocolour\n")
        for label, unique in zip(seqLabels, uniqueSeq):
            if unique in clustering:
                cFile.write(label + ',' + str(clustering[unique]) + '\n')
            else:
                sys.stderr.write("Cannot find " + unique + " in clustering\n")
                sys.exit(1)

# Simple scatter plot of distances

def plot_scatter(X, out_prefix, title):
    plt.ioff()
    plt.title(title)
    plt.scatter(X[:,0].flat,X[:,1].flat, s=0.8)
    plt.savefig(out_prefix + ".png")
    plt.close()

###################
# Plot model fits #
###################

def plot_results(X, Y, means, covariances, title, out_prefix):
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
        plt.scatter([X[Y == i, 0]], [X[Y == i, 1]], .8, color=color)

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
