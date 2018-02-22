'''Plot GMM results'''

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

def outputsForMicroreact(refList,queryList,distMat,clustering,outPrefix):
    
    # open output files
    cFile = open(outPrefix+".csv",'w')
    nFile = open(outPrefix+".dot",'w')
    
    print("Getting unique sequences")
    
    # convert distance list to matrix
    n = 0
    uniqueSeq = []
    seqLabels = []
    for r in refList:
        if r not in uniqueSeq:
            uniqueSeq.append(r)
            seqLabels.append(r.split('.')[0])
    numUniqueSeq = len(uniqueSeq)
    coreMat = np.empty((numUniqueSeq,numUniqueSeq,))
    coreMat[:] = 0
    accMat = coreMat.copy()
    
    print("Converting to matrix")
    
    for s in range(0,len(refList)):
        i = uniqueSeq.index(refList[s])
        j = uniqueSeq.index(queryList[s])
        if i != j:
            coreMat[i,j] = distMat[s,0]
            accMat[i,j] = distMat[s,1]

    print("Making triangular")
    
    core_dist_tril = []
    for i in range(numUniqueSeq):
        core_dist_tril.append([])
        core_dist_tril[i] = []
        for j in range(i+1):
            core_dist_tril[i].append(coreMat[i,j])
    core_dist_matrix = _Matrix(uniqueSeq,core_dist_tril)
    new_matrix = DM(names=seqLabels,matrix=core_dist_tril)

    print("Building phylogeny")
    
    # calculate phylogeny
    constructor = DistanceTreeConstructor()
    tree = constructor.nj(new_matrix)
    Bio.Phylo.write(tree,outPrefix+".nwk","newick")
    
    print("Running t-SNE")
    
    # generate accessory genome distance representation
    accArray = np.array(accMat)
    accArray_embedded = manifold.TSNE(n_components=2, perplexity=25.0).fit_transform(accArray)
    
    print("Printing t-SNE")
    
    # print dot file
    print("graph G { ",end='',file=nFile)
    for s in range(len(seqLabels)):
        print('"'+seqLabels[s]+'"'+'[x='+str(5*float(accArray_embedded[s][0]))+',y='+str(5*float(accArray_embedded[s][1]))+']; ',end='',file=nFile)
    print("}",file=nFile)

    print("Printing clustering")
    
    # print clustering
    print("id,Cluster__autocolour",file=cFile)
    for s in range(len(uniqueSeq)):
        if uniqueSeq[s] in clustering:
            print(seqLabels[s]+','+str(clustering[uniqueSeq[s]]),file=cFile)
        else:
            sys.exit("Cannot find "+uniqueSeq[s]+" in clustering")

    print("Done")
    
    # finalise
    nFile.close()
    cFile.close()
    
    return None

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
