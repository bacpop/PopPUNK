#!/usr/bin/env python
# Copyright 2018 John Lees and Nick Croucher

import pickle
import sys
import numpy as np
import argparse
from sklearn import manifold

# command line parsing
def get_options():
    
    parser = argparse.ArgumentParser(description='Run t-SNE projection with a specified perplexity', prog='t-SNE runner')
    
    # input options
    parser.add_argument('--distances', required=True, help='Prefix of input pickle and numpy file of pre-calculated distances (required)')
    parser.add_argument('--perplexity', help='Perplexity used to generate t-SNE projection', type=int)
    parser.add_argument('--output', required=True, help='Name of output file')
    
    return parser.parse_args()

def iterDistRows(refSeqs, querySeqs, self=True):
    if self:
        if refSeqs != querySeqs:
            raise RuntimeError('refSeqs must equal querySeqs for db building (self = true)')
        for i, ref in enumerate(refSeqs):
            for j in range(i + 1, len(refSeqs)):
                yield(refSeqs[j], ref)
    else:
        for query in querySeqs:
            for ref in refSeqs:
                yield(ref, query)

# main code
if __name__ == "__main__":
    
    # Check input ok
    args = get_options()
    
    # load saved distance matrix
    with open(args.distances + ".pkl", 'rb') as pickle_file:
        refList, queryList, self = pickle.load(pickle_file)
    distMat = np.load(args.distances + ".npy")

    # process list of file names
    seqLabels = [r.split('/')[-1].split('.')[0] for r in refList]

    # generate accMat
    accMat = np.zeros((len(seqLabels), len(seqLabels)))
    i = 0
    j = 1
    # ref v ref (used for --create-db)
    for row, (ref, query) in enumerate(iterDistRows(refList, refList, self=True)):
        accMat[i, j] = distMat[row, 1]
        accMat[j, i] = accMat[i, j]
        
        if j == len(refList) - 1:
            i += 1
            j = i + 1
        else:
            j += 1

    # generate accessory genome distance representation
    tsne_filename = args.output + "_perplexity" + str(args.perplexity) + "_accessory_tsne.dot"
    sys.stderr.write("Running t-SNE\n")
    accArray_embedded = manifold.TSNE(n_components=2, perplexity=args.perplexity).fit_transform(np.array(accMat))
    
    # print dot file
    with open(tsne_filename, 'w') as nFile:
        nFile.write("graph G { ")
        for s, seqLabel in enumerate(seqLabels):
            nFile.write('"' + seqLabel + '"' +
                        '[x='+str(5*float(accArray_embedded[s][0]))+',y='+str(5*float(accArray_embedded[s][1]))+']; ')
        nFile.write("}\n")
