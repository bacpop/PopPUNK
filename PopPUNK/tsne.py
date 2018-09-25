#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018 John Lees and Nick Croucher

import os
import sys
import numpy as np
from sklearn import manifold

from .utils import readPickle

def generate_tsne(seqLabels, accMat, perplexity, outPrefix, overwrite, verbosity = 0):
    """Generate t-SNE projection using accessory distances

    Writes a plot of t-SNE clustering of accessory distances (.dot)

    Args:
        seqLabels (list)
            Processed names of sequences being analysed.
        accMat (numpy.array)
            n x n array of accessory distances for n samples.
        perplexity (int)
            Perplexity parameter passed to t-SNE
        outPrefix (str)
            Prefix for all generated output files, which will be placed in
            `outPrefix` subdirectory
        overwrite (bool)
            Overwrite existing output if present

            (default = False)
        verbosity (int)
            Verbosity of t-SNE process (0-3)

            (default = 0)
    """
    # generate accessory genome distance representation
    tsne_filename = outPrefix + "/" + os.path.basename(outPrefix) + "_perplexity" + str(perplexity) + "_accessory_tsne.dot"
    if overwrite or not os.path.isfile(tsne_filename):
        sys.stderr.write("Running t-SNE\n")
        accArray_embedded = manifold.TSNE(n_components=2, perplexity=perplexity, verbose=verbosity).fit_transform(np.array(accMat))

        # print dot file
        with open(tsne_filename, 'w') as nFile:
            nFile.write("graph G { ")
            for s, seqLabel in enumerate(seqLabels):
                nFile.write('"' + seqLabel + '"' +
                            '[x='+str(5*float(accArray_embedded[s][0]))+',y='+str(5*float(accArray_embedded[s][1]))+']; ')
            nFile.write("}\n")
    else:
        sys.stderr.write("t-SNE analysis already exists; add --overwrite to replace\n")


# command line parsing
def get_options():

    import argparse

    parser = argparse.ArgumentParser(description='Run t-SNE projection with a specified perplexity',
                                     prog='poppunk_tsne')

    # input options
    parser.add_argument('--distances', required=True, help='Prefix of input pickle and numpy file of pre-calculated '
                                                           'distances')
    parser.add_argument('--output', required=True, help='Name of output file')
    parser.add_argument('--perplexity', help='Perplexity used to generate t-SNE projection [default = 30]', type=int, default=30)
    parser.add_argument('--verbosity', help='Verbosity level for t-SNE (0-3) [default = 0]', type=int, default=0)

    return parser.parse_args()


# main code
def main():

    # Check input ok
    args = get_options()
    verbosity = 0
    if args.verbosity > 3:
        verbosity = 3
    elif args.verbosity > 0:
        verbosity = args.verbosity

    # load saved distance matrix
    refList, queryList, self, distMat = readPickle(args.distances)

    # process list of file names
    seqLabels = [r.split('/')[-1].split('.')[0] for r in refList]

    # generate accMat
    accMat = np.zeros((len(seqLabels), len(seqLabels)))
    i = 0
    j = 1
    # ref v ref (used for --create-db)
    for row in distMat:
        accMat[i, j] = row[1]
        accMat[j, i] = accMat[i, j]

        if j == len(refList) - 1:
            i += 1
            j = i + 1
        else:
            j += 1

    # generate accessory genome distance representation
    generate_tsne(seqLabels, accMat, args.perplexity, args.output, overwrite = True, verbosity = verbosity)


if __name__ == "__main__":
    main()

    sys.exit(0)
