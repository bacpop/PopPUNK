#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018-2023 John Lees and Nick Croucher

import os
import sys
import numpy as np
from functools import partial
import random

import poppunk_refine
import pp_sketchlib
from SCE import wtsne
try:
    from SCE import wtsne_gpu_fp32
    gpu_fn_available = True
except ImportError:
    gpu_fn_available = False

from .utils import readPickle

def generate_embedding(seqLabels, accMat, perplexity, outPrefix, overwrite, kNN = 50,
                       maxIter = 10000000, n_threads = 1, use_gpu = False, device_id = 0):
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
        kNN (int)
            Number of neigbours to use with SCE (cannot be > n_samples)
            (default = 50)
        maxIter (int)
            Number of iterations to run
            (default = 1000000)
        n_threads (int)
            Number of CPU threads to use
            (default = 1)
        use_gpu (bool)
            Whether to use GPU libraries
        device_id (int)
            Device ID of GPU to be used
            (default = 0)

    Returns:
        mandrake_filename (str)
            Filename with .dot of embedding
    """
    # generate accessory genome distance representation
    mandrake_filename = outPrefix + "/" + os.path.basename(outPrefix) + "_perplexity" + str(perplexity) + "_accessory_mandrake.dot"
    if os.path.isfile(mandrake_filename) and not overwrite:
        sys.stderr.write("Mandrake analysis already exists; add --overwrite to replace\n")
    else:
        sys.stderr.write("Running mandrake\n")
        kNN = min(kNN, len(seqLabels) - 1)
        I, J, dists = poppunk_refine.get_kNN_distances(accMat, kNN, 1, n_threads)

        # Set up function call with either CPU or GPU
        weights = np.ones((len(seqLabels)))
        random.Random()
        seed = random.randint(0, 2**32)
        gpu_analysis_complete = False
        try:
          if use_gpu and gpu_fn_available:
              sys.stderr.write("Running on GPU\n")
              n_workers = 65536
              maxIter = round(maxIter / n_workers)
              wtsne_call = partial(wtsne_gpu_fp32,
                                  perplexity=perplexity,
                                  maxIter=maxIter,
                                  blockSize=128,
                                  n_workers=n_workers,
                                  nRepuSamp=5,
                                  eta0=1,
                                  bInit=0,
                                  animated=False,
                                  cpu_threads=n_threads,
                                  device_id=device_id,
                                  seed=seed)
              gpu_analysis_complete = True
        except:
          # If installed through conda/mamba mandrake is not GPU-enabled by default
          sys.stderr.write('Mandrake analysis with GPU failed; trying with CPU\n')
        if not gpu_analysis_complete:
            sys.stderr.write("Running on CPU\n")
            maxIter = round(maxIter / n_threads)
            wtsne_call = partial(wtsne,
                                perplexity=perplexity,
                                maxIter=maxIter,
                                nRepuSamp=5,
                                eta0=1,
                                bInit=0,
                                animated=False,
                                n_workers=n_threads,
                                n_threads=n_threads,
                                seed=seed)

        # Run embedding with C++ extension
        embedding_result = wtsne_call(I, J, dists, weights)
        embedding = np.array(embedding_result.get_embedding()).reshape(-1, 2)

        # print dot file
        with open(mandrake_filename, 'w') as nFile:
            nFile.write("graph G { ")
            for s, seqLabel in enumerate(seqLabels):
                nFile.write(f'"{seqLabel}"[x="{str(5*float(embedding[s][0]))}",y="{str(5*float(embedding[s][1]))}"]; ')
            nFile.write("}\n")

    return mandrake_filename

# command line parsing
def get_options():

    import argparse

    parser = argparse.ArgumentParser(description='Run mandrake embedding of accessory distances',
                                     prog='poppunk_mandrake')

    # input options
    parser.add_argument('--distances', required=True, help='Prefix of input pickle and numpy file of pre-calculated '
                                                           'distances')
    parser.add_argument('--output', required=True, help='Name of output file')
    parser.add_argument('--perplexity', help='Perplexity used to generate projection [default = 30]', type=int, default=30)
    parser.add_argument('--knn', help='Number of neighbours used to generate t-SNE projection [default = 50]', type=int, default=50)
    parser.add_argument('--iter', help='Number of iterations [default = 1000000]', type=int, default=10000000)
    parser.add_argument('--cpus', help="Number of CPU threads", type=int, default=1)
    parser.add_argument('--use-gpu', help='Whether to use GPU libraries for t-SNE calculation', default = False, action='store_true')
    parser.add_argument('--device-id', help="Device ID of GPU to use", type=int, default=0)

    return parser.parse_args()


# main code
def main():

    # Check input ok
    args = get_options()

    if not os.path.isdir(args.output):
        try:
            os.makedirs(args.output)
        except OSError:
            sys.stderr.write("Cannot create output directory\n")
            sys.exit(1)

    # load saved distance matrix
    refList, queryList, self, distMat = readPickle(args.distances,
                                                   enforce_self=True)

    # process list of file names
    seqLabels = [r.split('/')[-1].split('.')[0] for r in refList]

    # generate accMat
    accMat = pp_sketchlib.longToSquare(distVec=distMat[:, [1]],
                                       num_threads=args.cpus)

    # generate accessory genome distance representation
    generate_embedding(seqLabels,
                       accMat,
                       args.perplexity,
                       args.output,
                       overwrite=True,
                       kNN=args.knn,
                       maxIter=args.iter,
                       n_threads=args.cpus,
                       use_gpu = args.use_gpu,
                       device_id = args.device_id)

if __name__ == "__main__":
    main()

    sys.exit(0)
