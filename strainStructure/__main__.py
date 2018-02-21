#!/usr/bin/env python
# Copyright 2018 John Lees and Nick Croucher

# universal
import os
import sys
import re
# additional
import numpy as np
import subprocess

# import strainStructure package
from .__init__ import __version__

from .mash import createDatabaseDir
from .mash import storePickle
from .mash import readPickle
from .mash import readFile
from .mash import constructDatabase
from .mash import queryDatabase
from .mash import printQueryOutput
from .mash import assignQueriesToClusters

from .bgmm import fit2dMultiGaussian
from .bgmm import assignQuery

from .network import constructNetwork
from .network import extractReferences
from .network import findQueryLinksToNetwork
from .network import updateDatabase
from .network import updateClustering

#################
# run main code #
#################

# command line parsing
def get_options():

    import argparse

    parser = argparse.ArgumentParser(description='Strain structure analysis through mixture models',
                                     prog='strainStructure')

    modeGroup = parser.add_argument_group('Mode of operation')
    mode = modeGroup.add_mutually_exclusive_group(required=True)
    mode.add_argument('--create-db',
            help='Create pairwise distances database between reference sequences',
            default=False,
            action='store_true')
    mode.add_argument('--fit-model',
            help='Fit a mixture model to a reference database',
            default=False,
            action='store_true')
    mode.add_argument('--create-query-db',
            help='Create distances between query sequences and a reference database',
            default=False,
            action='store_true')
    mode.add_argument('--assign-query',
            help='Assign the cluster of query sequences without re-running the whole mixture model',
            default=False,
            action='store_true')

    # io options
    ioGroup = parser.add_argument_group('Input files')
    ioGroup.add_argument('--ref-db',type = str, help='Location of built reference database')
    ioGroup.add_argument('--r-files', help='File listing reference input assemblies')
    ioGroup.add_argument('--q-files', help='File listing query input assemblies')
    ioGroup.add_argument('--distances', help='Input pickle of pre-calculated distances')

    # processing options
    procGroup = parser.add_argument_group('Output options')
    procGroup.add_argument('--output', required=True, help='Prefix for output files (required)')
    procGroup.add_argument('--save-distances', help='Store pickle of calculated distances', default=False, action='store_true')
    procGroup.add_argument('--full-db', help='Keep full reference database, not just representatives', default=False, action='store_true')
    procGroup.add_argument('--update-db', help='Update reference database with query sequences', default=False, action='store_true')

    # comparison metrics
    kmerGroup = parser.add_argument_group('Kmer comparison options')
    kmerGroup.add_argument('--min-k', default = 9, type=int, help='Minimum kmer length [default = 9]')
    kmerGroup.add_argument('--min-K', default = 29, type=int, help='Maximum kmer length [default = 29]')
    kmerGroup.add_argument('--k-step', default = 4, type=int, help='K-mer step size [default = 4]')
    kmerGroup.add_argument('--sketch-size', default=10000, type=int, help='Kmer sketch size [default = 10000]')

    other = parser.add_argument_group('Other options')
    other.add_argument('--mash', default='mash', help='Location of mash executable')
    other.add_argument('--batch-size', default=1000, help='Number of queries to run per batch [default = 1000]')

    return parser.parse_args()

def main():

    args = get_options()

    # check mash is installed
    p = subprocess.Popen([args.mash + ' --version'], shell=True, stdout=subprocess.PIPE)
    for line in iter(p.stdout.readline, ''):
        if line.rstrip().split(".")[0] < 2:
            sys.stderr.write("Need mash v2 or higher")
            sys.exit(0)

    # identify kmer properties
    minkmer = 9
    maxkmer = 29
    stepSize = 4
    if args.k_step != 2:
        stepSize = args.k_step
    if args.min_k > minkmer:
        minkmer = int(args.min_k)
    if args.max_k < maxkmer:
        maxkmer = int(args.max_k)
    if minkmer >= maxkmer and minkmer >= 9 and maxkmer <= 31:
        sys.stderr.write("Minimum kmer size " + minkmer + " must be smaller than maximum kmer size " +
                         maxkmer + "; range must be between 19 and 31")
        sys.exit(1)

    kmers = np.arange(minkmer,maxkmer+1,stepSize)
    if args.sketch_size < 100 or args.sketch_size > 10**6:
        sys.stderr.write("Sketch size should be between 100 and 10^6")
        sys.exit(1)

    # run according to mode

    # database construction
    if args.create_db:
        sys.stderr.write("Building new database from input sequences\n")
        createDatabaseDir(args.output)
        assemblyList = readFile(args.r_files)
        constructDatabase(args.r_files, kmers, args.sketch_size, args.output)
        refList, queryList, distMat = queryDatabase(args.r_files, kmers, args.output, args.batch_size)
        # store distances in pickle if requested
        if args.save_distances:
            storePickle(refList, queryList, distMat, args.output + ".dists.pkl")

    # model fit and network construction
    elif args.fit_model:
        sys.stderr.write("Fitting model to reference database\n")
        refList, queryList, distMat = readPickle(args.distances)
        distanceAssignments, fitWeights, fitMeans, fitcovariances = fit2dMultiGaussian(distMat, args.output)
        genomeNetwork = constructNetwork(refList, queryList, distanceAssignments, fitWeights, fitMeans, fitcovariances)
        printClusters(genomeNetwork, args.output)
        # extract limited references from clique by default
        if not args.full_db:
            referenceGenomes = extractReferences(genomeNetwork, args.output)
            constructDatabase(referenceGenomes, kmers, args.sketch_size, args.output)
            map(os.remove, referenceGenomes) # tidy up
        printQueryOutput(refList, queryList, distMat, args.output)

    elif args.create_query_db:
        sys.stderr.write("Building new database from input sequences\n")
        refList, queryList, distMat = queryDatabase(args.q_files, kmers, args.ref_db, args.batch_size)
        printQueryOutput(refList, queryList, distMat, args.output)
        # store distances in pickle if requested
        if args.save_distances:
            storePickle(refList, queryList, distMat, args.output + "dists.pkl")

    elif args.assign_query:
        sys.stderr.write("Assigning clusters of query sequences\n")
        refList, queryList, distMat = readPickle(args.distances)
        queryAssignments, fitWeights, fitMeans, fitcovariances = assignQuery(distMat, args.ref_db)
        querySearchResults, queryNetwork = findQueryLinksToNetwork(refList, queryList, kmers,
                queryAssignments, fitWeights, fitMeans, fitcovariances, args.output, args.ref_db, args.batch_size)
        newClusterMembers, existingClusterMatches = assignQueriesToClusters(querySearchResults, queryNetwork, args.ref_db, args.output)
        # update databases if so instructed
        if args.update_dbe:
            updateDatabase(args.ref_db, newClusterMembers, queryNetwork, args.output, args.full_db)
            updateClustering(args.ref_db, existingClusterMatches)

if __name__ == '__main__':
    main()
