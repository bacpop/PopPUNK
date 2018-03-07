#!/usr/bin/env python
# Copyright 2018 John Lees and Nick Croucher

# universal
import os
import sys
# additional
import numpy as np
import subprocess

# import strainStructure package
from .__init__ import __version__

from .mash import createDatabaseDir
from .mash import storePickle
from .mash import readPickle
from .mash import readAssemblyList
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
from .network import printClusters

from .plot import outputsForMicroreact

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

    # input options
    iGroup = parser.add_argument_group('Input files')
    iGroup.add_argument('--ref-db',type = str, help='Location of built reference database')
    iGroup.add_argument('--r-files', help='File listing reference input assemblies')
    iGroup.add_argument('--q-files', help='File listing query input assemblies')
    iGroup.add_argument('--distances', help='Input pickle of pre-calculated distances')

    # output options
    oGroup = parser.add_argument_group('Output options')
    oGroup.add_argument('--output', required=True, help='Prefix for output files (required)')
    oGroup.add_argument('--save-distances', help='Store pickle of calculated distances for query sequences', default=False, action='store_true')
    oGroup.add_argument('--microreact', help='Generate output files for microreact', default=False, action='store_true')
    oGroup.add_argument('--full-db', help='Keep full reference database, not just representatives', default=False, action='store_true')
    oGroup.add_argument('--update-db', help='Update reference database with query sequences', default=False, action='store_true')

    # comparison metrics
    kmerGroup = parser.add_argument_group('Kmer comparison options')
    kmerGroup.add_argument('--min-k', default = 11, type=int, help='Minimum kmer length [default = 11]')
    kmerGroup.add_argument('--max-k', default = 29, type=int, help='Maximum kmer length [default = 29]')
    kmerGroup.add_argument('--k-step', default = 4, type=int, help='K-mer step size [default = 4]')
    kmerGroup.add_argument('--sketch-size', default=10000, type=int, help='Kmer sketch size [default = 10000]')

    modelGroup = parser.add_argument_group('Mixture model options')
    modelGroup.add_argument('--priors', help='File specifying model priors. See documentation for help', default=None)
    modelGroup.add_argument('--dpgmm', help='Use EM rather than ADVI to fit the mixture model', default=False, action='store_true')
    modelGroup.add_argument('--K', help='Maximum number of mixture components (--dpgmm only) [default = 2]', type=int, default=2)

    other = parser.add_argument_group('Other options')
    other.add_argument('--mash', default='mash', help='Location of mash executable')
    other.add_argument('--batch-size', default=1000, help='Number of queries to run per batch [default = 1000]')

    other.add_argument('--version', action='version',
                       version='%(prog)s '+__version__)

    return parser.parse_args()

def main():

    args = get_options()

    # check mash is installed
    p = subprocess.Popen([args.mash + ' --version'], shell=True, stdout=subprocess.PIPE)
    version = 0
    for line in iter(p.stdout.readline, ''):
        if line != '':
            version = line.rstrip().decode().split(".")[0]
            break
    if not version.isdigit() or int(version) < 2:
        sys.stderr.write("Need mash v2 or higher\n")
        sys.exit(0)

    # identify kmer properties
    minkmer = 9
    maxkmer = 29
    stepSize = 4
    if args.k_step is not None and args.k_step != 2:
        stepSize = args.k_step
    if args.min_k is not None and args.min_k > minkmer:
        minkmer = int(args.min_k)
    if args.max_k is not None and args.max_k < maxkmer:
        maxkmer = int(args.max_k)
    if minkmer >= maxkmer or minkmer < 9 or maxkmer > 31:
        sys.stderr.write("Minimum kmer size " + minkmer + " must be smaller than maximum kmer size " +
                         maxkmer + "; range must be between 9 and 31\n")
        sys.exit(1)

    kmers = np.arange(minkmer,maxkmer+1,stepSize)

    # define sketch sizes, store in hash in case on day
    # different kmers get different hash sizes
    sketch_sizes = {}
    if args.sketch_size >= 100 and args.sketch_size <= 10**6:
        for k in kmers:
            sketch_sizes[k] = args.sketch_size
    else:
        sys.stderr.write("Sketch size should be between 100 and 10^6\n")
        sys.exit(1)

    # run according to mode
    sys.stderr.write("PopPUNK (POPulation Partitioning Using Nucleotide Kmers)\n")

    # database construction
    if args.create_db:
        sys.stderr.write("Mode: Building new database from input sequences\n")
        if args.r_files is not None:
            createDatabaseDir(args.output)
            assemblyList = readAssemblyList(args.r_files)
            constructDatabase(args.r_files, kmers, sketch_sizes, args.output)
            refList, queryList, distMat = queryDatabase(args.r_files, kmers, args.output, args.batch_size)
            storePickle(refList, queryList, distMat, args.output + "/" + args.output + ".dists.pkl")
        else:
            sys.stderr.write("Need to provide a list of reference files with --r-files; need to use --save-distances to fit model subsequently")
            sys.exit(1)

    # model fit and network construction
    elif args.fit_model:
        if args.distances is not None:
            sys.stderr.write("Mode: Fitting model to reference database\n\n")
            refList, queryList, distMat = readPickle(args.distances)
            distanceAssignments, fitWeights, fitMeans, fitcovariances = fit2dMultiGaussian(distMat, args.output, args.priors, args.dpgmm, args.K)
            genomeNetwork = constructNetwork(refList, queryList, distanceAssignments, fitWeights, fitMeans, fitcovariances)
            isolateClustering = printClusters(genomeNetwork, args.output)
            # generate outputs for microreact if asked
            if args.microreact:
                outputsForMicroreact(refList,queryList,distMat,isolateClustering,args.output)
            # extract limited references from clique by default
            if not args.full_db:
                referenceGenomes = extractReferences(genomeNetwork, args.output)
                constructDatabase(referenceGenomes, kmers, sketch_sizes, args.output)
                map(os.remove, referenceGenomes) # tidy up
            printQueryOutput(refList, queryList, distMat, args.output)
        else:
            sys.stderr.write("Need to provide an input set of distances with --distances\n\n")
            sys.exit(1)

    elif args.create_query_db:
        if args.ref_db is not None and args.q_files is not None:
            sys.stderr.write("Mode: Building new database from input sequences\n")
            refList, queryList, distMat = queryDatabase(args.q_files, kmers, args.ref_db, args.batch_size)
            printQueryOutput(refList, queryList, distMat, args.output)
            # store distances in pickle if requested
            if args.save_distances:
                storePickle(refList, queryList, distMat, args.output + ".dists.pkl")
        else:
            sys.stderr.write("Need to provide both a reference database with --ref-db and query list with --q-files; use --save-distances to subsequently assign queries to clusters\n")
            sys.exit(1)

    elif args.assign_query:
        if args.ref_db is not None and args.distances is not None:
            sys.stderr.write("Mode: Assigning clusters of query sequences\n\n")
            refList, queryList, distMat = readPickle(args.distances)
            queryAssignments, fitWeights, fitMeans, fitcovariances = assignQuery(distMat, args.ref_db)
            querySearchResults, queryNetwork = findQueryLinksToNetwork(refList, queryList, kmers,
                    queryAssignments, fitWeights, fitMeans, fitcovariances, args.output, args.ref_db, args.batch_size)
            newClusterMembers, existingClusterMatches = assignQueriesToClusters(querySearchResults, queryNetwork, args.ref_db, args.output)
            # update databases if so instructed
            if args.update_db:
                updateDatabase(args.ref_db, newClusterMembers, queryNetwork, args.output, args.full_db)
                updateClustering(args.ref_db, existingClusterMatches)
        else:
            sys.stderr.write("Need to provide both a reference database with --ref-db and calculated distances with --distances\n\n")
            sys.exit(1)

    sys.stderr.write("\nDone\n")

if __name__ == '__main__':
    main()

    sys.exit(0)
