#!/usr/bin/env python

# universal
import os
import sys
import argparse
import re
# additional
import pickle
import gzip, io
import random
import networkx as nx
# package specific
from bgmm import *
from mash import *
from network import *
from plot import *

#################
# run main code #
#################

if __name__ == '__main__':
    
    # command line parsing
    parser = argparse.ArgumentParser(description='Strain structure analysis software usage:')
    parser.add_argument('-d',type = str, help='Directory containing reference database')
    parser.add_argument('-q', help='File listing query input assemblies')
    parser.add_argument('-r', help='File listing reference input assemblies')
    parser.add_argument('-i', help='Input pickle of pre-calculated distances')
    parser.add_argument('-m', help='Minimum kmer length (default = 19)')
    parser.add_argument('-M', help='Maximum kmer length (default = 31)')
    parser.add_argument('-s', help='Kmer sketch size (default = 10000)')
    parser.add_argument('-o', help='Prefix for output files')
    parser.add_argument('-p', help='Store pickle of calculated distances')
    parser.add_argument('-f', help='Keep full reference database', default=False, action='store_true')
    parser.add_argument('-u', help='Update reference database with query sequences', default=False, action='store_true')
    args = parser.parse_args()
    
    # check mash is installed
    try:
        os.system("mash > /dev/null 2> /dev/null")
    except:
        sys.exit("mash not installed on your path")
    
    # identify kmer properties
    minkmer = 19
    maxkmer = 31
    if args.m is not None and int(args.m) > minkmer:
        minkmer = int(args.m)
    if args.M is not None and int(args.M) < maxkmer:
        maxkmer = int(args.M)
    if minkmer >= maxkmer and minkmer >= 19 and maxkmer <= 31:
        sys.exit("Minimum kmer size "+minkmer+" must be smaller than maximum kmer size "+maxkmer+"; range must be between 19 and 31")
    kmers = np.arange(minkmer,maxkmer+1,2)
    sketchSize = 10000
    if args.s is not None:
        sketchSize = arg.s
    
    # check on output prefix
    if args.o is None:
        sys.exit("Please provide an output file prefix")

    # universal data structures
    refList = []
    queryList = []
    distMat = np.matrix
    coreDist = {}
    accessoryDist = {}

    # determine mode for running
    mode = ""
    if args.d is None:
        if args.i is None:
            if args.r is not None:
                mode = "newDb"
            else:
                sys.exit("Creation of new database "+args.d+" requires reference sequence list")
        else:
            mode = "storeDb"
    elif args.r is None:
        if args.i is None:
            if args.q is not None:
                mode = "newQuery"
            else:
                sys.exit("Cannot query database "+args.d+" without a list of query sequences")
        else:
            mode = "storeQuery"
    else:
        sys.exit("Cannot accept a list of references "+args.r+" and a database "+args.d+" in the same analysis")
        
    # run according to mode
    
    ##############################
    # database construction mode #
    ##############################
    
    if mode == "newDb" or mode == "storeDb":
        # calculate distances between sequences
        if mode  == "newDb":
            print("Building new database from input sequences")
            createDatabaseDir(args.o)
            assemblyList = readFile(args.r)
            #            constructDatabase(assemblyList,kmers,sketchSize,args.o)
            constructDatabase(args.r,kmers,sketchSize,args.o)
            refList,queryList,distMat = queryDatabase(args.r,kmers,args.o)
            # store distances in pickle if requested
            if (args.p is not None):
                storePickle(refList,queryList,distMat,args.p)
        # load distances between sequences
        elif mode == "storeDb":
            print("Retrieving new database statistics from input sequences")
            refList,queryList,distMat = readPickle(args.i)
        distanceAssignments,fitWeights,fitMeans,fitcovariances = fit2dMultiGaussian(distMat,args.o)
        genomeNetwork = constructNetwork(refList,queryList,distanceAssignments,fitWeights,fitMeans,fitcovariances)
        printClusters(genomeNetwork,args.o)
        # extract limited references from clique by default
        if args.f is False:
            referenceGenomes = extractReferences(genomeNetwork,args.o)
            constructDatabase(referenceGenomes,kmers,sketchSize,args.o)
            os.system("rm "+referenceGenomes) # tidy up
        printQueryOutput(refList,queryList,distMat,args.o)

    ##########################
    # database querying mode #
    ##########################

    elif mode == "newQuery" or mode == "storeQuery":
        # calculate distances between queries and database
        if mode == "newQuery":
            print("Running query from input files")
            refList,queryList,distMat = queryDatabase(args.q,kmers,args.d)
            printQueryOutput(refList,queryList,distMat,args.o)
            # store distances in pickle if requested
            if args.p is not None:
                storePickle(refList,queryList,distMat,args.p)
        # load distances between queries and database
        elif mode == "storeQuery":
            print("Retrieving query statistics from input sequences")
            refList,queryList,distMat = readPickle(args.i)
            queryAssignments,fitWeights,fitMeans,fitcovariances = assignQuery(distMat,args.d)
            querySearchResults,queryNetwork = findQueryLinksToNetwork(refList,queryList,kmers,queryAssignments,fitWeights,fitMeans,fitcovariances,args.o)
            newClusterMembers,existingClusterMatches = assignQueriesToClusters(querySearchResults,queryNetwork,args.d,args.o)
            # update databases if so instructed
            if args.u is True:
                updateDatabase(args.d,newClusterMembers,queryNetwork,args.o,args.f)
                updateClustering(args.d,existingClusterMatches)

    # something's broken
    else:
        sys.exit("Unable to determine running mode")
