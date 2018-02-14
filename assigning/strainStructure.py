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
import numpy as np
# import strainStructure package
import strainStructure

#################
# run main code #
#################

if __name__ == '__main__':
    
    # command line parsing
    parser = argparse.ArgumentParser(description='Strain structure analysis software usage:')
    # io options
    ioGroup = parser.add_argument_group('Input/output file names')
    ioGroup.add_argument('-d',type = str, help='Directory containing reference database')
    ioGroup.add_argument('-r', help='File listing reference input assemblies')
    ioGroup.add_argument('-q', help='File listing query input assemblies')
    ioGroup.add_argument('-i', help='Input pickle of pre-calculated distances')
    ioGroup.add_argument('-o', help='Prefix for output files')
    # comparison metrics
    kmerGroup = parser.add_argument_group('Kmer comparison options')
    kmerGroup.add_argument('-m', help='Minimum kmer length (default = 19)')
    kmerGroup.add_argument('-M', help='Maximum kmer length (default = 31)')
    kmerGroup.add_argument('-s', help='Kmer sketch size (default = 10000)')
    # processing options
    procGroup = parser.add_argument_group('Processing options')
    procGroup.add_argument('-b', help='Number of queries to run per batch (default = 1000)')
    procGroup.add_argument('-p', help='Store pickle of calculated distances')
    procGroup.add_argument('-f', help='Keep full reference database', default=False, action='store_true')
    procGroup.add_argument('-u', help='Update reference database with query sequences', default=False, action='store_true')
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
    
    # check on batch size for running queries
    batchSize = 1000
    if args.b is not None:
        batchSize = int(args.b)

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
            strainStructure.createDatabaseDir(args.o)
            assemblyList = strainStructure.readFile(args.r)
            strainStructure.constructDatabase(args.r,kmers,sketchSize,args.o)
            refList,queryList,distMat = strainStructure.queryDatabase(args.r,kmers,args.o,batchSize)
            # store distances in pickle if requested
            if (args.p is not None):
                strainStructure.storePickle(refList,queryList,distMat,args.p)
        # load distances between sequences
        elif mode == "storeDb":
            print("Retrieving new database statistics from input sequences")
            refList,queryList,distMat = strainStructure.readPickle(args.i)
        distanceAssignments,fitWeights,fitMeans,fitcovariances = strainStructure.fit2dMultiGaussian(distMat,args.o)
        genomeNetwork = strainStructure.constructNetwork(refList,queryList,distanceAssignments,fitWeights,fitMeans,fitcovariances)
        strainStructure.printClusters(genomeNetwork,args.o)
        # extract limited references from clique by default
        if args.f is False:
            referenceGenomes = strainStructure.extractReferences(genomeNetwork,args.o)
            strainStructure.constructDatabase(referenceGenomes,kmers,sketchSize,args.o)
            os.system("rm "+referenceGenomes) # tidy up
        strainStructure.printQueryOutput(refList,queryList,distMat,args.o)

    ##########################
    # database querying mode #
    ##########################

    elif mode == "newQuery" or mode == "storeQuery":
        # calculate distances between queries and database
        if mode == "newQuery":
            print("Running query from input files")
            refList,queryList,distMat = strainStructure.queryDatabase(args.q,kmers,args.d,batchSize)
            strainStructure.printQueryOutput(refList,queryList,distMat,args.o)
            # store distances in pickle if requested
            if args.p is not None:
                strainStructure.storePickle(refList,queryList,distMat,args.p)
        # load distances between queries and database
        elif mode == "storeQuery":
            print("Retrieving query statistics from input sequences")
            refList,queryList,distMat = strainStructure.readPickle(args.i)
        queryAssignments,fitWeights,fitMeans,fitcovariances = strainStructure.assignQuery(distMat,args.d)
        querySearchResults,queryNetwork = strainStructure.findQueryLinksToNetwork(refList,queryList,kmers,queryAssignments,fitWeights,fitMeans,fitcovariances,args.o,args.d,batchSize)
        newClusterMembers,existingClusterMatches = strainStructure.assignQueriesToClusters(querySearchResults,queryNetwork,args.d,args.o)
        # update databases if so instructed
        if args.u is True:
            strainStructure.updateDatabase(args.d,newClusterMembers,queryNetwork,args.o,args.f)
            strainStructure.updateClustering(args.d,existingClusterMatches)

    # something's broken
    else:
        sys.exit("Unable to determine running mode")
