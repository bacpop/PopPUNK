'''Mash functions for database construction'''

# universal
import os
import sys
import argparse
import re
# additional
import pickle
import networkx as nx
import numpy as np
import pymc3 as pm
from scipy import stats
from scipy import linalg
# import strainStructure package
import strainStructure

#####################
# Get database name #
#####################

def getDatabaseName(prefix,k):
    fn = prefix+"/"+prefix+"."+k+".msh"
    return fn

#############################
# create database directory #
#############################

def createDatabaseDir(outPrefix):
    outputDir = os.getcwd()+"/"+outPrefix
    # check for writing
    if not os.path.isdir(outputDir):
        try:
            os.makedirs(outputDir)
        except:
            sys.exit("Cannot create output directory")

    return None

#####################################
# Store distance matrix in a pickle #
#####################################

def storePickle(rlist,qlist,X,pklName):
    with open(pklName,'wb') as f:
        pickle.dump([rlist,qlist,X],f)

####################################
# Load distance matrix from pickle #
####################################

def readPickle(pklName):
    with open(pklName,'rb') as f:
        rlist,qlist,X = pickle.load(f)
    return rlist,qlist,X

#########################
# Print output of query #  # needs work still
#########################

def assignQueriesToClusters(links,G,databaseName,outPrefix):

    # open output file
    outFileName = outPrefix+"_clusterAssignment.out"
    oFile = ""
    try:
        oFile = open(outFileName,'w')
    except:
        sys.exit("Cannot write to "+outFileName)
    print("Query,Cluster",file=oFile)

    # parse existing clusters into existingCluster dict
    # also record the current maximum cluster number for adding new clusters
    maxCluster = 0;
    existingCluster = {}
    dbClusterFileName = "./"+databaseName+"/"+databaseName+"_clusters.csv"
    cFile = ""
    try:
        cFile = open(dbClusterFileName,'r')
    except:
        sys.exit("Cannot read database clusters file "+dbClusterFileName)
    for line in cFile.readlines():
        clusterVals = line.strip().split(",")
        if clusterVals[0] != "Taxon":
            # account for decimal clusters that have been merged
            intCluster = int(clusterVals[1].split('.')[0])
            #            existingCluster[clusterVals[0]] = clusterVals[1]
            existingCluster[clusterVals[0]] = intCluster
            if intCluster > maxCluster:
                maxCluster = intCluster
    cFile.close()

    # calculate query clusters here
    queryCluster = {}
    queriesInCluster = {}
    clusters = sorted(nx.connected_components(G), key=len, reverse=True)
    cl_id = maxCluster+1
    for cl_id, cluster in enumerate(clusters):
        queriesInCluster[cl_id] = []
        for cluster_member in cluster:
            queryCluster[cluster_member] = cl_id
            queriesInCluster[cl_id].append(cluster_member)
        if cl_id > maxCluster:
            maxCluster = cl_id

    # iterate through links, which comprise both query-ref links
    # and query-query links
    translateClusters = {}
    additionalClusters = {}
    existingHits = {}
    for query in links:
        existingHits[query] = {}
        print(query+',',end='',file=oFile)
        newHits = []
        
        # populate existingHits dict with links to already-clustered reference sequences
        for link in links[query]:
            if link in existingCluster:
                existingHits[query][existingCluster[link]] = 1
    
        # if no links to existing clusters found in the existingHits dict
        # then look at whether there are links to other queries, and whether
        # they have already been clustered
        if len(existingHits[query].keys()) == 0:
            
            # initialise the new cluster
            newCluster = "NA"
            
            # check if any of the other queries in the same query cluster
            # match an existing cluster - if so, assign to existing cluster
            # as a transitive property
            if query in queryCluster:
                for similarQuery in queriesInCluster[queryCluster[query]]:
                    if len(existingHits[similarQuery].keys()) > 0:
                        if newCluster == "NA":
                            newCluster = str(';'.join(str(v) for v in existingHits[similarQuery].keys()))
                        else:
                            newCluster = newCluster+';'+str(';'.join(str(v) for v in existingHits[similarQuery].keys()))
        
            # check if a link to an existing query was found
            if newCluster != "NA":
                print(str(newCluster),file=oFile)
            
            # if no similar queries match a reference sequence
            else:
                if query in queryCluster:
                    # matches a query that has already been assigned a new cluster
                    if queryCluster[query] in translateClusters:
                        newCluster = translateClusters[queryCluster[query]]
                    # otherwise define a new cluster, incrementing from the previous maximum number
                    else:
                        newCluster = queryCluster[query]
                        translateClusters[queryCluster[query]] = queryCluster[query]
                    print(str(newCluster),file=oFile)
                    additionalClusters[query] = newCluster
                else:
                    maxCluster = maxCluster+1
                    print(maxCluster,file=oFile)

        # if multiple links to existing clusters found in the existingHits dict
        # then the clusters will have to be merged if the database is updated
        # for the moment, they can be recorded as just matching two clusters
        else:
            # matching multiple existing clusters that will need to be merged
            if len(existingHits[query].keys()) > 1:
                hitString = str(';'.join(str(v) for v in existingHits[query].keys()))
            # matching only one cluster
            else:
                hitString = str(list(existingHits[query])[0])
            print(hitString,file=oFile)

    oFile.close()
        
    # returns:
    # existingHits: dict of dicts listing hits to references already in the database
    # additionalClusters: dict of query assignments to new clusters, if they do not match existing references
    return additionalClusters,existingHits

###########################
# read assembly file list #
###########################

def readFile (fn):
    assemblyList = []
    try:
        ifile = open(fn,'r')
        assemblyList = [line.rstrip('\n') for line in ifile]
        ifile.close()
        return assemblyList
    except:
        sys.exit("Unable to read input file "+fn)
    return None

########################
# construct a database #
########################

def constructDatabase(assemblyList,klist,sketch,oPrefix):
    
    # create kmer databases
    for k in klist:
        dbname = "./"+oPrefix+"/"+oPrefix+"."+str(k)
        try:
            os.system("mash sketch -s "+str(sketch)+" -o "+dbname+" -k "+str(k)+" -l "+assemblyList+" 2> /dev/null")
        except:
            sys.exit("Cannot create database "+dbname)
    
    # finish
    return None

####################
# query a database #
####################

def queryDatabase(qFile,klist,dbPrefix,batchSize):
    
    # initialise dictionary
    raw = {}
    queryList = readFile(qFile)
    for query in queryList:
        raw[query] = {}

    # batch query files to save on memory
    qFiles = []
    fileNumber = 1
    counter = 0
    tmpOutFn = "tmp."+dbPrefix+"."+str(fileNumber)
    tmpOut = open(tmpOutFn,'w')
    for qF in raw:
        if counter == batchSize:
            tmpOut.close()
            qFiles.append(tmpOutFn)
            fileNumber = fileNumber+1
            tmpOutFn = "tmp."+dbPrefix+"."+str(fileNumber)
            tmpOut = open(tmpOutFn,'w')
            counter = 0
        print(qF,file=tmpOut)
        counter = counter+1
    if not tmpOut.closed:
        tmpOut.close()
        qFiles.append(tmpOutFn)

    # initialise data structures
    core = {}
    accessory = {}
    dbSketch = {}
    querySeqs = []
    refSeqs = []
    coreVals = []
    accVals = []

    # calculate sketch size
    sketchSize = strainStructure.getSketchSize(dbPrefix,klist)

    # search each query file
    for qF in qFiles:
        
        print("Processing file "+qF)
        
        # iterate through kmer lengths
        for k in klist:

            # run mash distance query based on current file
            dbname = "./"+dbPrefix+"/"+dbPrefix+"."+str(k)+".msh"
            print("mash dist -l "+dbname+" "+qF+" 2> "+dbPrefix+".err.log")
            rawOutput = os.popen("mash dist "+dbname+" "+qF+" 2> "+dbPrefix+".err.log").read()
            for line in rawOutput.split("\n"):
                mashVals = line.strip().split()
                if (len(mashVals) > 2):
                    mashMatch = mashVals[len(mashVals)-1].split('/')
                    if (k == klist[0]):
                        raw[mashVals[1]][mashVals[0]] = {}
                    raw[mashVals[1]][mashVals[0]][str(k)] = mashMatch[0]
            
        # run pairwise analyses across kmer lengths
        for query in raw:
            core[query] = {}
            accessory[query] = {}
            for ref in raw[query]:
                pairwise = []
                for k in klist:
                    pairwise.append(int(raw[query][ref][str(k)]))
                gradient,intercept,r_value,p_value,std_err = stats.linregress(klist,pairwise)
                accessory[query][ref] = 1.0-float(intercept)/float(sketchSize)
                core[query][ref] = -1*float(gradient)/float(sketchSize)
                # store output
                querySeqs.append(query)
                refSeqs.append(ref)
                coreVals.append(core[query][ref])
                accVals.append(accessory[query][ref])
                
            # clear for memory purposes
            raw[query].clear()
        
    distMat = np.transpose(np.matrix((coreVals,accVals)))
    return(refSeqs,querySeqs,distMat)

##############################
# write query output to file #
##############################

def printQueryOutput(rlist,qlist,X,outPrefix):
    
    # open output file
    outFileName = outPrefix+".search.out"
    try:
        oFile = open(outFileName,'w')
    except:
        sys.exit("Cannot write to output file "+outPrefix+".search.out")
    
    # print header
    print("Query\tReference\tCore\tAccessory",file=oFile)
    
    # add results
    for i in range(0,len(rlist)):
        print(qlist[i]+"\t"+rlist[i]+"\t"+str(X[i,0])+"\t"+str(X[i,1]),file=oFile)

    oFile.close()

    return None



