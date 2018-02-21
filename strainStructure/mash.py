'''Mash functions for database construction'''

# universal
import os
import sys
# additional
import collections
import pickle
import numpy as np
from scipy.stats import linregress

#####################
# Get database name #
#####################

def getDatabaseName(prefix, k):
    return prefix + "/" + prefix + "." + k + ".msh"

#############################
# create database directory #
#############################

def createDatabaseDir(outPrefix):
    outputDir = os.getcwd() + "/" + outPrefix
    # check for writing
    if not os.path.isdir(outputDir):
        try:
            os.makedirs(outputDir)
        except:
            sys.stderr.write("Cannot create output directory")
            sys.exit(1)

#####################################
# Store distance matrix in a pickle #
#####################################

def storePickle(rlist, qlist, X, pklName):
    with open(pklName, 'wb') as pickle_file:
        pickle.dump([rlist, qlist, X], pickle_file)

####################################
# Load distance matrix from pickle #
####################################

def readPickle(pklName):
    with open(pklName, 'rb') as pickle_file:
        rlist, qlist, X = pickle.load(pickle_file)
    return rlist, qlist, X

#########################
# Print output of query #  # needs work still
#########################

def assignQueriesToClusters(links, G, databaseName, outPrefix):

    # open output file
    outFileName = outPrefix + "_clusterAssignment.out"
    with open(outFileName, 'w') as oFile:
        oFile.write("Query,Cluster\n")

        # parse existing clusters into existingCluster dict
        # also record the current maximum cluster number for adding new clusters
        maxCluster = 0;
        existingCluster = {}
        dbClusterFileName = "./" + databaseName + "/" + databaseName + "_clusters.csv"
        with open(dbClusterFileName, 'r') as cFile:
            for line in cFile:
                clusterVals = line.rstrip().split(",")
                if clusterVals[0] != "Taxon":
                    # account for decimal clusters that have been merged
                    intCluster = int(clusterVals[1].split('.')[0])
                    #            existingCluster[clusterVals[0]] = clusterVals[1]
                    existingCluster[clusterVals[0]] = intCluster
                    if intCluster > maxCluster:
                        maxCluster = intCluster

        # calculate query clusters here
        queryCluster = {}
        queriesInCluster = {}
        clusters = sorted(nx.connected_components(G), key=len, reverse=True)
        cl_id = maxCluster + 1
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
            oFile.write(query + ",")
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
                newCluster = None

                # check if any of the other queries in the same query cluster
                # match an existing cluster - if so, assign to existing cluster
                # as a transitive property
                if query in queryCluster:
                    for similarQuery in queriesInCluster[queryCluster[query]]:
                        if len(existingHits[similarQuery].keys()) > 0:
                            if newCluster is None:
                                newCluster = str(';'.join(str(v) for v in existingHits[similarQuery].keys()))
                            else:
                                newCluster = newCluster + ';' + str(';'.join(str(v) for v in existingHits[similarQuery].keys()))

                # if no similar queries match a reference sequence
                if newCluster is None:
                    if query in queryCluster:
                        # matches a query that has already been assigned a new cluster
                        if queryCluster[query] in translateClusters:
                            newCluster = translateClusters[queryCluster[query]]
                        # otherwise define a new cluster, incrementing from the previous maximum number
                        else:
                            newCluster = queryCluster[query]
                            translateClusters[queryCluster[query]] = queryCluster[query]
                        additionalClusters[query] = newCluster
                    else:
                        maxCluster += 1
                        newCluster = maxCluster

                oFile.write(str(newCluster) + '\n')

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
                oFile.write(hitString + "\n")

    # returns:
    # existingHits: dict of dicts listing hits to references already in the database
    # additionalClusters: dict of query assignments to new clusters, if they do not match existing references
    return additionalClusters, existingHits

###########################
# read assembly file list #
###########################

def readAssemblyList(fn):
    assemblyList = []
    with open(fn, 'r') as iFile:
        for line in iFile:
            assemblyList.append(line.rstrip())
    return assemblyList

##########################################
# Get sketch size from existing database #
##########################################

def getSketchSize(dbPrefix, klist):

    # identify sketch lengths used to generate databases
    sketch = 0
    oldSketch = 0
    dbname = "./" + dbPrefix + "/" + dbPrefix + "." + str(k) + ".msh"
    for k in klist:
        try:
            mash_info = subprocess.run("mash info -t " + dbname, shell=True, check=True, stdout=subprocess.PIPE)
            for line in mash_info.readlines():
                if (line.startswith("#") is False):
                    sketchValues = line.split("\t")
                    if len(sketchValues[0]) > 0:
                        if oldSketch == 0:
                            oldSketch = int(sketchValues[0])
                        else:
                            oldSketch = sketch
                        sketch = int(sketchValues[0])
                        if (sketch != oldSketch):
                            sys.stderr.write("Problem with database; not all files have same sketch size")
                            sys.exit(1)
                        else:
                            break
        except:
            sys.stderr.write("Could not get into about " + dbname)
            sys.exit(1)

    return sketch

########################
# construct a database #
########################

def constructDatabase(assemblyList, klist, sketch, oPrefix):
    # create kmer databases
    for k in klist:
        sys.stderr.write("Creating mash database for k = " + str(k) + "\n")
        dbname = "./" + oPrefix + "/" + oPrefix + "." + str(k)
        try:
            mash_cmd = "mash sketch -s " + str(sketch) + " -o " + dbname + " -k " + str(k) + " -l " + assemblyList + " 2> /dev/null"
            subprocess.run(mash_cmd, shell=True, check=True)
        except:
            sys.stderr.write("Could not create mash database " + dbname)
            sys.exit(1)

# split input queries into chunks
def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

####################
# query a database #
####################

def queryDatabase(qFile,klist,dbPrefix,batchSize):

    # initialise dictionary
    nested_dict = lambda: collections.defaultdict(nested_dict)
    raw = nested_dict()
    queryList = readFile(qFile)

    # batch query files to save on memory
    qFiles = []
    fileNumber = 1
    counter = 0

    for fileNumber, qFileChunk in enumerate(chunks(queryList, batchSize)):
        tmpOutFn = "tmp." + dbPrefix + "." + str(fileNumber)
        qFiles.append(tmpOutFn)
        with open(tmpOutFn, 'w') as tmpFile:
            for qF in qFileChunk:
                tmpFile.write(qf + '\n')

    # initialise data structures
    core = nested_dict()
    accessory = nested_dict()
    dbSketch = {}
    querySeqs = []
    refSeqs = []
    coreVals = []
    accVals = []

    # calculate sketch size
    sketchSize = getSketchSize(dbPrefix, klist)

    # search each query file
    for qF in qFiles:
        sys.stderr.write("Processing file " + qF + "\n")

        # iterate through kmer lengths
        for k in klist:
            # run mash distance query based on current file
            dbname = "./" + dbPrefix + "/" + dbPrefix + "." + str(k) + ".msh"
            mash_cmd = "mash dist -l " + dbname + " " + qF + " 2> " + dbPrefix + ".err.log"
            sys.stderr.write(mash_cmd)

            try:
                rawOutput = subprocess.run(mash_cmd, shell=True, stdout=subprocess.PIPE, check=True)

                for line in rawOutput.stdout.readlines():
                    mashVals = line.rstrip().split()
                    if (len(mashVals) > 2):
                        mashMatch = mashVals[len(mashVals)-1].split('/')
                        if (k == klist[0]):
                            raw[mashVals[1]][mashVals[0]][str(k)] = mashMatch[0]
            except:
                sys.stderr.write("mash dist command failed")
                sys.exit(1)

        # run pairwise analyses across kmer lengths
        for query in raw:
            for ref in raw[query]:
                pairwise = []
                for k in klist:
                    pairwise.append(np.log(float(raw[query][ref][str(k)])/float(sketchSize)))
                # curve fit pr = (1-a)(1-c)^k
                # log pr = log(1-a) + k*log(1-c)
                gradient, intercept, r_value, p_value, std_err = linregress(klist, pairwise)
                accessory[query][ref] = 1 - exp(intercept)
                core[query][ref] = 1 - exp(gradient)
                # store output
                querySeqs.append(query)
                refSeqs.append(ref)
                coreVals.append(core[query][ref])
                accVals.append(accessory[query][ref])

            # clear for memory purposes
            raw[query].clear()

    distMat = np.transpose(np.matrix((coreVals, accVals)))
    return(refSeqs, querySeqs, distMat)

##############################
# write query output to file #
##############################

def printQueryOutput(rlist, qlist, X, outPrefix):

    # open output file
    outFileName = outPrefix + ".search.out"
    with open(outFileName, 'w') as oFile:
        oFile.write("\t".join(['Query', 'Reference', 'Core', 'Accessory']) + "\n")
        # add results
        for i, (query, ref) in enumerate(zip(qlist, rlist)):
            oFile.write("\t".join([query, ref, str(X[i,0]), str(X[i,1])]) + "\n")

