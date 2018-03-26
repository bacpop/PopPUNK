'''Mash functions for database construction'''

# universal
import os
import sys
import subprocess
# additional
import collections
import pickle
from multiprocessing import Pool, Lock
from functools import partial
import numpy as np
import networkx as nx
from scipy import optimize

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
            sys.stderr.write("Cannot create output directory\n")
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

def getSketchSize(dbPrefix, klist, mash_exec = 'mash'):

    # identify sketch lengths used to generate databases
    sketchdb = {}
    sketch = 0
    oldSketch = 0

    # iterate over kmer lengths
    for k in klist:
        dbname = "./" + dbPrefix + "/" + dbPrefix + "." + str(k) + ".msh"
        try:
            mash_info = subprocess.Popen(mash_exec + " info -t " + dbname, shell=True, stdout=subprocess.PIPE)
            for line in iter(mash_info.stdout.readline, ''):
                line = line.rstrip().decode()
                if (line.startswith("#") is False):
                    sketchValues = line.split("\t")
                    if len(sketchValues[0]) > 0:
                        if oldSketch == 0:
                            oldSketch = int(sketchValues[1])
                        else:
                            oldSketch = sketch
                        sketch = int(sketchValues[1])
                        if (sketch == oldSketch):
                            sketchdb[k] = sketch
                        else:
                            sys.stderr.write("Problem with database; sketch size for kmer length "+str(k)+" is "+str(oldSketch)+", but smaller kmers have sketch sizes of "+str(sketch)+"\n")
                            sys.exit(1)

                        break

            # Make sure process executed correctly
            mash_info.wait()
            if mash_info.returncode != 0:
                raise RuntimeError('mash info failed')
        except subprocess.CalledProcessError as e:
            sys.stderr.write("Could not get info about " + dbname + "; command "+mash_exec + " info -t " + dbname+" returned "+str(mash_info.returncode)+": "+e.message+"\n")
            sys.exit(1)

    return sketchdb

# Return an array with the sequences in the passed mash database
def getSeqsInDb(mashSketch, mash_exec = 'mash'):

    seqs = []

    mash_cmd = mash_exec + " info -t " + mashSketch
    try:
        mash_info = subprocess.Popen(mash_cmd, shell=True, stdout=subprocess.PIPE)
        for line in iter(mash_info.stdout.readline, ''):
            line = line.rstrip().decode()
            if (line.startswith("#") is False):
                seqs.append(line.split("\t")[2])

            # Make sure process executed correctly
            mash_info.wait()
            if mash_info.returncode != 0:
                raise RuntimeError('mash info failed')
        except subprocess.CalledProcessError as e:
            sys.stderr.write("Could not get info about " + dbname + "; command " + mash_cmd + " returned " + str(mash_info.returncode) + ": "+e.message+"\n")
            sys.exit(1)

    return seqs

########################
# construct a database #
########################

# Multithread wrapper around sketch
def constructDatabase(assemblyList, klist, sketch, oPrefix, threads = 1, mash_exec = 'mash'):

    # create kmer databases
    l = Lock()
    pool = Pool(processes=threads, initializer=init_lock, initargs=(l,))
    pool.map(partial(runSketch,assemblyList=assemblyList, sketch=sketch, oPrefix=oPrefix, mash_exec=mash_exec), klist)
    pool.close()
    pool.join()

# lock on stderr
def init_lock(l):
    global lock
    lock = l

# create kmer databases
def runSketch(k, assemblyList, sketch, oPrefix, mash_exec = 'mash'):
    lock.acquire()
    sys.stderr.write("Creating mash database for k = " + str(k) + "\n")
    lock.release()

    dbname = "./" + oPrefix + "/" + oPrefix + "." + str(k)
    if not os.path.isfile(dbname + ".msh"):
        mash_cmd = mash_exec + " sketch -w 1 -s " + str(sketch[k]) + " -o " + dbname + " -k " + str(k) + " -l " + assemblyList + " 2> /dev/null"
        subprocess.run(mash_cmd, shell=True, check=True)
    else:
        lock.acquire()
        sys.stderr.write("Found existing mash database " + dbname + ".msh for k = " + str(k) + "\n")
        lock.release()

####################
# query a database #
####################

def queryDatabase(qFile, klist, dbPrefix, self = True, mash_exec = 'mash', threads = 1):

    queryList = readAssemblyList(qFile)
    refList = getSeqsInDb("./" + dbPrefix + "/" + dbPrefix + "." + str(klist[0]) + ".msh", mash_exec)

    # write the order which comparisons will appear in the output array
    number_pairs = 0
    querySeqs = []
    refSeqs = []
    # TODO write to disk instead of storing?
    if self:
        number_pairs = 0.5 * len(refList) * (len(refList) - 1)
        for i, ref in enumerate(0, refList):
            for j in range(i + 1, len(refList)):
                refSeqs.append(ref)
                querySeqs.append(refList[j])
    else:
        for query in queryList:
            for ref in refList:
                if ref != query:
                    refSeqs.append(ref)
                    querySeqs.append(query)
                    number_pairs += 1

    # Pre-assign array for storage
    raw = np.zeros((number_pairs, len(klist))

    # iterate through kmer lengths
    for k_idx, k in enumerate(klist):
        # run mash distance query based on current file
        dbname = "./" + dbPrefix + "/" + dbPrefix + "." + str(k) + ".msh"

        row = 0
        try:
            mash_cmd = mash_exec + " dist -p " + str(threads)
            if self:
                mash_cmd += " " + dbname + " " + dbname
            else:
                mash_cmd += " -l " + dbname + " " + qFile
            mash_cmd += " 2> " + dbPrefix + ".err.log"
            sys.stderr.write(mash_cmd)

            rawOutput = subprocess.Popen(mash_cmd, shell=True, stdout=subprocess.PIPE)

            prev_ref = ""
            skip = 0
            skipped = 0
            for line in rawOutput.stdout.readlines():
                # Skip the first row with self and symmetric elements
                if skipped < skip:
                    skipped += 1
                    continue

                mashVals = line.decode().rstrip().split("\t")
                if (len(mashVals) > 2):
                    if self and mashVals[1] != prev_ref:
                        prev_ref = mashVals[1]
                        skip += 1
                        skipped = 0
                    else:
                        mashMatch = mashVals[-1].split('/')
                        raw[row][k_idx] = float(mashMatch[0])/int(mashMatch[1])
                        row += 1
        except:
            sys.stderr.write("mash dist command failed\n")
            sys.exit(1)

        os.remove(dbPrefix + ".err.log")

    # run pairwise analyses across kmer lengths
    # Hessian = 0, so Jacobian for regression is a constant
    jacobian = -np.hstack((np.ones((klist.shape[0], 1)), klist.reshape(-1, 1)))
    distMat = np.apply_along_axis(fitKmerCurve, axis = 0, arr = raw, klist, jacobian)

    return(refSeqs, querySeqs, distMat)

# fit the function pr = (1-a)(1-c)^k
def fitKmerCurve(pairwise, klist, jacobian):

    if jacobian = None:
        jacobian = -np.hstack((np.ones((klist.shape[0], 1)), klist.reshape(-1, 1)))

    pairwise.append(np.log(raw[query][ref][str(k)]))
    # curve fit pr = (1-a)(1-c)^k
    # log pr = log(1-a) + k*log(1-c)
    distFit = optimize.least_squares(fun=lambda p, x, y: y - (p[0] + p[1] * x),
                                     x0=[0.0, -0.01],
                                     jac=lambda p, x, y: jacobian,
                                     args=(klist, np.log(pairwise)),
                                     bounds=([-np.inf, -np.inf], [0, 0]))
    transformed_params = 1 - np.exp(distFit.x)

    return(transformed_params)


##############################
# write query output to file #
##############################

def printQueryOutput(rlist, qlist, X, outPrefix):

    # open output file
    outFileName = outPrefix + "/" + outPrefix + ".search.out"
    with open(outFileName, 'w') as oFile:
        oFile.write("\t".join(['Query', 'Reference', 'Core', 'Accessory']) + "\n")
        # add results
        for i, (query, ref) in enumerate(zip(qlist, rlist)):
            oFile.write("\t".join([query, ref, str(X[i,0]), str(X[i,1])]) + "\n")

