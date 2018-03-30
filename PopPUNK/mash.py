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
from random import sample
import numpy as np
import networkx as nx
from scipy import optimize

from .plot import plot_fit

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

def storePickle(rlist, qlist, self, X, pklName):
    with open(pklName + ".pkl", 'wb') as pickle_file:
        pickle.dump([rlist, qlist, self], pickle_file)
    np.save(pklName + ".npy", X)

####################################
# Load distance matrix from pickle #
####################################

def readPickle(pklName):
    with open(pklName + ".pkl", 'rb') as pickle_file:
        rlist, qlist, self = pickle.load(pickle_file)
    X = np.load(pklName + ".npy")
    return rlist, qlist, self, X

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
                            sys.stderr.write("Problem with database; sketch size for kmer length " +
                                    str(k) + " is " + str(oldSketch) +
                                    ", but smaller kmers have sketch sizes of " + str(sketch) + "\n")
                            sys.exit(1)

                        break

            # Make sure process executed correctly
            mash_info.wait()
            if mash_info.returncode != 0:
                raise RuntimeError('mash info failed')
        except subprocess.CalledProcessError as e:
            sys.stderr.write("Could not get info about " + dbname + "; command " + mash_exec +
                    " info -t " + dbname + " returned " + str(mash_info.returncode) +
                    ": " + e.message + "\n")
            sys.exit(1)

    return sketchdb

# Return an array with the sequences in the passed mash database
def getSeqsInDb(mashSketch, mash_exec = 'mash'):

    seqs = []
    mash_cmd = str(mash_exec) + " info -t " + str(mashSketch)
    try:
        mash_info = subprocess.Popen(mash_cmd, shell=True, stdout=subprocess.PIPE)
        for line in iter(mash_info.stdout.readline, ''):
            line = line.rstrip().decode()
            if line != '':
                if line.startswith("#") is False:
                    seqs.append(line.split("\t")[2])
            else:
                mash_info.wait()
                break

        # Make sure process executed correctly
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

    # Genome length needed to calculate prob of random matches
    genome_length = 1 # min of 1 to avoid div/0 errors
    with open(assemblyList, 'r') as assemblyFiles:
       exampleFile = assemblyFiles.readline()
       with open(exampleFile.rstrip(), 'r') as exampleAssembly:
           for line in exampleAssembly:
               if line[0] != ">":
                   genome_length += len(line.rstrip())

    # create kmer databases
    if threads > len(klist):
        num_processes = 1
        num_threads = threads
    else:
        num_processes = threads
        num_threads = 1

    l = Lock()
    with Pool(processes=num_processes, initializer=init_lock, initargs=(l,)) as pool:
        pool.map(partial(runSketch, assemblyList=assemblyList, sketch=sketch, genome_length=genome_length,
                                    oPrefix=oPrefix, mash_exec=mash_exec, threads=num_threads),
                 klist)

# lock on stderr
def init_lock(l):
    global lock
    lock = l

# create kmer databases
def runSketch(k, assemblyList, sketch, genome_length, oPrefix, mash_exec = 'mash', threads = 1):
    lock.acquire()
    sys.stderr.write("Creating mash database for k = " + str(k) + "\n")

    dbname = "./" + oPrefix + "/" + oPrefix + "." + str(k)
    if not os.path.isfile(dbname + ".msh"):

        random_prob = 1/(pow(4, k)/float(genome_length) + 1)
        sys.stderr.write("Random " + str(k) + "-mer probability: " + "{:.2f}".format(random_prob) + "\n")
        lock.release()

        # Run sketch
        mash_cmd = mash_exec + " sketch -w 1 -p " + str(threads) + " -s " + str(sketch[k]) + " -o " + dbname + " -k " + str(k) + " -l " + assemblyList + " 2> /dev/null"
        subprocess.run(mash_cmd, shell=True, check=True)
    else:
        sys.stderr.write("Found existing mash database " + dbname + ".msh for k = " + str(k) + "\n")
        lock.release()

####################
# query a database #
####################

def queryDatabase(qFile, klist, dbPrefix, self = True, number_plot_fits = 0, mash_exec = 'mash', threads = 1):

    queryList = []
    with open(qFile, 'r') as queryFile:
        for line in queryFile:
            queryList.append(line.rstrip())
    refList = getSeqsInDb("./" + dbPrefix + "/" + dbPrefix + "." + str(klist[0]) + ".msh", mash_exec)

    if self:
        number_pairs = int(0.5 * len(refList) * (len(refList) - 1))
    else:
        number_pairs = int(len(refList) * len(queryList))

    # Pre-assign array for storage. float32 sufficient accuracy for 10**4 sketch size, halves memory use
    raw = np.zeros((number_pairs, len(klist)), dtype=np.float32)

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
            sys.stderr.write(mash_cmd + "\n")

            rawOutput = subprocess.Popen(mash_cmd, shell=True, stdout=subprocess.PIPE)

            # Check mash output is consistent with expected order
            # This is ok in all tests, but best to check and exit in case something changes between mash versions
            expected_names = iterDistRows(refList, queryList, self=True)

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
                        skipped = 1
                    else:
                        mashMatch = mashVals[-1].split('/')
                        (e_ref, e_query) = next(expected_names)
                        if mashVals[0] == e_ref and mashVals[1] == e_query:
                            raw[row, k_idx] = float(mashMatch[0])/int(mashMatch[1])
                            row += 1
                        else:
                            sys.stderr.write("mash dist output order:" + e_query + "," + e_ref + "\n" +
                                             "not as expected: " + mashVals[0] + "," + mashVals[1] + "\n")
                            sys.exit(1)

                # EOF
                if line == '':
                    break

            rawOutput.wait()
            if rawOutput.returncode != 0:
                raise RuntimeError('mash dist failed')
            else:
                os.remove(dbPrefix + ".err.log")
        except:
            sys.stderr.write("mash dist command failed\n")
            sys.exit(1)

    # Pre-assign return (to higher precision)
    sys.stderr.write("Calculating core and accessory distances\n")

    # Hessian = 0, so Jacobian for regression is a constant
    jacobian = -np.hstack((np.ones((klist.shape[0], 1)), klist.reshape(-1, 1)))

    # option to plot core/accessory fits. Choose a random number from cmd line option
    if number_plot_fits > 0:
        examples = sample(range(number_pairs), k=number_plot_fits)
        for plot_idx, plot_example in enumerate(sorted(examples)):
            fit = fitKmerCurve(raw[plot_example, :], klist, jacobian)
            plot_fit(klist, raw[plot_example, :], fit,
                    dbPrefix + "/fit_example_" + str(plot_idx + 1),
                    "Example fit " + str(plot_idx + 1) + " (row " + str(plot_example) + ")")

    # run pairwise analyses across kmer lengths, mutating distMat
    with Pool(processes=threads) as pool:
        distMat = pool.map(partial(fitKmerCurve, klist=klist, jacobian=jacobian), raw, chunksize = int(number_pairs / threads))
    del raw
    distMat = np.vstack(distMat)

    return(refList, queryList, distMat)

# fit the function pr = (1-a)(1-c)^k
# supply jacobian = -np.hstack((np.ones((klist.shape[0], 1)), klist.reshape(-1, 1)))
def fitKmerCurve(pairwise, klist, jacobian):
    # curve fit pr = (1-a)(1-c)^k
    # log pr = log(1-a) + k*log(1-c)
    distFit = optimize.least_squares(fun=lambda p, x, y: y - (p[0] + p[1] * x),
                                     x0=[0.0, -0.01],
                                     jac=lambda p, x, y: jacobian,
                                     args=(klist, np.log(pairwise)),
                                     bounds=([-np.inf, -np.inf], [0, 0]))
    transformed_params = 1 - np.exp(distFit.x)

    return(transformed_params)

# Gets the ref and query ID for each row of the distance matrix
def iterDistRows(refSeqs, querySeqs, self=True):
    if self:
        if refSeqs != querySeqs:
            raise RuntimeError('refSeqs must equal querySeqs for db building (self = true)')
        for i, ref in enumerate(refSeqs):
            for j in range(i + 1, len(refSeqs)):
                yield(refSeqs[j], ref)
    else:
        for ref in refSeqs:
            for query in querySeqs:
                yield(ref, query)

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

