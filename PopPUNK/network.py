'''Network functions'''

# universal
import os
import sys
import re
# additional
import glob
import shutil
import subprocess
import networkx as nx
import numpy as np

from .mash import createDatabaseDir
from .mash import constructDatabase
from .mash import queryDatabase
from .mash import getDatabaseName
from .mash import getSketchSize
from .mash import iterDistRows

from .bgmm import assign_samples
from .bgmm import findWithinLabel

#######################################
# extract references based on cliques #
#######################################

def extractReferences(G, outPrefix):

    # define reference list
    references = {}
    # extract cliques from network
    cliques = list(nx.find_cliques(G))
    # order list by size of clique
    cliques.sort(key = len, reverse=True)
    # iterate through cliques
    for clique in cliques:
        alreadyRepresented = 0
        for node in clique:
            if node in references:
                alreadyRepresented = 1
        if alreadyRepresented == 0:
            references[clique[0]] = 1

    # write references to file
    refFileName = "./" + outPrefix + "/" + outPrefix + ".refs"
    with open(refFileName, 'w') as rFile:
        for ref in references:
            rFile.write(ref + '\n')

    return refFileName

####################################
# Construct network from model fit #
####################################

def constructNetwork(rlist, qlist, assignments, within_label):
    """Construct an unweighted, undirected network without self-loops.
    Nodes are samples and edges where samples are within the same cluster

    Will print summary statistics about the network to `STDERR`

    Args:
        rlist (list)
            List of reference sequence labels
        qlist (list)
            List of query sequence labels
        assignments (numpy.array)
            Labels of most likely cluster assignment from :func:`~PopPUNK.bgmm.assign_samples`
        within_label (int)
            The label for the cluster representing within-strain distances
            from :func:`~PopPUNK.bgmm.findWithinLabel`

    Returns:
        G (networkx.Graph)
            The resulting network
    """
    connections = []
    for assignment, (ref, query) in zip(assignments, iterDistRows(rlist, qlist, self=True)):
        if assignment == within_label:
            connections.append((ref, query))

    # build the graph
    G = nx.Graph()
    G.add_nodes_from(rlist)
    for connection in connections:
        G.add_edge(*connection)

    # give some summaries
    components = nx.number_connected_components(G)
    density = nx.density(G)
    transitivity = nx.transitivity(G)
    score = transitivity * (1-density)
    sys.stderr.write("Network summary:\n" + "\n".join(["\tComponents\t" + str(components),
                                                       "\tDensity\t" + "{:.4f}".format(density),
                                                       "\tTransitivity\t" + "{:.4f}".format(transitivity),
                                                       "\tScore\t" + "{:.4f}".format(score)])
                                                       + "\n")

    return G

#########################################
# Update clustering CSV following query #
#########################################

def updateClustering(dbPrefix, existingHits):

    # identify whether there are any clusters to be merged
    # use existingHits => dict (per query) of dict of hits to clusters (full, decimalised)
    groups = 0
    # forMerging: dict of lists; group => list of clusters assigned to it, which are to be merged
    forMerging = {}
    # beingMerged: dict of lists; cluster => groups to which it has been assigned
    beingMerged = {}

    for query in existingHits:
        newList = {}
        # find queries that match multiple clusters
        # BUT these could be diff
        if len(existingHits[query].keys()) > 1:
            # store multiple hits just as leading integers
            for hit in existingHits[query]:
                # list all the clusters to which a query matches
                # as the dict newList
                #                newList[hit.split('.')[0]] = 1
                newList[hit] = 1
        # if query matches multiple integer clusters
        if len(newList.keys()) > 1:
            groups += 1
            # make a dict of lists; each list is a group of clusters
            # to be merged into a supergroup
            forMerging[groups] = newList.keys()
            for cluster in newList:
                # list all groups to which the clusters have been
                # assigned in "beingMerged" in case of multiple links
                if cluster not in beingMerged:
                    beingMerged[cluster] = []
                beingMerged[cluster].append(groups)

    # stop function early if no merging required
    superGroups = {}
    if groups > 0:
        # recursively build supergroups based on the lists of cross-linked clusters
        # processed dict stores those groups that have already been looked at
        processed = {}
        for group in range(1, groups):
            if group not in processed:
                # iterate over each cluster in group and assign to common supergroup
                for cluster in forMerging[group]:
                    superGroups[cluster] = group
                    sys.stderr.write("cluster " + str(cluster) + " group " + str(group) + '\n')
                # newLinks stores the clusters being added to supergroup i
                newLinks = []
                newLinks.append(group)
                # iterative search until no new clusters in newLinks to be appended to supergroup
                while len(newLinks) > 0:
                    # nextLevel looks at the groups associated with the clusters in beingMerged
                    # then appends all the clusters in any new groups using the forMerging dict
                    additionalLinks, processed = nextLevel(group, newLinks, beingMerged, forMerging, processed)
                    newLinks = additionalLinks
                    # append the new finds to our supergroup
                    if len(newLinks) > 0:
                        for cluster in newLinks:
                            superGroups[cluster] = group
                            sys.stderr.write("cluster " + str(cluster) + " group " + str(group) + '\n')

        # change indexing of superGroups to keep things minimal
        newClusterTranslation = {}
        for cluster in superGroups:
            sys.stderr.write("cluster " + str(cluster) + " supergroup " + str(superGroups[cluster]) + '\n')
            if superGroups[cluster] not in newClusterTranslation or newClusterTranslation[superGroups[cluster]] > cluster:
                newClusterTranslation[superGroups[cluster]] = cluster
                sys.stderr.write("cluster " + str(cluster) + " supergroup " + str(superGroups[cluster]) +
                        " translate " + str(newClusterTranslation[superGroups[cluster]]) + '\n')

        # parse original clustering
        maxCluster = 0
        oldClustering = {}
        oldClusteringCsvName = "./" + dbPrefix + "/" + dbPrefix + "_clusters.csv"
        with open(oldClusteringCsvName, 'r') as oldFile:
            for line in oldFile:
                clusteringVals = line.strip().split(',')
                if clusteringVals[0] != "Taxon":
                    oldClustering[clusteringVals[0]] = clusteringVals[1]
                    if int(clusteringVals[1].split('.')[0]) > maxCluster:
                        maxCluster = int(clusteringVals[1].split('.')[0])

        maxCluster += 1

        # print new clustering file
        newClusteringCsvName = "./" + dbPrefix + "/new." + dbPrefix + "_clusters.csv"
        newFile = ""
        with open(oldClusteringCsvName, 'r') as oldFile, open(newClusteringCsvName, 'w') as newFile:
            newFile.write("Taxon,Cluster\n")
            # update original clustering
            for line in oldFile:
                clusteringVals = line.rstrip().split(',')
                if clusteringVals[0] != "Taxon":
                    intCluster = int(clusteringVals[1].split('.')[0])
                    if intCluster in superGroups:
                        if superGroups[intCluster] in newClusterTranslation:
                            newFile.write(clusteringVals[0] + ',' +
                                    str(newClusterTranslation[superGroups[intCluster]]) +
                                    "." + str(clusteringVals[1]) + '\n')
                        else:
                            sys.stderr.write("Problem with supergroup " + superGroups[intCluster] + '\n')
                            sys.exit(1)
                    else:
                        newFile.write(line)
            # now add new query hits
            for query in existingHits:
                # if the query hit multiple groups and now matches a supergroup
                if len(existingHits[query].keys()) >= 1:
                    assignation = getAssignation(query, existingHits[query], superGroups, newClusterTranslation)
                    newFile.write(q + ',' + str(assignation) + '\n')

        # now update the cluster assignation file
        os.rename(newClusteringCsvName, oldClusteringCsvName)

######################################################################
# Iterative link search function for clustering non-matching queries #
######################################################################
def nextLevel(group, newLinks, beingMerged, forMerging, processed):

    newAdditions = []
    for cluster in beingMerged:
        if len(beingMerged[cluster]) > 1:
            for merge_group in beingMerged[cluster]:
                if group != merge_group and merge_group not in processed:
                    for cluster in forMerging[merge_group]:
                        newAdditions.append(cluster)
                processed[merge_group] = True

    return newAdditions, processed

############################################
# Get cluster to which query is assigned   #
############################################
def getAssignation(query, existingQueryHits, newFile, superGroups, newClusterTranslation):
    for exstingHit in existingQueryHits:
        if existingHit in superGroups:
            if superGroups[existingHit] in newClusterTranslation:
                assignation = newClusterTranslation[superGroups[existingHit]]
            else:
                sys.stderr.write("Problem with supergroup " + superGroups[existingHit] + '\n')
                sys.exit(1)
        else:
            assignation = existingHit

    return assignation

##########################################
# Identify links to network from queries #
##########################################

def findQueryLinksToNetwork(rlist, qlist, self, kmers, assignments, weights, means,
        covariances, outPrefix, dbPrefix, batchSize, threads = 1, mash_exec = 'mash'):

    # identify within-strain links (closest component to origin)
    within_label = findWithinLabel(means, assignments)

    # initialise links data structure
    links = {}
    for query in qlist:
        if query not in links:
            links[query] = []

    # store links for each query in a dict of lists: links[query] => list of hits
    for assignment, (ref, query) in zip(assignments, iterDistRows(rlist, qlist, self=False)):
        if assignment == within_label:
            links[query].append(ref)

    # identify potentially new lineages in list: unassigned is a list of queries with no hits
    unassigned = []
    for query in links:
        if len(links[query]) == 0:
            unassigned.append(query)

    # process unassigned query sequences, if there are any
    if len(unassigned) > 0:

        # write unassigned queries to file as if a list of references
        tmpDirString = "tmp_" + outPrefix
        tmpFileName = tmpDirString + ".in"
        createDatabaseDir(tmpDirString)
        with open(tmpFileName, 'w') as tFile:
            for query in unassigned:
                tFile.write(query + '\n')

        # use database construction methods to find links between unassigned queries
        sketchSize = getSketchSize(dbPrefix, kmers, mash_exec)
        constructDatabase(tmpFileName, kmers, sketchSize, tmpDirString, threads, mash_exec)
        qlist1, qlist2, distMat = queryDatabase(tmpFileName, kmers, tmpDirString, True, mash_exec, threads)
        queryAssignation = assign_samples(distMat, weights, means, covariances)

        # identify any links between queries and store in the same links dict
        # links dict now contains lists of links both to original database and new queries
        for assignment, (ref, query) in zip(assignments, iterDistRows(rlist, qlist, self=True)):
            if assignment == within_label:
                links[query].append(ref)

        # build network based on connections between queries
        # store links as a network
        G = constructNetwork(qlist1, qlist2, queryAssignation, findWithinLabel(means, queryAssignation))
        # not used?
        #clusters = sorted(nx.connected_components(G), key=len, reverse=True)
        #cl_id = 1
        #outFileName = "tmp_" + outPrefix + "_clusters.csv"

        # remove directory
        shutil.rmtree("./" + tmpDirString)
        os.remove(tmpFileName)
        os.remove("tmp_" + outPrefix + ".err.log")

    # finish by returning network and dict of query-ref and query-query link lists
    return links, G

####################################################
# Update reference database with query information #
####################################################

def updateDatabase(dbPrefix, additionalIsolates, G, outPrefix, full_db=False, threads = 1, mash_exec = 'mash'):

    # append information to csv
    clusteringCsvName = "./" + dbPrefix + "/" + dbPrefix + "_clusters.csv"
    with open(clusteringCsvName, 'a') as cFile:
        for genome in additionalIsolates:
            cFile.write(genome + "," + str(additionalIsolates[genome]) + '\n')

    # network is composed of links between queries that do not match
    # any existing references
    # extract cliques from network
    # identify new reference sequences
    if not full_db:
        referenceFile = extractReferences(G,outPrefix)
    else:
        referenceFile = "./" + outPrefix + "/" + outPrefix + ".refs"
        with open(referenceFile, 'w') as rFile:
            for ref in G.nodes():
                rFile.write(ref + '\n')

    # identify kmers used to construct original database
    dbFileList = glob.glob(dbPrefix + "/" + dbPrefix + ".*.msh")
    klist = []
    for filename in dbFileList:
        k = re.search(r'\d+', filename)
        if k:
            klist.append(k.group(0))

    # identify sketch lengths used to generate databases
    sketch = getSketchSize(dbPrefix, klist)

    # make new databases and append
    createDatabaseDir(outPrefix)
    constructDatabase(referenceFile, klist, sketch, outPrefix, threads, mash_exec)
    for k in klist:
        f1 = getDatabaseName(dbPrefix, k)
        f2 = getDatabaseName(outPrefix, k)
        try:
            subprocess.run("mash paste tmp." + dbPrefix + "." + k + " " + f1 + " " + f2 + " > /dev/null 2> /dev/null",
                shell=True, check=True)
            os.rename("tmp." + dbPrefix + "." + k + ".msh", f1)
        except:
            sys.stderr.write("Failed to combine databases " + f1 + " and " + f2 + "\n")
            sys.exit(1)

############################
# Print network components #
############################

def printClusters(G, outPrefix):

    # data structure
    clustering = {}

    # identify network components
    clusters = sorted(nx.connected_components(G), key=len, reverse=True)
    cl_id = 1
    outFileName = outPrefix + "/" + outPrefix + "_clusters.csv"

    # print clustering to file
    with open(outFileName, 'w') as cluster_file:
        cluster_file.write("Taxon,Cluster\n")
        for cl_id, cluster in enumerate(clusters):
            for cluster_member in cluster:
                cluster_file.write(",".join((cluster_member,str(cl_id))) + "\n")
                clustering[cluster_member] = cl_id

    return clustering


