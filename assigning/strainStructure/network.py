'''Network functions'''

# universal
import os
import sys
import argparse
import re
# additional
import networkx as nx
import numpy as np
import pymc3 as pm
# import strainStructure package
import strainStructure

#######################################
# extract references based on cliques #
#######################################

def extractReferences(G,outPrefix):
    
    # define reference list
    references = {}
    # extract cliques from network
    cliques = list(nx.find_cliques(G))
    # order list by size of clique
    cliques.sort(key = len,reverse=True)
    # iterate through cliques
    for clique in cliques:
        alreadyRepresented = 0
        for node in clique:
            if node in references:
                alreadyRepresented = 1
        if alreadyRepresented == 0:
            references[clique[0]] = 1

    # write references to file
    refFileName = "./"+outPrefix+"/"+outPrefix+".refs"
    rFile = ""
    try:
        rFile = open(refFileName,'w')
    except:
        sys.exit("Cannot write to reference file "+refFileName)
        for ref in references:
            print(ref,file=rFile)
    rFile.close()
    
    return refFileName

####################################
# Construct network from model fit #
####################################

def constructNetwork(rlist,qlist,assignments,weights,means,covariances):
    
    # identify within-strain links
    within_label = np.argmin(np.apply_along_axis(np.linalg.norm, 1, means))
    samples = set()
    connections = []
    distances = []
    for i in range(0,len(rlist)):
        if (rlist[i] != qlist[i]):    # remove self matches
            if rlist[i] not in samples:
                samples.add(rlist[i])
            if qlist[i] not in samples:
                samples.add(qlist[i])
            if assignments[i] == within_label:
                connections.append((rlist[i], qlist[i]))

    # build the graph
    G = nx.Graph()
    G.add_nodes_from(samples)
    for connection in connections:
        G.add_edge(*connection)

    return G

######################################################################
# Recursive link search function for clustering non-matching queries #
######################################################################

def nextLevel(i,n,b,f,p):
    
    newAdditions = []
    for cluster in b:
        if len(b[cluster]) > 1:
            for j in b[cluster]:
                if i != j and j not in p:
                    for cluster in f[j]:
                        newAdditions.append(cluster)
                p[j] = 1

    return newAdditions,p

#########################################
# Update clustering CSV following query #
#########################################

def updateClustering(dbPrefix,existingHits):
    
    # identify whether there are any clusters to be merged
    # use existingHits => dict (per query) of dict of hits to clusters (full, decimalised)
    group = 0
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
            group = group+1
            # make a dict of lists; each list is a group of clusters
            # to be merged into a supergroup
            forMerging[group] = newList.keys()
            for cluster in newList:
                # list all groups to which the clusters have been
                # assigned in "beingMerged" in case of multiple links
                if cluster not in beingMerged:
                    beingMerged[cluster] = []
                beingMerged[cluster].append(group)

    # stop function early if no merging required
    superGroups = {}
    if group == 0:
        return None
    else:
        # recursively build supergroups based on the lists of cross-linked clusters
        # processed dict stores those groups that have already been looked at
        processed = {}
        for i in range(1,group):
            if i not in processed:
                # iterate over each cluster in group and assign to common supergroup
                for cluster in forMerging[i]:
                    superGroups[cluster] = i
                    print("cluster "+str(cluster)+" i "+str(i))
                # newLinks stores the clusters being added to supergroup i
                newLinks = []
                newLinks.append(i)
                # recursive search until no new clusters in newLinks to be appended to supergroup
                while len(newLinks) > 0:
                    # nextLevel looks at the groups associated with the clusters in beingMerged
                    # then appends all the clusters in any new groups using the forMerging dict
                    additionalLinks,processed = nextLevel(i,newLinks,beingMerged,forMerging,processed)
                    newLinks = additionalLinks
                    # append the new finds to our supergroup
                    if len(newLinks) > 0:
                        for cluster in newLinks:
                            superGroups[cluster] = i
                            print("cluster "+str(cluster)+" i "+str(i))

    # change indexing of superGroups to keep things minimal
    newClusterTranslation = {}
    for cluster in superGroups:
        print("cluster "+str(cluster)+" supergroup "+str(superGroups[cluster]))
        if superGroups[cluster] not in newClusterTranslation or newClusterTranslation[superGroups[cluster]] > cluster:
            newClusterTranslation[superGroups[cluster]] = cluster
            print("cluster "+str(cluster)+" supergroup "+str(superGroups[cluster])+" translate "+str(newClusterTranslation[superGroups[cluster]]))

    # parse original clustering
    maxCluster = 0
    oldClustering = {}
    oldClusteringCsvName = "./"+dbPrefix+"/"+dbPrefix+"_clusters.csv"
    oldFile = ""
    try:
        oldFile = open(oldClusteringCsvName,'r')
    except:
        sys.exit("Cannot read old clustering file "+oldClusteringCsvName+" prior to updating")
        for line in oldFile.readlines():
            clusteringVals = line.strip().split(',')
            if clusteringVals[0] != "Taxon":
                oldClustering[clusteringVals[0]] = clusteringVals[1]
                if int(clusteringVals[1].split('.')[0]) > maxCluster:
                    maxCluster = int(clusteringVals[1].split('.')[0])
    oldFile.close()
        
    maxCluster = maxCluster+1
    
    # print new clustering file
    newClusteringCsvName = "./"+dbPrefix+"/new."+dbPrefix+"_clusters.csv"
    newFile = ""
    try:
        oldFile = open(oldClusteringCsvName,'r')
        newFile = open(newClusteringCsvName,'w')
    except:
        sys.exit("Cannot write to new clustering file "+newClusteringCsvName+" prior to updating")
        print("Taxon,Cluster",file=newFile)
        # update original clustering
        for line in oldFile.readlines():
            clusteringVals = line.strip().split(',')
            if clusteringVals[0] != "Taxon":
                intCluster = int(clusteringVals[1].split('.')[0])
                if intCluster in superGroups:
                    if superGroups[intCluster] in newClusterTranslation:
                        print(clusteringVals[0]+','+str(newClusterTranslation[superGroups[intCluster]])+"."+str(clusteringVals[1]),file=newFile)
                    else:
                        sys.exit("Problem with supergroup "+superGroups[intCluster])
                else:
                    print(line.strip(),file=newFile)
    # now add new query hits
    for query in existingHits:
        # if the query hit multiple groups and now matches a supergroup
        if len(existingHits[query].keys()) >= 1:
            strainStructure.printAssignation(query,existingHits[query],newFile,superGroups,newClusterTranslation)

        newFile.close()

    os.system("mv "+newClusteringCsvName+" "+oldClusteringCsvName)
    
    # now update the cluster assignation file
    
    return None

############################################
# Print cluster to which query is assigned #
############################################

def printAssignation(q,e,f,s,t):
    so = ""
    for i in e:
        if i in s:
            if s[i] in t:
                so = t[s[i]]
            else:
                sys.exit("Problem with supergroup "+s[i])
        else:
            so = i
    print(q+','+str(so),file=f)

##########################################
# Get sketch size from existing database #
##########################################

def getSketchSize(dbPrefix,klist):

    # identify sketch lengths used to generate databases
    sketch = 0
    oldSketch = 0
    for k in klist:
        dbname = "./"+dbPrefix+"/"+dbPrefix+"."+str(k)+".msh"
        dbInfo = os.popen("mash info -t "+dbname).read();
        for line in dbInfo.split("\n"):
            if (line.startswith("#") is False):
                sketchValues = line.split("\t")
                if len(sketchValues[0]) > 0:
                    if oldSketch == 0:
                        oldSketch = int(sketchValues[0])
                    else:
                        oldSketch = sketch
                    sketch = int(sketchValues[0])
                    if (sketch != oldSketch):
                        sys.exit("Problem with database; not all files have same sketch size")
                    break

    return sketch

##########################################
# Identify links to network from queries #
##########################################

def findQueryLinksToNetwork(rlist,qlist,kmers,assignments,weights,means,covariances,outPrefix,dbPrefix,batchSize):
    
    # identify within-strain links
    within_label = np.argmin(np.apply_along_axis(np.linalg.norm, 1, means))
    
    # initialise links data structure
    links = {}
    for query in qlist:
        if query not in links:
            links[query] = []
    
    # store links for each query in a dict of lists: links[query] => list of hits
    for i in range(0,len(qlist)-1):
        if assignments[i] == within_label:
            links[qlist[i]].append(rlist[i])

    # identify potentially new lineages in list: unassigned is a list of queries with no hits
    unassigned = []
    for query in links:
        if len(links[query]) == 0:
            unassigned.append(query)

    # define graph here as returned at end
    G = nx.Graph()

    # process unassigned query sequences, if there are any
    if len(unassigned) > 0:
        
        # write unassigned queries to file as if a list of references
        tmpDirString = "tmp_"+outPrefix
        tmpFileName = tmpDirString+".in"
        strainStructure.createDatabaseDir(tmpDirString)
        try:
            tFile = open(tmpFileName,'w')
            for query in unassigned:
                print(query,file=tFile)
            tFile.close()
        except:
            sys.exit("Cannot write to temporary file "+tmpFileName)
    
        # use database construction methods to find links between unassigned queries
        sketchSize = getSketchSize(dbPrefix,kmers)
        strainStructure.constructDatabase(tmpFileName,kmers,sketchSize,tmpDirString)
        qlist1,qlist2,distMat = strainStructure.queryDatabase(tmpFileName,kmers,tmpDirString,batchSize)
        queryAssignation = strainStructure.assign_samples(distMat, weights, means, covariances)
        
        # identify any links between queries and store in the same links dict
        # links dict now contains lists of links both to original database and new queries
        for i in range(0,len(queryAssignation)):
            if queryAssignation[i] == within_label:
                links[qlist1[i]].append(qlist2[i])

        # build network based on connections between queries
        # store links as a network
        G = strainStructure.constructNetwork(qlist1,qlist2,queryAssignation,weights,means,covariances)
        clusters = sorted(nx.connected_components(G), key=len, reverse=True)
        cl_id = 1
        outFileName = "tmp_"+outPrefix+"_clusters.csv"
        
        # remove directory
        try:
            os.system("rm -rf ./"+tmpDirString+ " "+tmpFileName+" "+tmpFileName+".err.log")
        except:
            sys.exit("Unable to remove temporary files")

    # finish by returning network and dict of query-ref and query-query link lists
    return links,G

####################################################
# Update reference database with query information #
####################################################

def updateDatabase(dbPrefix,additionalIsolates,G,outPrefix,f):
    
    # append information to csv
    clusteringCsvName = "./"+dbPrefix+"/"+dbPrefix+"_clusters.csv"
    cFile = ""
    try:
        cFile = open(clusteringCsvName,'a')
    except:
        sys.exit("Cannot append to clustering file "+clusteringCsvName)
    for genome in additionalIsolates:
        print(genome+","+str(additionalIsolates[genome]),file=cFile)
        print("genome: "+genome+","+str(additionalIsolates[genome]))
    cFile.close()

    # network is composed of links between queries that do not match
    # any existing references
    # extract cliques from network
    # identify new reference sequences
    referenceFile = ""
    if f is False:
        referenceFile = strainStructure.extractReferences(G,outPrefix)
    else:
        referenceFile = "./"+outPrefix+"/"+outPrefix+".refs"
        rFile = open(referenceFile,'w')
        for ref in G.nodes():
            print(ref,file=rFile)
        rFile.close()

    # identify kmers used to construct original database
    dbFileList = os.popen("ls -1 "+dbPrefix+"/"+dbPrefix+".*.msh").read();
    klist = []
    for filename in dbFileList.split("\n"):
        k = re.search(r'\d+', filename)
        if k:
            klist.append(k.group(0))

    # identify sketch lengths used to generate databases
    sketch = getSketchSize(dbPrefix,klist)

    # make new databases and append
    strainStructure.createDatabaseDir(outPrefix)
    strainStructure.constructDatabase(referenceFile,klist,sketch,outPrefix)
    for k in klist:
        f1 = strainStructure.getDatabaseName(dbPrefix,k)
        f2 = strainStructure.getDatabaseName(outPrefix,k)
        try:
            os.system("mash paste tmp."+dbPrefix+"."+k+" "+f1+" "+f2+" > /dev/null 2> /dev/null")
        except:
            sys.exit("Failed to combine databases "+f1+" and "+f2)
        try:
            os.system("mv tmp."+dbPrefix+"."+k+".msh "+f1)
        except:
            sys.exit("Unable to update database "+f1)

    return None

############################
# Print network components #
############################

def printClusters(G, outPrefix):
    
    # identify network components
    clusters = sorted(nx.connected_components(G), key=len, reverse=True)
    cl_id = 1
    outFileName = outPrefix+"/"+outPrefix+"_clusters.csv"
    
    # print clustering to file
    try:
        cluster_file = open(outFileName, 'w')
        print("Taxon,Cluster",file=cluster_file)
        for cl_id, cluster in enumerate(clusters):
            for cluster_member in cluster:
                cluster_file.write(",".join((cluster_member,str(cl_id))) + "\n")
        cluster_file.close()
    except:
        sys.exit("Unable to write to file "+outFileName)
    
    return None
