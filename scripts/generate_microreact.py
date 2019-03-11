#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018 John Lees and Nick Croucher

import pickle
import sys
import numpy as np
import argparse
import os
import dendropy
import pandas as pd
from sklearn import manifold
from collections import defaultdict

sys.path.append(os.path.dirname(__file__) + '/../PopPUNK/')
#from PopPUNK.utils import update_distance_matrices
#from utils import iterDistRows

#from tsne import generate_tsne

#from plot import outputsForMicroreact

# command line parsing
def get_options():

    parser = argparse.ArgumentParser(description='Generate files for microreact from successful, or unsuccessful, clustering runs', prog='generate_microreact')

    # input options
    #parser.add_argument('--input', required=True, help='Input list of sequences (required)')
    parser.add_argument('--distances', required=True, help='Prefix of input pickle and numpy file of pre-calculated distances (required)')
    parser.add_argument('--clustering-csv', help='CSV of clustering', default = None)
    parser.add_argument('--info-csv', help='CSV of epidemiological information')
    parser.add_argument('--rapidnj', help='Path to rapidNJ binary to build NJ tree for Microreact', default = None)
    parser.add_argument('--overwrite', help='Overwrite existing output', default = False, action = 'store_true')
    parser.add_argument('--perplexity', type=float, default = 20.0,
                         help='Perplexity used to calculate t-SNE projection [default=20.0]')
    parser.add_argument('--output', required=True, help='Name of output file')

    return parser.parse_args()

def iterDistRows(refSeqs, querySeqs, self=True):
    """Gets the ref and query ID for each row of the distance matrix

    Returns an iterable with ref and query ID pairs by row.

    Args:
        refSeqs (list)
            List of reference sequence names.
        querySeqs (list)
            List of query sequence names.
        self (bool)
            Whether a self-comparison, used when constructing a database.

            Requires refSeqs == querySeqs

            Default is True
        Returns:
            ref, query (str, str)
                Iterable of tuples with ref and query names for each distMat row.
        """
    if self:
        if refSeqs != querySeqs:
            raise RuntimeError('refSeqs must equal querySeqs for db building (self = true)')
        for i, ref in enumerate(refSeqs):
            for j in range(i + 1, len(refSeqs)):
                yield(refSeqs[j], ref)
    else:
        for query in querySeqs:
            for ref in refSeqs:
                yield(ref, query)

###########
def outputsForMicroreact(combined_list, coreMat, accMat, clustering, perplexity, outPrefix, epiCsv,
                         rapidnj, queryList = None, overwrite = False):
    
    # generate sequence labels
    seqLabels = [r.split('/')[-1].split('.')[0] for r in combined_list]
    
    # write the phylogeny .nwk; t-SNE network .dot; clusters + data .csv
    generate_phylogeny(coreMat, seqLabels, outPrefix, "_core_NJ.nwk", rapidnj, overwrite)
    generate_tsne(seqLabels, accMat, perplexity, outPrefix, overwrite)
    writeClusterCsv(outPrefix + "/" + os.path.basename(outPrefix) + "_microreact_clusters.csv",
                    combined_list, combined_list, clustering, 'microreact', epiCsv, queryList)

def generate_phylogeny(coreMat, seqLabels, outPrefix, tree_suffix, rapidnj, overwrite):

    # Save distances to file
    core_dist_file = outPrefix + "/" + os.path.basename(outPrefix) + "_core_dists.csv"
    np.savetxt(core_dist_file, coreMat, delimiter=",", header = ",".join(seqLabels), comments="")
    
    # calculate phylogeny
    tree_filename = outPrefix + "/" + os.path.basename(outPrefix) + tree_suffix
    if overwrite or not os.path.isfile(tree_filename):
        sys.stderr.write("Building phylogeny\n")
        if rapidnj is not None:
            tree = buildRapidNJ(rapidnj, seqLabels, coreMat, outPrefix, tree_filename)
        else:
            pdm = dendropy.PhylogeneticDistanceMatrix.from_csv(src=open(core_dist_file),
                                                               delimiter=",",
                                                               is_first_row_column_names=True,
                                                               is_first_column_row_names=False)
            tree = pdm.nj_tree()
        
        # Not sure why, but seems that this needs to be run twice to get
        # what I would think of as a midpoint rooted tree
        tree.reroot_at_midpoint(update_bipartitions=True, suppress_unifurcations=False)
        tree.reroot_at_midpoint(update_bipartitions=True, suppress_unifurcations=False)
        tree.write(path=tree_filename,
                   schema="newick",
                   suppress_rooting=True,
                   unquoted_underscores=True)
    else:
        sys.stderr.write("NJ phylogeny already exists; add --overwrite to replace\n")
    
    # remove file as it can be large
    os.remove(core_dist_file)

def writeClusterCsv(outfile, nodeNames, nodeLabels, clustering, output_format = 'microreact', epiCsv = None, queryNames = None):

    # set order of column names
    colnames = []
    colnames = ['id']
    for cluster_type in clustering:
        col_name = cluster_type + '_Cluster__autocolour'
        colnames.append(col_name)
    if queryNames is not None:
        colnames.append('Status')
        colnames.append('Status__colour')


    # process epidemiological data
    if epiCsv is not None:
        epiData = pd.read_csv(epiCsv, index_col = 0, quotechar='"')
    
    d = defaultdict(list)
    if epiCsv is not None:
        for e in epiData.columns.values:
            colnames.append(str(e))

    columns_to_be_omitted = []

    # process clustering data
    nodeLabels = [r.split('/')[-1].split('.')[0] for r in nodeNames]

    for name, label in zip(nodeNames, nodeLabels):
        if name in clustering['combined']:
            if output_format == 'microreact':
                d['id'].append(label)
                for cluster_type in clustering:
                    col_name = cluster_type + "_Cluster__autocolour"
                    d[col_name].append(clustering[cluster_type][name])
                if queryNames is not None:
                    if name in queryNames:
                        d['Status'].append("Query")
                        d['Status__colour'].append("red")
                    else:
                        d['Status'].append("Reference")
                        d['Status__colour'].append("black")

            if epiCsv is not None:
                # avoid adding
                if len(columns_to_be_omitted) == 0:
                    columns_to_be_omitted = ['id', 'Id', 'ID', 'combined_Cluster__autocolour',
                                             'core_Cluster__autocolour', 'accessory_Cluster__autocolour']
                    for c in d:
                        columns_to_be_omitted.append(c)
                if label in epiData.index:
                    for col, value in zip(epiData.columns.values, epiData.loc[label].values):
                         if col not in columns_to_be_omitted:
                             d[col].append(str(value))
                else:
                    for col in colnames:
                         d[col].append('nan')
        else:
            sys.stderr.write("Cannot find " + name + " in clustering\n")
            sys.exit(1)

    # print CSV
    pd.DataFrame(data=d).to_csv(outfile, columns = colnames, index = False)

def update_distance_matrices(refList, distMat, queryList = None, query_ref_distMat = None,
                             query_query_distMat = None):

    seqLabels = refList
    if queryList is not None:
        seqLabels = seqLabels + queryList

    coreMat = np.zeros((len(seqLabels), len(seqLabels)))
    accMat = np.zeros((len(seqLabels), len(seqLabels)))

    # Fill in symmetric matrices for core and accessory distances
    i = 0
    j = 1

    # ref v ref (used for --create-db)
    for row in distMat:
        coreMat[i, j] = row[0]
        coreMat[j, i] = coreMat[i, j]
        accMat[i, j] = row[1]
        accMat[j, i] = accMat[i, j]

        if j == len(refList) - 1:
            i += 1
            j = i + 1
        else:
            j += 1

    # if query vs refdb (--assign-query), also include these comparisons
    if queryList is not None:

        # query v query - symmetric
        i = len(refList)
        j = len(refList)+1
        for row in query_query_distMat:
            coreMat[i, j] = row[0]
            coreMat[j, i] = coreMat[i, j]
            accMat[i, j] = row[1]
            accMat[j, i] = accMat[i, j]
            if j == (len(refList) + len(queryList) - 1):
                i += 1
                j = i + 1
            else:
                j += 1

        # ref v query - asymmetric
        i = len(refList)
        j = 0
        for row in query_ref_distMat:
            coreMat[i, j] = row[0]
            coreMat[j, i] = coreMat[i, j]
            accMat[i, j] = row[1]
            accMat[j, i] = accMat[i, j]
            if j == (len(refList) - 1):
                i += 1
                j = 0
            else:
                j += 1

    # return outputs
    return seqLabels, coreMat, accMat

def generate_tsne(seqLabels, accMat, perplexity, outPrefix, overwrite, verbosity = 0):

    # generate accessory genome distance representation
    tsne_filename = outPrefix + "/" + os.path.basename(outPrefix) + "_perplexity" + str(perplexity) + "_accessory_tsne.dot"
    if overwrite or not os.path.isfile(tsne_filename):
        sys.stderr.write("Running t-SNE\n")
        accArray_embedded = manifold.TSNE(n_components=2, perplexity=perplexity, verbose=verbosity).fit_transform(np.array(accMat))
        
        # print dot file
        with open(tsne_filename, 'w') as nFile:
            nFile.write("graph G { ")
            for s, seqLabel in enumerate(seqLabels):
                nFile.write('"' + seqLabel + '"' +
                            '[x='+str(5*float(accArray_embedded[s][0]))+',y='+str(5*float(accArray_embedded[s][1]))+']; ')
            nFile.write("}\n")
    else:
        sys.stderr.write("t-SNE analysis already exists; add --overwrite to replace\n")


###########

# main code
if __name__ == "__main__":

    # Check input ok
    args = get_options()

    # open stored distances
    with open(args.distances + ".pkl", 'rb') as pickle_file:
        rlist, qlist, self = pickle.load(pickle_file)
    X = np.load(args.distances + ".npy")

    # get names order
    names = iterDistRows(rlist, qlist, self)

    # open output file
    with open(args.output + "/" + args.output + "_dists.csv", 'w') as oFile:
        oFile.write("\t".join(['Query', 'Reference', 'Core', 'Accessory']) + "\n")
        for i, (ref, query) in enumerate(names):
            oFile.write("\t".join([query, ref, str(X[i,0]), str(X[i,1])]) + "\n")

    # process loaded distances
    combined_seq, core_distMat, acc_distMat = update_distance_matrices(rlist, X)

    # get clustering or create it
    isolateClustering = {}
    isolateClustering['combined'] = {}
    for r in rlist:
        isolateClustering['combined'][r] = "none"

    # generate microreact outputs
    outputsForMicroreact(rlist, core_distMat, acc_distMat, isolateClustering, args.perplexity,
                         args.output, args.info_csv, args.rapidnj, overwrite = args.overwrite)

