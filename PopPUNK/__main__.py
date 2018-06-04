#!/usr/bin/env python
# Copyright 2018 John Lees and Nick Croucher

# universal
import os
import sys
# additional
import numpy as np
import networkx as nx
import subprocess

# import poppunk package
from .__init__ import __version__

from .mash import checkMashVersion
from .mash import createDatabaseDir
from .mash import storePickle
from .mash import readPickle
from .mash import joinDBs
from .mash import writeTmpFile
from .mash import constructDatabase
from .mash import queryDatabase
from .mash import printQueryOutput
from .mash import readMashDBParams

from .models import *

from .network import constructNetwork
from .network import extractReferences
from .network import writeReferences
from .network import addQueryToNetwork
from .network import printClusters

from .plot import outputsForMicroreact
from .plot import outputsForCytoscape

#################
# run main code #
#################

# command line parsing
def get_options():

    import argparse

    parser = argparse.ArgumentParser(description='PopPUNK (POPulation Partitioning Using Nucleotide Kmers)',
                                     prog='PopPUNK')

    modeGroup = parser.add_argument_group('Mode of operation')
    mode = modeGroup.add_mutually_exclusive_group(required=True)
    mode.add_argument('--easy-run',
            help='Create clusters from assemblies with default settings',
            default=False,
            action='store_true')
    mode.add_argument('--create-db',
            help='Create pairwise distances database between reference sequences',
            default=False,
            action='store_true')
    mode.add_argument('--fit-model',
            help='Fit a mixture model to a reference database',
            default=False,
            action='store_true')
    mode.add_argument('--refine-model',
            help='Refine the accuracy of a fitted model',
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
    iGroup.add_argument('--distances', help='Prefix of input pickle of pre-calculated distances')

    # output options
    oGroup = parser.add_argument_group('Output options')
    oGroup.add_argument('--output', required=True, help='Prefix for output files (required)')
    oGroup.add_argument('--plot-fit', help='Create this many plots of some fits relating k-mer to core/accessory distances '
                                            '[default = 0]', default=0, type=int)
    oGroup.add_argument('--full-db', help='Keep full reference database, not just representatives', default=False, action='store_true')
    oGroup.add_argument('--update-db', help='Update reference database with query sequences', default=False, action='store_true')
    oGroup.add_argument('--overwrite', help='Overwrite any existing database files', default=False, action='store_true')

    # comparison metrics
    kmerGroup = parser.add_argument_group('Kmer comparison options')
    kmerGroup.add_argument('--min-k', default = 9, type=int, help='Minimum kmer length [default = 9]')
    kmerGroup.add_argument('--max-k', default = 29, type=int, help='Maximum kmer length [default = 29]')
    kmerGroup.add_argument('--k-step', default = 4, type=int, help='K-mer step size [default = 4]')
    kmerGroup.add_argument('--sketch-size', default=10000, type=int, help='Kmer sketch size [default = 10000]')

    # model fitting
    modelGroup = parser.add_argument_group('Model fit options')
    modelGroup.add_argument('--K', help='Maximum number of mixture components [default = 2]', type=int, default=2)
    modelGroup.add_argument('--dbscan', help='Use DBSCAN rather than mixture model', default=False, action='store_true')
    modelGroup.add_argument('--D', help='Maximum number of clusters in DBSCAN fitting [default = 100]', type=int, default=100)
    modelGroup.add_argument('--min-cluster-prop', help='Minimum proportion of points in a cluster '
                                                        'in DBSCAN fitting [default = 0.0001]', type=float, default=0.0001)

    # model refinement
    refinementGroup = parser.add_argument_group('Refine model options')
    refinementGroup.add_argument('--pos-shift', help='Maximum amount to move the boundary away from origin [default = 0.2]',
            type=float, default=0.2)
    refinementGroup.add_argument('--neg-shift', help='Maximum amount to move the boundary towards the origin [default = 0.4]',
            type=float, default=0.4)
    refinementGroup.add_argument('--manual-start', help='A file containing information for a start point. '
            'See documentation for help.', default=None)
    refinementGroup.add_argument('--no-local', help='Do not perform the local optimization step (speed up on very large datasets)',
            default=False, action='store_true')

    # sequence querying
    queryingGroup = parser.add_argument_group('Database querying options')
    queryingGroup.add_argument('--model-dir', help='Directory containing model to use for assigning queries to clusters [default = reference database directory]', type = str)

    # model output
    faGroup = parser.add_argument_group('Further analysis options')
    faGroup.add_argument('--microreact', help='Generate output files for microreact visualisation', default=False, action='store_true')
    faGroup.add_argument('--cytoscape', help='Generate network output files for Cytoscape', default=False, action='store_true')
    faGroup.add_argument('--rapidnj', help='Path to rapidNJ binary to build NJ tree for Microreact', default=None)
    faGroup.add_argument('--perplexity', type=float, default = 20.0,
                         help='Perplexity used to calculate t-SNE projection (with --microreact) [default=20.0]')
    faGroup.add_argument('--info-csv',
                     help='Epidemiological information CSV formatted for microreact (with --microreact or --cytoscape)')

    # processing
    other = parser.add_argument_group('Other options')
    other.add_argument('--mash', default='mash', help='Location of mash executable')
    other.add_argument('--threads', default=1, type=int, help='Number of threads to use [default = 1]')
    other.add_argument('--no-stream', help='Use temporary files for mash dist interfacing. Reduce memory use/increase disk use for large datasets', default=False, action='store_true')

    other.add_argument('--version', action='version',
                       version='%(prog)s '+__version__)

    return parser.parse_args()

def main():

    args = get_options()

    # check mash is installed
    checkMashVersion(args.mash)

    # identify kmer properties
    minkmer = 9
    maxkmer = 29
    stepSize = 4
    if args.k_step is not None and args.k_step >= 2:
        stepSize = args.k_step
    if args.min_k is not None and args.min_k > minkmer:
        minkmer = int(args.min_k)
    if args.max_k is not None and args.max_k < maxkmer:
        maxkmer = int(args.max_k)
    if minkmer >= maxkmer or minkmer < 9 or maxkmer > 31:
        sys.stderr.write("Minimum kmer size " + minkmer + " must be smaller than maximum kmer size " +
                         maxkmer + "; range must be between 9 and 31\n")
        sys.exit(1)

    kmers = np.arange(minkmer, maxkmer + 1, stepSize)

    # define sketch sizes, store in hash in case on day
    # different kmers get different hash sizes
    sketch_sizes = {}
    if args.sketch_size >= 100 and args.sketch_size <= 10**6:
        for k in kmers:
            sketch_sizes[k] = args.sketch_size
    else:
        sys.stderr.write("Sketch size should be between 100 and 10^6\n")
        sys.exit(1)

    # check on file paths and whether files will be overwritten
    # confusing to overwrite command line parameter
    #if not args.full_db and not (args.create_db or args.easy_run or args.assign_query):
    #    args.overwrite = True
    if args.output is not None and args.output.endswith('/'):
        args.output = args.output[:-1]
    if args.ref_db is not None and args.ref_db.endswith('/'):
        args.ref_db = args.ref_db[:-1]

    # run according to mode
    sys.stderr.write("PopPUNK (POPulation Partitioning Using Nucleotide Kmers)\n")

    # database construction
    if args.create_db or args.easy_run:
        if args.create_db:
            sys.stderr.write("Mode: Building new database from input sequences\n")
        elif args.easy_run:
            sys.stderr.write("Mode: Creating clusters from assemblies (create_db & fit_model)\n")
        if args.r_files is not None:
            createDatabaseDir(args.output, kmers)
            constructDatabase(args.r_files, kmers, sketch_sizes, args.output, args.threads, args.mash, args.overwrite)
            refList, queryList, distMat = queryDatabase(args.r_files, kmers, args.output, args.output, True,
                    args.plot_fit, args.no_stream, args.mash, args.threads)

            dists_out = args.output + "/" + args.output + ".dists"
            storePickle(refList, queryList, True, distMat, dists_out)
        else:
            sys.stderr.write("Need to provide a list of reference files with --r-files")
            sys.exit(1)

    # model fit and network construction
    # refine model also needs to run all model steps
    if args.fit_model or args.refine_model or args.easy_run:
        # Set up saved data from first step, if easy_run mode
        if args.easy_run:
            distances = dists_out
            ref_db = args.output
        else:
            if args.fit_model:
                sys.stderr.write("Mode: Fitting model to reference database\n\n")
            else:
                sys.stderr.write("Mode: Refining model fit using network properties\n\n")

            if args.distances is not None and args.ref_db is not None:
                distances = args.distances
                ref_db = args.ref_db
            else:
                sys.stderr.write("Need to provide an input set of distances with --distances "
                                 "and reference database directory with --ref-db\n\n")
                sys.exit(1)

        refList, queryList, self, distMat = readPickle(distances)
        if not self:
            sys.stderr.write("Model fit should be to a reference db made with --create-db\n")
            sys.exit(1)

        # Run refinement
        if args.refine_model:
            old_model = loadClusterFit(args.ref_db + "/" + args.ref_db + '_fit.pkl',
                                       args.ref_db + "/" + args.ref_db + '_fit.npz')
            if old_model.type == 'refine':
                sys.stderr.write("Model needs to be from --fit-model not --refine-model\n")
                sys.exit(1)

            model = RefineFit(args.output)
            assignments = model.fit(distMat, refList, old_model, args.pos_shift, args.neg_shift, args.manual_start,
                    args.no_local, args.threads)
            model.plot(distMat)
        # Run DBSCAN model
        elif args.dbscan:
            model = DBSCANFit(args.output)
            assignments = model.fit(distMat, args.D, args.min_cluster_prop)
            model.plot()
        # Run Gaussian model
        else:
            model = BGMMFit(args.output)
            assignments = model.fit(distMat, args.K)
            model.plot(distMat, assignments)

        model.save()
        genomeNetwork = constructNetwork(refList, queryList, assignments, model.within_label)

        isolateClustering = printClusters(genomeNetwork, args.output)
        # generate outputs for microreact if asked
        if args.microreact:
            outputsForMicroreact(refList, distMat, isolateClustering, args.perplexity,
                    args.output, args.info_csv, args.rapidnj, args.overwrite)
        # generate outputs for cytoscape if asked
        if args.cytoscape:
            outputsForCytoscape(genomeNetwork, isolateClustering, args.output, args.info_csv)
        # extract limited references from clique by default
        if not args.full_db:
            newReferencesNames, newReferencesFile = extractReferences(genomeNetwork, args.output)
            genomeNetwork.remove_nodes_from(set(refList).difference(newReferencesNames))

            # Read previous database
            kmers, sketch_sizes = readMashDBParams(ref_db, kmers, sketch_sizes)
            constructDatabase(newReferencesFile, kmers, sketch_sizes, args.output, args.threads, args.mash, True) # overwrite old db

        printQueryOutput(refList, queryList, distMat, args.output, self)
        nx.write_gpickle(genomeNetwork, args.output + "/" + args.output + '_graph.gpickle')

    elif args.assign_query:
        if args.ref_db is not None and args.q_files is not None:
            sys.stderr.write("Mode: Assigning clusters of query sequences\n\n")
            self = False
            if args.ref_db == args.output:
                sys.stderr.write("--output and --ref-db must be different to "
                                 "prevent overwrite.\n")

            # Find distances to reference db
            kmers, sketch_sizes = readMashDBParams(args.ref_db, kmers, sketch_sizes)

            createDatabaseDir(args.output, kmers)
            constructDatabase(args.q_files, kmers, sketch_sizes, args.output, args.threads, args.mash, args.overwrite)
            refList, queryList, distMat = queryDatabase(args.q_files, kmers, args.ref_db, args.output, False, args.plot_fit,
                                                        args.no_stream, args.mash, args.threads)
            printQueryOutput(refList, queryList, distMat, args.output, self)

            # Assign these distances as within or between
            model_prefix = args.ref_db
            if args.model_dir is not None:
                model_prefix = args.model_dir
            model = loadClusterFit(model_prefix + "/" + model_prefix + '_fit.pkl',
                                   model_prefix + "/" + model_prefix + '_fit.npz')
            queryAssignments = model.assign(distMat)
            genomeNetwork = nx.read_gpickle(model_prefix + "/" + model_prefix + '_graph.gpickle')

            # Assign clustering by adding to network
            addQueryToNetwork(refList, queryList, args.q_files, genomeNetwork, kmers, queryAssignments,
                    model, args.output, args.no_stream, args.update_db, args.threads, args.mash)
            isolateClustering = printClusters(genomeNetwork, args.output,
                    model_prefix + "/" + model_prefix + '_clusters.csv', False)

            # update_db like no full_db
            if args.update_db:
                sys.stderr.write("Updating reference database to " + args.output + "\n")

                # Update the network + ref list
                newRepresentativesNames, newRepresentativesFile = extractReferences(genomeNetwork, args.output)
                genomeNetwork.remove_nodes_from(set(genomeNetwork.nodes).difference(newRepresentativesNames))
                nx.write_gpickle(genomeNetwork, args.output + "/" + args.output + '_graph.gpickle')

                # Update the mash database
                newQueries = set(newRepresentativesNames).intersection(queryList)
                tmpRefFile = writeTmpFile(newQueries)
                constructDatabase(tmpRefFile, kmers, sketch_sizes, args.output, args.threads, args.mash, True) # overwrite old db
                joinDBs(args.output, args.ref_db, kmers)
                os.remove(tmpRefFile)
        else:
            sys.stderr.write("Need to provide both a reference database with --ref-db and "
                             "query list with --q-files\n")
            sys.exit(1)


    sys.stderr.write("\nDone\n")

if __name__ == '__main__':
    main()

    sys.exit(0)
