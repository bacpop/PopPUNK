#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018-2020 John Lees and Nick Croucher

# universal
import os
import sys
# additional
import numpy as np

# required from v2.1.1 onwards (no mash support)
import pp_sketchlib

# import poppunk package
from .__init__ import __version__

from .models import loadClusterFit

from .network import fetchNetwork

from .plot import outputsForMicroreact
from .plot import outputsForCytoscape
from .plot import outputsForPhandango
from .plot import outputsForGrapetree
from .plot import writeClusterCsv

from .prune_db import prune_distance_matrix

from .sketchlib import readDBParams
from .sketchlib import getKmersFromReferenceDatabase
from .sketchlib import addRandom

from .utils import readPickle
from .utils import setGtThreads
from .utils import update_distance_matrices
from .utils import readIsolateTypeFromCsv
from .utils import joinClusterDicts

#******************************#
#*                            *#
#* Command line parsing       *#
#*                            *#
#******************************#
def get_options():

    import argparse

    parser = argparse.ArgumentParser(description='Create visualisations from PopPUNK results',
                                     prog='poppunk_visualise')

    # input options
    iGroup = parser.add_argument_group('Input files')
    iGroup.add_argument('--ref-db',
                        type = str,
                        help='Location of built reference database',
                        required=True)
    iGroup.add_argument('--query-db',
                        type=str,
                        help='Location of query database, if distances '
                             'are from ref-query')
    iGroup.add_argument('--distances',
                        help='Prefix of input pickle of pre-calculated distances')
    iGroup.add_argument('--include-files',
                         help='File with list of sequences to include in visualisation. '
                              'Default is to use all sequences in database.',
                         default=None)
    iGroup.add_argument('--external-clustering',
                        help='File with cluster definitions or other labels '
                             'generated with any other method.',
                        default=None)
    iGroup.add_argument('--model-dir',
                        help='Directory containing model to use for assigning queries '
                             'to clusters [default = reference database directory]',
                        type = str)
    iGroup.add_argument('--previous-clustering',
                        help='Directory containing previous cluster definitions '
                             'and network [default = use that in the directory '
                             'containing the model]',
                        type = str)
    iGroup.add_argument('--previous-query-clustering',
                        help='Directory containing previous cluster definitions '
                             'from poppunk_assign [default = use that in the directory '
                             'containing the model]',
                        type = str)

    # output options
    oGroup = parser.add_argument_group('Output options')
    oGroup.add_argument('--output',
                        required=True,
                        help='Prefix for output files (required)')
    oGroup.add_argument('--overwrite',
                        help='Overwrite any existing visualisation files',
                        default=False,
                        action='store_true')

    # query options
    queryingGroup = parser.add_argument_group('Database querying options')
    queryingGroup.add_argument('--core-only', help='(with a \'refine\' model) '
                                                   'Use a core-distance only model for assigning queries '
                                                   '[default = False]', default=False, action='store_true')
    queryingGroup.add_argument('--accessory-only', help='(with a \'refine\' or \'lineage\' model) '
                                                        'Use an accessory-distance only model for assigning queries '
                                                        '[default = False]', default=False, action='store_true')

    # plot output
    faGroup = parser.add_argument_group('Visualisation options')
    faGroup.add_argument('--microreact', help='Generate output files for microreact visualisation', default=False, action='store_true')
    faGroup.add_argument('--cytoscape', help='Generate network output files for Cytoscape', default=False, action='store_true')
    faGroup.add_argument('--phandango', help='Generate phylogeny and TSV for Phandango visualisation', default=False, action='store_true')
    faGroup.add_argument('--grapetree', help='Generate phylogeny and CSV for grapetree visualisation', default=False, action='store_true')
    faGroup.add_argument('--rapidnj', help='Path to rapidNJ binary to build NJ tree for Microreact', default='rapidnj')
    faGroup.add_argument('--perplexity',
                         type=float, default = 20.0,
                         help='Perplexity used to calculate t-SNE projection (with --microreact) [default=20.0]')
    faGroup.add_argument('--info-csv',
                         help='Epidemiological information CSV formatted for microreact (can be used with other outputs)')

    other = parser.add_argument_group('Other options')
    other.add_argument('--threads', default=1, type=int, help='Number of threads to use [default = 1]')
    other.add_argument('--gpu-dist', default=False, action='store_true', help='Use a GPU when calculating distances [default = False]')
    other.add_argument('--deviceid', default=0, type=int, help='CUDA device ID, if using GPU [default = 0]')
    other.add_argument('--strand-preserved', default=False, action='store_true',
                       help='If distances being calculated, treat strand as known when calculating random '
                            'match chances [default = False]')

    other.add_argument('--version', action='version',
                       version='%(prog)s '+__version__)


    # combine
    args = parser.parse_args()

    # ensure directories do not have trailing forward slash
    for arg in [args.ref_db, args.model_dir, args.output, args.external_clustering, args.previous_clustering]:
        if arg is not None:
            arg = arg.rstrip('\\')

    if args.rapidnj == "":
        args.rapidnj = None

    return args

def generate_visualisations(query_db,
                            ref_db,
                            distances,
                            threads,
                            output,
                            gpu_dist,
                            deviceid,
                            external_clustering,
                            microreact,
                            phandango,
                            grapetree,
                            cytoscape,
                            perplexity,
                            strand_preserved,
                            include_files,
                            model_dir,
                            previous_clustering,
                            previous_query_clustering,
                            info_csv,
                            rapidnj,
                            overwrite,
                            core_only,
                            accessory_only,
                            web):

    # Check on parallelisation of graph-tools
    setGtThreads(threads)

    sys.stderr.write("PopPUNK: visualise\n")
    if not (microreact or phandango or grapetree or cytoscape):
        sys.stderr.write("Must specify at least one type of visualisation to output\n")
        sys.exit(1)

    # make directory for new output files
    if not os.path.isdir(output):
        try:
            os.makedirs(output)
        except OSError:
            sys.stderr.write("Cannot create output directory\n")
            sys.exit(1)

    # Load original distances
    if distances is None:
        if query_db is None:
            distances = os.path.basename(ref_db) + "/" + ref_db + ".dists"
        else:
            distances = os.path.basename(query_db) + "/" + query_db + ".dists"
    else:
        distances = distances

    rlist, qlist, self, complete_distMat = readPickle(distances)
    if not self:
        qr_distMat = complete_distMat
    else:
        rr_distMat = complete_distMat

    # Fill in qq-distances if required
    if self == False:
        sys.stderr.write("Note: Distances in " + distances + " are from assign mode\n"
                         "Note: Distance will be extended to full all-vs-all distances\n"
                         "Note: Re-run poppunk_assign with --update-db to avoid this\n")
        ref_db = os.path.basename(ref_db) + "/" + ref_db
        rlist_original, qlist_original, self_ref, rr_distMat = readPickle(ref_db + ".dists")
        if not self_ref:
            sys.stderr.write("Distances in " + ref_db + " not self all-vs-all either\n")
            sys.exit(1)
        kmers, sketch_sizes, codon_phased = readDBParams(query_db)
        addRandom(query_db, qlist, kmers,
                  strand_preserved = strand_preserved, threads = threads)
        query_db = os.path.basename(query_db) + "/" + query_db
        qq_distMat = pp_sketchlib.queryDatabase(query_db, query_db,
                                                qlist, qlist, kmers,
                                                True, False,
                                                threads,
                                                gpu_dist,
                                                deviceid)

        # If the assignment was run with references, qrDistMat will be incomplete
        if rlist != rlist_original:
            rlist = rlist_original
            qr_distMat = pp_sketchlib.queryDatabase(ref_db, query_db,
                                                    rlist, qlist, kmers,
                                                    True, False,
                                                    threads,
                                                    gpu_dist,
                                                    deviceid)

    else:
        qlist = None
        qr_distMat = None
        qq_distMat = None

    # Turn long form matrices into square form
    combined_seq, core_distMat, acc_distMat = \
            update_distance_matrices(rlist, rr_distMat,
                                     qlist, qr_distMat, qq_distMat,
                                     threads = threads)

    # extract subset of distances if requested
    if include_files is not None:
        viz_subset = set()
        with open(include_files, 'r') as assemblyFiles:
            for assembly in assemblyFiles:
                viz_subset.add(assembly.rstrip())
        if len(viz_subset.difference(combined_seq)) > 0:
            sys.stderr.write("--subset contains names not in --distances")

        # Only keep found rows
        row_slice = [True if name in viz_subset else False for name in combined_seq]
        combined_seq = [name for name in combined_seq if name in viz_subset]
        if qlist != None:
            qlist = list(viz_subset.intersection(qlist))
        core_distMat = core_distMat[np.ix_(row_slice, row_slice)]
        acc_distMat = acc_distMat[np.ix_(row_slice, row_slice)]
    else:
        viz_subset = None
    # Either use strain definitions, lineage assignments or external clustering
    isolateClustering = {}
    # Use external clustering if specified
    if external_clustering:
        cluster_file = external_clustering
        isolateClustering = readIsolateTypeFromCsv(cluster_file,
                                                   mode = 'external',
                                                   return_dict = True)

    # identify existing model and cluster files
    if model_dir is not None:
        model_prefix = model_dir
    else:
        model_prefix = ref_db
    try:
        model_file = os.path.basename(model_prefix) + "/" + os.path.basename(model_prefix)
        model = loadClusterFit(model_file + '_fit.pkl',
                               model_file + '_fit.npz')
    except FileNotFoundError:
        sys.stderr.write('Unable to locate previous model fit in ' + model_prefix + '\n')
        sys.exit(1)

    # Load previous clusters
    mode = "clusters"
    suffix = "_clusters.csv"
    if model.type == "lineage":
        mode = "lineages"
        suffix = "_lineages.csv"
    if model.indiv_fitted:
        sys.stderr.write("Note: Individual (core/accessory) fits found, but "
                         "visualisation only supports combined boundary fit\n")

    # Set directories of previous fit
    if previous_clustering is not None:
        prev_clustering = previous_clustering
    else:
        prev_clustering = os.path.dirname(model_file)
    cluster_file = prev_clustering + '/' + os.path.basename(prev_clustering) + suffix
    isolateClustering = readIsolateTypeFromCsv(cluster_file,
                                               mode = mode,
                                               return_dict = True)
    # Join clusters with query clusters if required
    if not self:
        if previous_query_clustering is not None:
            prev_query_clustering = previous_query_clustering + '/' + os.path.basename(previous_query_clustering)
        else:
            prev_query_clustering = query_db

        queryIsolateClustering = readIsolateTypeFromCsv(
                prev_query_clustering + suffix,
                mode = mode,
                return_dict = True)
        isolateClustering = joinClusterDicts(isolateClustering, queryIsolateClustering)

    # Now have all the objects needed to generate selected visualisations
    if microreact:
        sys.stderr.write("Writing microreact output\n")
        outputsForMicroreact(combined_seq, core_distMat, acc_distMat, isolateClustering, perplexity,
                             output, info_csv, rapidnj, queryList = qlist, overwrite = overwrite)
    if phandango:
        sys.stderr.write("Writing phandango output\n")
        outputsForPhandango(combined_seq, core_distMat, isolateClustering, output, info_csv, rapidnj,
                            queryList = qlist, overwrite = overwrite, microreact = microreact)
    if grapetree:
        sys.stderr.write("Writing grapetree output\n")
        outputsForGrapetree(combined_seq, core_distMat, isolateClustering, output, info_csv, rapidnj,
                            queryList = qlist, overwrite = overwrite, microreact = microreact)
    if cytoscape:
        sys.stderr.write("Writing cytoscape output\n")
        genomeNetwork, cluster_file = fetchNetwork(prev_clustering, model, rlist, False, core_only, accessory_only)
        outputsForCytoscape(genomeNetwork, isolateClustering, output, info_csv, viz_subset = viz_subset)
        if model.type == 'lineage':
            sys.stderr.write("Note: Only support for output of cytoscape graph at lowest rank\n")

    sys.stderr.write("\nDone\n")

def main():
    """Main function. Parses cmd line args and runs in the specified mode.
    """
    args = get_options()

    generate_visualisations(args.query_db,
                            args.ref_db,
                            args.distances,
                            args.threads,
                            args.output,
                            args.gpu_dist,
                            args.deviceid,
                            args.external_clustering,
                            args.microreact,
                            args.phandango,
                            args.grapetree,
                            args.cytoscape,
                            args.perplexity,
                            args.strand_preserved,
                            args.include_files,
                            args.model_dir,
                            args.previous_clustering,
                            args.previous_query_clustering,
                            args.info_csv,
                            args.rapidnj,
                            args.overwrite,
                            args.core_only,
                            args.accessory_only,
                            web = False)

if __name__ == '__main__':
    main()

    sys.exit(0)
