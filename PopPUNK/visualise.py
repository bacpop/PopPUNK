#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018-2020 John Lees and Nick Croucher

# universal
import os
import sys
# additional
import numpy as np
from scipy import sparse

try:
    import cudf
    import rmm
    import cupy
    import cugraph
    from numba import cuda
    gpu_lib = True
except ImportError as e:
    gpu_lib = False

# required from v2.1.1 onwards (no mash support)
import pp_sketchlib

# import poppunk package
from .__init__ import __version__

#******************************#
#*                            *#
#* Command line parsing       *#
#*                            *#
#******************************#
def get_options():

    import argparse
    from .__main__ import accepted_weights_types

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
                        help='Prefix of input pickle of pre-calculated distances',
                        default=None)
    iGroup.add_argument('--rank-fit',
                        help='Location of rank fit, a sparse matrix (*_rank*_fit.npz)')
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
                        help='File containing previous cluster definitions '
                             'and network [default = use that in the directory '
                             'containing the model]',
                        type = str)
    iGroup.add_argument('--previous-query-clustering',
                        help='File containing previous cluster definitions '
                             'from poppunk_assign [default = use that in the directory '
                             'of the query database]',
                        type = str)
    iGroup.add_argument('--previous-mst',
                        help='File containing previous minimum spanning tree',
                        default=None,
                        type = str)
    iGroup.add_argument('--previous-distances',
                        help='Prefix of distance files used to generate the previous '
                        'minimum spanning tree',
                        default=None,
                        type = str)
    iGroup.add_argument('--network-file',
                        help='Specify a file to use for any graph visualisations',
                        type = str)
    iGroup.add_argument('--display-cluster',
                        help='Column of clustering CSV to use for plotting',
                        default=None)

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
    faGroup.add_argument('--tree', help='Type of tree to calculate (not for cytoscape) [default = nj]', type=str, default='nj',
        choices=['nj', 'mst', 'both', 'none'])
    faGroup.add_argument('--mst-distances', help='Distances used to calculate a minimum spanning tree [default = core]', type=str,
        default='core', choices=accepted_weights_types)
    faGroup.add_argument('--rapidnj', help='Path to rapidNJ binary to build NJ tree for Microreact', default='rapidnj')

    faGroup.add_argument('--api-key', help='API key for www.microreact.org, to directly create a visualisation', default=None)
    faGroup.add_argument('--perplexity',
                         type=float, default = 20.0,
                         help='Perplexity used to calculate mandrake projection (with --microreact) [default=20.0]')
    faGroup.add_argument('--info-csv',
                         help='Epidemiological information CSV formatted for microreact (can be used with other outputs)')

    other = parser.add_argument_group('Other options')
    other.add_argument('--threads', default=1, type=int, help='Number of threads to use [default = 1]')
    other.add_argument('--gpu-dist', default=False, action='store_true', help='Use a GPU when calculating distances [default = False]')
    other.add_argument('--gpu-graph', default=False, action='store_true', help='Use a GPU when calculating graphs [default = False]')
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
                            rank_fit,
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
                            previous_mst,
                            previous_distances,
                            network_file,
                            gpu_graph,
                            info_csv,
                            rapidnj,
                            api_key,
                            tree,
                            mst_distances,
                            overwrite,
                            display_cluster):

    from .models import loadClusterFit

    from .network import construct_network_from_assignments
    from .network import generate_minimum_spanning_tree
    from .network import load_network_file
    from .network import cugraph_to_graph_tool
    from .network import save_network
    from .network import sparse_mat_to_network

    from .plot import drawMST
    from .plot import outputsForMicroreact
    from .plot import outputsForCytoscape
    from .plot import outputsForPhandango
    from .plot import outputsForGrapetree
    from .plot import createMicroreact

    from .sketchlib import readDBParams
    from .sketchlib import addRandom

    from .sparse_mst import generate_mst_from_sparse_input

    from .trees import load_tree, generate_nj_tree, mst_to_phylogeny

    from .utils import isolateNameToLabel
    from .utils import readPickle
    from .utils import setGtThreads
    from .utils import update_distance_matrices
    from .utils import readIsolateTypeFromCsv
    from .utils import joinClusterDicts
    from .utils import read_rlist_from_distance_pickle

    #******************************#
    #*                            *#
    #* Initial checks and set up  *#
    #*                            *#
    #******************************#

    # Check on parallelisation of graph-tools
    setGtThreads(threads)

    sys.stderr.write("PopPUNK: visualise\n")
    if not (microreact or phandango or grapetree or cytoscape):
        sys.stderr.write("Must specify at least one type of visualisation to output\n")
        sys.exit(1)
    if cytoscape and not (microreact or phandango or grapetree):
        if rank_fit == None and not os.path.isfile(network_file):
            sys.stderr.write("For cytoscape, specify either a network file to visualise "
                             "with --network-file or a lineage model with --rank-fit\n")
            sys.exit(1)
        tree = 'none'

    # make directory for new output files
    if not os.path.isdir(output):
        try:
            os.makedirs(output)
        except OSError:
            sys.stderr.write("Cannot create output directory\n")
            sys.exit(1)

    #******************************#
    #*                            *#
    #* Process dense or sparse    *#
    #* distances                  *#
    #*                            *#
    #******************************#

    if distances is None:
        if query_db is None:
            distances = ref_db + "/" + os.path.basename(ref_db) + ".dists"
        else:
            distances = query_db + "/" + os.path.basename(query_db) + ".dists"
    else:
        distances = distances

    # Determine whether to use sparse distances
    combined_seq = None
    use_sparse = False
    use_dense = False
    if (tree == "mst" or tree == "both") and rank_fit is not None:
        # Set flag
        use_sparse = True
        # Read list of sequence names and sparse distance matrix
        rlist = read_rlist_from_distance_pickle(distances + '.pkl')
        sparse_mat = sparse.load_npz(rank_fit)
        combined_seq = rlist
        # Check previous distances have been supplied if building on a previous MST
        old_rlist = None
        if previous_distances is not None:
            old_rlist = read_rlist_from_distance_pickle(previous_distances + '.pkl')
        elif previous_mst is not None:
            sys.stderr.write('The prefix of the distance files used to create the previous MST'
                             ' is needed to use the network')
    if (tree == "nj" or tree == "both") or rank_fit == None:
        use_dense = True
        # Process dense distance matrix
        rlist, qlist, self, complete_distMat = readPickle(distances)
        if not self:
            qr_distMat = complete_distMat
            combined_seq = rlist + qlist
        else:
            rr_distMat = complete_distMat
            combined_seq = rlist

        # Fill in qq-distances if required
        if self == False:
            sys.stderr.write("Note: Distances in " + distances + " are from assign mode\n"
                             "Note: Distance will be extended to full all-vs-all distances\n"
                             "Note: Re-run poppunk_assign with --update-db to avoid this\n")
            ref_db_loc = ref_db + "/" + os.path.basename(ref_db)
            rlist_original, qlist_original, self_ref, rr_distMat = readPickle(ref_db_loc + ".dists")
            if not self_ref:
                sys.stderr.write("Distances in " + ref_db + " not self all-vs-all either\n")
                sys.exit(1)
            kmers, sketch_sizes, codon_phased = readDBParams(query_db)
            addRandom(query_db, qlist, kmers,
                      strand_preserved = strand_preserved, threads = threads)
            query_db_loc = query_db + "/" + os.path.basename(query_db)
            qq_distMat = pp_sketchlib.queryDatabase(ref_db_name=query_db_loc,
                                                    query_db_name=query_db_loc,
                                                    rList=qlist,
                                                    qList=qlist,
                                                    klist=kmers,
                                                    random_correct=True,
                                                    jaccard=False,
                                                    num_threads=threads,
                                                    use_gpu=gpu_dist,
                                                    device_id=deviceid)

            # If the assignment was run with references, qrDistMat will be incomplete
            if rlist != rlist_original:
                rlist = rlist_original
                qr_distMat = pp_sketchlib.queryDatabase(ref_db_name=ref_db_loc,
                                                        query_db_name=query_db_loc,
                                                        rList=rlist,
                                                        qList=qlist,
                                                        klist=kmers,
                                                        random_correct=True,
                                                        jaccard=False,
                                                        num_threads=threads,
                                                        use_gpu=gpu_dist,
                                                        device_id=deviceid)

        else:
            qlist = None
            qr_distMat = None
            qq_distMat = None

        # Turn long form matrices into square form
        combined_seq, core_distMat, acc_distMat = \
                update_distance_matrices(rlist, rr_distMat,
                                         qlist, qr_distMat, qq_distMat,
                                         threads = threads)

    #*******************************#
    #*                             *#
    #* Extract subset of sequences *#
    #*                             *#
    #*******************************#

    # extract subset of distances if requested
    all_seq = combined_seq
    if include_files is not None:
        viz_subset = set()
        with open(include_files, 'r') as assemblyFiles:
            for assembly in assemblyFiles:
                viz_subset.add(assembly.rstrip())
        if len(viz_subset.difference(combined_seq)) > 0:
            sys.stderr.write("--include-files contains names not in --distances\n")

        # Only keep found rows
        row_slice = [True if name in viz_subset else False for name in combined_seq]
        combined_seq = [name for name in combined_seq if name in viz_subset]
        if use_sparse:
            sparse_mat = sparse_mat[np.ix_(row_slice, row_slice)]
        if use_dense:
            if qlist != None:
                qlist = list(viz_subset.intersection(qlist))
            core_distMat = core_distMat[np.ix_(row_slice, row_slice)]
            acc_distMat = acc_distMat[np.ix_(row_slice, row_slice)]
    else:
        viz_subset = None

    #**********************************#
    #*                                *#
    #* Process clustering information *#
    #*                                *#
    #**********************************#

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
        model_file = os.path.join(model_prefix, os.path.basename(model_prefix))
        model = loadClusterFit(model_file + '_fit.pkl',
                               model_file + '_fit.npz')
        model.set_threads(threads)
    except FileNotFoundError:
        sys.stderr.write('Unable to locate previous model fit in ' + model_prefix + '\n')
        sys.exit(1)

    # Load previous clusters
    if previous_clustering is not None:
        prev_clustering = previous_clustering
        mode = "clusters"
        suffix = "_clusters.csv"
        if prev_clustering.endswith('_lineages.csv'):
            mode = "lineages"
            suffix = "_lineages.csv"
    else:
        # Identify type of clustering based on model
        mode = "clusters"
        suffix = "_clusters.csv"
        if model.type == "lineage":
            mode = "lineages"
            suffix = "_lineages.csv"
        prev_clustering = os.path.join(model_prefix, os.path.basename(model_prefix) + suffix)
    isolateClustering = readIsolateTypeFromCsv(prev_clustering,
                                               mode = mode,
                                               return_dict = True)

    # Add individual refinement clusters if they exist
    if model.indiv_fitted:
        for type, suffix in zip(['Core','Accessory'],['_core_clusters.csv','_accessory_clusters.csv']):
            indiv_clustering = os.path.join(model_prefix, os.path.basename(model_prefix) + suffix)
            if os.path.isfile(indiv_clustering):
                indiv_isolateClustering = readIsolateTypeFromCsv(indiv_clustering,
                                                                   mode = mode,
                                                                   return_dict = True)
                isolateClustering[type] = indiv_isolateClustering['Cluster']

    # Join clusters with query clusters if required
    if use_dense:
        if not self:
            if previous_query_clustering is not None:
                prev_query_clustering = previous_query_clustering
            else:
                prev_query_clustering = os.path.join(query_db, os.path.basename(query_db) + suffix)

            queryIsolateClustering = readIsolateTypeFromCsv(
                    prev_query_clustering,
                    mode = mode,
                    return_dict = True)
            isolateClustering = joinClusterDicts(isolateClustering, queryIsolateClustering)

    #*******************#
    #*                 *#
    #* Generate trees  *#
    #*                 *#
    #*******************#

    # Generate trees
    mst_tree = None
    mst_graph = None
    nj_tree = None
    if tree != 'none':
        if len(combined_seq) >= 3:
            # MST tree
            if tree == 'mst' or tree == 'both':
                existing_tree = None
                if not overwrite:
                    existing_tree = load_tree(output, "MST", distances=mst_distances)
                if existing_tree is None:
                    # Check selecting clustering type is in CSV
                    clustering_name = 'Cluster'
                    if display_cluster != None:
                        if display_cluster not in isolateClustering.keys():
                            clustering_name = list(isolateClustering.keys())[0]
                            sys.stderr.write('Unable to find clustering column ' + display_cluster + ' in file ' +
                                            prev_clustering + '; instead using ' + clustering_name + '\n')
                        else:
                            clustering_name = display_cluster
                    else:
                        clustering_name = list(isolateClustering.keys())[0]
                    if use_sparse:
                        G = generate_mst_from_sparse_input(sparse_mat,
                                                            rlist,
                                                            old_rlist = old_rlist,
                                                            previous_mst = previous_mst,
                                                            gpu_graph = gpu_graph)
                    elif use_dense:
                        # Get distance matrix
                        complete_distMat = \
                            np.hstack((pp_sketchlib.squareToLong(core_distMat, threads).reshape(-1, 1),
                                    pp_sketchlib.squareToLong(acc_distMat, threads).reshape(-1, 1)))
                        # Dense network may be slow
                        sys.stderr.write("Generating MST from dense distances (may be slow)\n")
                        G = construct_network_from_assignments(combined_seq,
                                                                combined_seq,
                                                                [0]*complete_distMat.shape[0],
                                                                within_label = 0,
                                                                distMat = complete_distMat,
                                                                weights_type = mst_distances,
                                                                use_gpu = gpu_graph,
                                                                summarise = False)
                        if gpu_graph:
                            G = cugraph.minimum_spanning_tree(G, weight='weights')
                    else:
                        sys.stderr.write("Need either sparse or dense distances matrix to construct MST\n")
                        exit(1)
                    mst_graph = generate_minimum_spanning_tree(G, gpu_graph)
                    del G
                    # save outputs
                    save_network(mst_graph,
                                    prefix = output,
                                    suffix = '_mst',
                                    use_graphml = False,
                                    use_gpu = gpu_graph)
                    if gpu_graph:
                        mst_graph = cugraph_to_graph_tool(mst_graph, isolateNameToLabel(combined_seq))
                    else:
                        vid = mst_graph.new_vertex_property('string',
                                                    vals = isolateNameToLabel(combined_seq))
                        mst_graph.vp.id = vid
                    mst_as_tree = mst_to_phylogeny(mst_graph,
                                                    isolateNameToLabel(combined_seq),
                                                    use_gpu = False)
                    mst_as_tree = mst_as_tree.replace("'","")
                    with open(os.path.join(output,os.path.basename(output) + '_mst.nwk'),'w') as tree_out:
                        tree_out.write(mst_as_tree)
                    drawMST(mst_graph, output, isolateClustering, clustering_name, overwrite)
                else:
                    mst_tree = existing_tree

            # Generate NJ tree
            if tree == 'nj' or tree == 'both':
                existing_tree = None
                if not overwrite:
                    existing_tree = load_tree(output, "NJ")
                if existing_tree is None:
                    nj_tree = generate_nj_tree(core_distMat,
                                                combined_seq,
                                                output,
                                                rapidnj,
                                                threads = threads)
                else:
                    nj_tree = existing_tree
        else:
            sys.stderr.write("Fewer than three sequences, not drawing trees\n")

    #****************#
    #*              *#
    #* Write output *#
    #*              *#
    #****************#

    # Now have all the objects needed to generate selected visualisations
    if microreact:
        sys.stderr.write("Writing microreact output\n")
        microreact_files = outputsForMicroreact(combined_seq,
                                                isolateClustering,
                                                nj_tree,
                                                mst_tree,
                                                acc_distMat,
                                                perplexity,
                                                output,
                                                info_csv,
                                                queryList=qlist,
                                                overwrite=overwrite,
                                                n_threads=threads,
                                                use_gpu=gpu_graph,
                                                device_id=deviceid)
        url = createMicroreact(output, microreact_files, api_key)
        if url != None:
            sys.stderr.write("Microreact: " + url + "\n")
        else:
            sys.stderr.write("Provide --api-key to create microreact automatically\n")

    if phandango:
        sys.stderr.write("Writing phandango output\n")
        outputsForPhandango(combined_seq,
                            isolateClustering,
                            nj_tree,
                            mst_tree,
                            output,
                            info_csv,
                            queryList = qlist,
                            overwrite = overwrite)

    if grapetree:
        sys.stderr.write("Writing grapetree output\n")
        outputsForGrapetree(combined_seq,
                            isolateClustering,
                            nj_tree,
                            mst_tree,
                            output,
                            info_csv,
                            queryList = qlist,
                            overwrite = overwrite)

    if cytoscape:
        sys.stderr.write("Writing cytoscape output\n")
        import graph_tool.all as gt
        if network_file is not None:
            genomeNetwork = load_network_file(network_file, use_gpu = gpu_graph)
            if gpu_graph:
                genomeNetwork = cugraph_to_graph_tool(genomeNetwork, isolateNameToLabel(all_seq))
            # Hard delete from network to remove samples (mask doesn't work neatly)
            if viz_subset is not None:
                remove_list = []
                for keep, idx in enumerate(row_slice):
                    if not keep:
                        remove_list.append(idx)
                genomeNetwork.remove_vertex(remove_list)
        elif rank_fit is not None:
            genomeNetwork = sparse_mat_to_network(sparse_mat, combined_seq, use_gpu = gpu_graph)
        else:
            sys.stderr.write('Cytoscape output requires a network file or lineage rank fit to be provided\n')
            sys.exit(1)
        outputsForCytoscape(genomeNetwork,
                            mst_graph,
                            combined_seq,
                            isolateClustering,
                            output,
                            info_csv)
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
                            args.rank_fit,
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
                            args.previous_mst,
                            args.previous_distances,
                            args.network_file,
                            args.gpu_graph,
                            args.info_csv,
                            args.rapidnj,
                            args.api_key,
                            args.tree,
                            args.mst_distances,
                            args.overwrite,
                            args.display_cluster)

if __name__ == '__main__':
    main()

    sys.exit(0)
