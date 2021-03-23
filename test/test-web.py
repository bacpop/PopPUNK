import os
import sys
import subprocess
from shutil import copyfile

# testing without install
#sys.path.insert(0, '..')
from PopPUNK.assign import assign_query
from PopPUNK.web import default_options, summarise_clusters, get_colours, api, graphml_to_json
from PopPUNK.utils import setupDBFuncs
from PopPUNK.visualise import generate_visualisations

def main():
    # Copy and move args and sketch files into example dirs
    copyfile("web_args.txt", "example_db/args.txt")
    copyfile("example_viz/example_viz_core_NJ.nwk", "example_viz/example_viz.nwk")

    # Test the output of the PopPUNk-web upload route for incorrect data types
    sys.stderr.write('\nTesting assign for PopPUNK-web\n')
    with open("json_sketch.txt", "r") as s:
        sketch = s.read()
    species = "Listeria monocytogenes"
    species_db = "example_db"
    outdir = "example_api"
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    args = default_options(species_db)
    qc_dict = {'run_qc': False }
    dbFuncs = setupDBFuncs(args.assign, args.assign.min_kmer_count, qc_dict)
    ClusterResult = assign_query(dbFuncs,
                                args.assign.ref_db,
                                args.assign.q_files,
                                outdir,
                                qc_dict,
                                args.assign.update_db,
                                args.assign.write_references,
                                args.assign.distances,
                                args.assign.threads,
                                args.assign.overwrite,
                                args.assign.plot_fit,
                                args.assign.graph_weights,
                                args.assign.max_a_dist,
                                args.assign.max_pi_dist,
                                args.assign.type_isolate,
                                args.assign.model_dir,
                                args.assign.strand_preserved,
                                args.assign.previous_clustering,
                                args.assign.external_clustering,
                                args.assign.core_only,
                                args.assign.accessory_only,
                                args.assign.gpu_sketch,
                                args.assign.gpu_dist,
                                args.assign.gpu_graph,
                                args.assign.deviceid,
                                args.assign.web,
                                sketch,
                                args.assign.save_partial_query_graph)
    query, query_prevalence, clusters, prevalences, alias_dict, to_include = \
        summarise_clusters(outdir, species, species_db)
    colours = get_colours(query, clusters)
    url = api(query, "example_viz")
    sys.stderr.write('PopPUNK-web assign test successful\n')

    # Test generate_visualisations() for PopPUNK-web
    sys.stderr.write('\nTesting visualisations for PopPUNK-web\n')
    if len(to_include) < 3:
        args.visualise.microreact = False
    generate_visualisations(outdir,
                            species_db,
                            None,
                            args.visualise.threads,
                            outdir,
                            args.visualise.gpu_dist,
                            args.visualise.deviceid,
                            args.visualise.external_clustering,
                            args.visualise.microreact,
                            args.visualise.phandango,
                            args.visualise.grapetree,
                            args.visualise.cytoscape,
                            args.visualise.perplexity,
                            args.visualise.strand_preserved,
                            outdir + "/include.txt",
                            species_db,
                            species_db + "/" + os.path.basename(species_db) + "_clusters.csv",
                            args.visualise.previous_query_clustering,
                            outdir + "/" + os.path.basename(outdir) + "_graph.gt",
                            args.visualise.gpu_graph,
                            args.visualise.info_csv,
                            args.visualise.rapidnj,
                            args.visualise.tree,
                            args.visualise.mst_distances,
                            args.visualise.overwrite,
                            args.visualise.core_only,
                            args.visualise.accessory_only,
                            args.visualise.display_cluster,
                            web=True)
    networkJson = graphml_to_json(outdir)
    if len(to_include) >= 3:
        with open(os.path.join(outdir, os.path.basename(outdir) + "_core_NJ.nwk"), "r") as p:
            phylogeny = p.read()
    else:
        phylogeny = "A tree cannot be built with fewer than 3 samples."

    # ensure web api outputs are of the correct type
    if not isinstance(species, str):
        raise TypeError('"Species" datatype is incorrect, should be string.\n')
    if not (isinstance(query_prevalence, float) or isinstance(query_prevalence, int)):
        raise TypeError('"query_prevalence" datatype is incorrect, should be float/integer.\n')
    if not isinstance(query, str):
        raise TypeError('"query" datatype is incorrect, should be string.\n')
    if not isinstance(clusters, list) and not isinstance(clusters[0], str):
        raise TypeError('"clusters" datatype is incorrect, should be list of strings.\n')
    if not isinstance(prevalences, list) and not (isinstance(prevalences[0], float) or isinstance(prevalences[0], int)):
        raise TypeError('"prevalences" datatype is incorrect, should be list of floats/integers.\n')
    if not isinstance(colours, list) and not isinstance(colours[0], str):
        raise TypeError('"colours" datatype is incorrect, should be list of strings.\n')
    if not isinstance(url, str):
        raise TypeError('"url" datatype is incorrect, should be string.\n')
    if not isinstance(alias_dict, dict):
        raise TypeError('"alias_dict" datatype is incorrect, should be dictionary.\n')
    if not isinstance(outdir, str):
        raise TypeError('"outdir" datatype is incorrect, should be string.\n')
    if not isinstance(networkJson, dict):
        raise TypeError('"networkJson" datatype is incorrect, should be dict.\n')
    if not isinstance(phylogeny, str):
        raise TypeError('"phylogeny" datatype is incorrect, should be str.\n')

    sys.stderr.write('\nAPI tests complete\n')

if __name__ == "__main__":
    main()