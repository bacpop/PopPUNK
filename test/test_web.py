import os
import sys 
import subprocess

# Insert PopPUNK module into PATH
sys.path.insert(0, '..')
from PopPUNK.assign import assign_query
from PopPUNK.web import default_options, summarise_clusters, get_colours, api, graphml_to_json
from PopPUNK.utils import setupDBFuncs
from PopPUNK.visualise import generate_visualisations

# Copy and move args and sketch files into example dirs
subprocess.run("cp web_args.txt args.txt", shell=True, check=True)
subprocess.run("cp example_viz/example_viz_core_NJ.nwk example_viz/example_viz.nwk", shell=True, check=True)
subprocess.run("mv args.txt example_db", shell=True, check=True)

# Test the output of the PopPUNk-web upload route for incorrect data types
sys.stderr.write('\nTesting PopPUNK-web upload route\n')
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
                            args.assign.update_db,
                            args.assign.write_references,
                            args.assign.distances,
                            args.assign.threads,
                            args.assign.overwrite,
                            args.assign.plot_fit,
                            args.assign.graph_weights,
                            args.assign.max_a_dist,
                            args.assign.model_dir,
                            args.assign.strand_preserved,
                            args.assign.previous_clustering,
                            args.assign.external_clustering,
                            args.assign.core_only,
                            args.assign.accessory_only,
                            args.assign.web,
                            sketch)
query, query_prevalence, clusters, prevalences, alias_dict = \
    summarise_clusters(outdir, species, species_db)
colours = get_colours(query, clusters)
url = api(query, "example_viz")
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
sys.stderr.write('PopPUNK-web upload route test successful\n')

# Test the output of the PopPUNk-web network route for incorrect data types
sys.stderr.write('\nTesting PopPUNK-web network route\n')
with open(outdir + "/include.txt", "r") as i:
    to_include = (i.read()).split("\n")
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
                        outdir,
                        args.visualise.previous_query_clustering,
                        args.visualise.info_csv,
                        args.visualise.rapidnj,
                        args.visualise.tree,
                        args.visualise.mst_distances,
                        args.visualise.overwrite,
                        args.visualise.core_only,
                        args.visualise.accessory_only)
networkJson = graphml_to_json(outdir)
if len(to_include) >= 3:
    with open(os.path.join(outdir, os.path.basename(outdir) + "_core_NJ.nwk"), "r") as p:
        phylogeny = p.read()
else:
    phylogeny = "A tree cannot be built with fewer than 3 samples."
if not isinstance(networkJson, dict):
    raise TypeError('"networkJson" datatype is incorrect, should be dict.\n')
if not isinstance(phylogeny, str):
    raise TypeError('"phylogeny" datatype is incorrect, should be str.\n')
sys.stderr.write('\nAPI tests complete\n')
