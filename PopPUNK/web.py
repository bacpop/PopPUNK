from types import SimpleNamespace
import h5py
import os
import re
import sys
import numpy as np
import pandas as pd
import graph_tool.all as gt
import requests
import networkx as nx
from networkx.readwrite import json_graph

import subprocess
import uuid
import glob
import atexit
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import json
from flask_apscheduler import APScheduler

from PopPUNK.assign import assign_query
from PopPUNK.utils import setupDBFuncs
from PopPUNK.visualise import generate_visualisations

# data locations
db_prefix = 'WebOutput'
db_location = '/home/poppunk-usr/' + os.getenv('POPPUNK_DBS_LOC', 'poppunk_dbs')

app = Flask(__name__, instance_relative_config=True)
app.config.update(
    TESTING=True,
    SCHEDULER_API_ENABLED=True,
    SECRET_KEY=os.environ.get('FLASK_SECRET_KEY')
)
CORS(app, expose_headers='Authorization')
scheduler = APScheduler()

@app.route('/')
def api_up():
    return 'PopPUNK.web running\n'

@app.route('/upload', methods=['POST'])
@cross_origin()
def sketchAssign():
    """Recieve sketch and respond with clustering information"""
    if not request.json:
        return "not a json post"
    if request.json:
        sketch_dict = request.json
        # determine database to use
        if sketch_dict["species"] == "S.pneumoniae":
            species = 'Streptococcus pneumoniae'
            species_db = db_location + "/GPS_v3_references"
        args = default_options(species_db)

        outdir = "/tmp/" + db_prefix + "_" + str(uuid.uuid4())
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        qc_dict = {'run_qc': False }
        dbFuncs = setupDBFuncs(args.assign, args.assign.min_kmer_count, qc_dict)

        # assign query to strain
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
                                    sketch_dict["sketch"],
                                    args.assign.save_partial_query_graph)
        query, query_prevalence, clusters, prevalences, alias_dict, to_include = \
            summarise_clusters(outdir, species, species_db)
        colours = get_colours(query, clusters)
        url = api(query, args.assign.ref_db)

        # generate visualisations from assign output
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
                                outdir + "/" + os.path.basename(outdir) + "_graph.gt",                                args.visualise.gpu_graph,
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

        # Convert outputs to JSON for post to site
        response = {"species":species,
                    "prev":str(query_prevalence) + '%',
                    "query":query,
                    "clusters":clusters,
                    "prevalences":prevalences,
                    "colours":colours,
                    "microreactUrl":url,
                    "aliases":alias_dict,
                    "network": networkJson,
                    "phylogeny": phylogeny}
        response = json.dumps(response)
        subprocess.run(["rm", "-rf", outdir]) # shutil issues on azure
        return jsonify(response)

def default_options(species_db):
    """Default options for WebAPI"""
    with open(os.path.join(species_db, "args.txt")) as a:
        args_json = a.read()
    args = json.loads(args_json, object_hook=lambda d: SimpleNamespace(**d))
    args.assign.ref_db = species_db
    args.assign.previous_clustering = species_db
    args.assign.model_dir = species_db
    args.assign.distances = os.path.join(species_db, os.path.basename(species_db) + ".dists")
    args.assign.q_files = os.path.join(args.assign.output, "queries.txt")
    return (args)

def get_colours(query, clusters):
    """Colour array for Plotly.js"""
    colours = []
    for clus in clusters:
        if str(clus) == str(query):
            colours.append('rgb(255,128,128)')
        else:
            colours.append('blue')
    return colours

def sketch_to_hdf5(sketch, output):
    """Convert JSON sketch to query hdf5 database"""
    kmers = []
    dists = []

    sketch_dict = json.loads(sketch)
    qNames = ["query"]
    queryDB = h5py.File(os.path.join(output, os.path.basename(output) + '.h5'), 'w')
    sketches = queryDB.create_group("sketches")
    sketch_props = sketches.create_group(qNames[0])

    for key, value in sketch_dict.items():
        try:
            kmers.append(int(key))
            dists.append(np.array(value, dtype='uint64'))
        except (TypeError, ValueError):
            if key == "version":
                sketches.attrs['sketch_version'] = value
            elif key == "codon_phased":
                sketches.attrs['codon_phased'] = value
            elif key == "densified":
                sketches.attrs['densified'] = value
            elif key == "bases":
                sketch_props.attrs['base_freq'] = value
            elif key == "bbits":
                sketch_props.attrs['bbits'] = value
            elif key == "length":
                sketch_props.attrs['length'] = value
            elif key == "missing_bases":
                sketch_props.attrs['missing_bases'] = value
            elif key == "sketchsize64":
                sketch_props.attrs['sketchsize64'] = value
            elif key == "species":
                pass
            else:
                sys.stderr.write(key + " not recognised")

    sketch_props.attrs['kmers'] = kmers
    for k_index in range(len(kmers)):
        k_spec = sketch_props.create_dataset(str(kmers[k_index]), data=dists[k_index], dtype='uint64')
        k_spec.attrs['kmer-size'] = kmers[k_index]
    queryDB.close()
    return qNames

def graphml_to_json(network_dir):
    """Converts full GraphML file to JSON subgraph"""
    labels = []
    nodes_list = []
    edges_list = []
    full_graph = gt.load_graph(os.path.join(network_dir, os.path.basename(network_dir) + "_cytoscape.graphml"))
    components = gt.label_components(full_graph)[0].a
    subgraph = gt.GraphView(full_graph, vfilt=(components == components[-1]))
    subgraph = gt.Graph(subgraph, prune=True)
    subgraph.save(os.path.join(network_dir,"subgraph.graphml"))

    G = nx.read_graphml(os.path.join(network_dir,"subgraph.graphml"))
    for value in G.nodes.values():
        labels.append(value['id'])
    data = json_graph.node_link_data(G)

    for k,v in data.items():
        if k == "nodes":
            for node in range(len(v)):
                node_attr = v[node]
                node_attr['label'] = labels[node]
                nodes_list.append({'data':node_attr})
        elif k == "links":
             for edge in range(len(v)):
                edge_attr = v[edge]
                edges_list.append({'data':edge_attr})

    network_dict = {'elements':{'nodes':nodes_list, 'edges':edges_list}}

    return network_dict

def highlight_cluster(query, cluster):
    """Colour assigned cluster in Microreact output"""
    if str(cluster) == str(query):
        colour = "red"
    else:
        colour = "blue"
    return colour

def api(query, ref_db):
    """Post cluster and tree information to microreact"""
    url = "https://microreact.org/api/project/"
    microreactDF = pd.read_csv(os.path.join(ref_db, os.path.basename(ref_db) + "_microreact_clusters.csv"))
    microreactDF["Cluster__autocolour"] = microreactDF['Cluster_Cluster__autocolour']
    microreactDF["Highlight_Query__colour"] = microreactDF.apply(lambda row: highlight_cluster(query, row["Cluster__autocolour"]), axis = 1)
    microreactDF = microreactDF.drop(columns=['Cluster_Cluster__autocolour'])
    clusters = microreactDF.to_csv()

    with open(os.path.join(ref_db, os.path.basename(ref_db) + ".nwk"), "r") as nwk:
        tree = nwk.read()

    description = "A tree representing all samples in the reference database, excluding the query sequence but highlighting its assigned cluster. The cluster assigned to the query is coloured red. If no clusters are highlighted red, query sequence was assigned to a new cluster."
    data = {"name":"PopPUNK-web","description": description,"data":clusters,"tree":tree}
    urlResponse = requests.post(url, data = data)
    microreactResponse = json.loads(urlResponse.text)
    for key, value in microreactResponse.items():
        if key == "url":
            url = value
    return url

def calc_prevalence(cluster, cluster_list, num_samples):
    """Cluster prevalences for Plotly.js"""
    clusterCount = cluster_list.count(cluster)
    prevalence = round(clusterCount / num_samples * 100, 2)
    return prevalence

def get_aliases(aliasDF, clusterLabels, species):
    if species == 'Streptococcus pneumoniae':
        GPS_name = 'unrecognised'
        for label in clusterLabels:
            if label in list(aliasDF['sample']):
                index = list(aliasDF['sample']).index(label)
                GPS_name = aliasDF['GPSC'][index]
        alias_dict = {"GPSC":str(GPS_name)}
    return alias_dict

def summarise_clusters(output, species, species_db):
    """Retreieve assigned query and all cluster prevalences.
    Write list of all isolates in cluster for tree subsetting"""
    totalDF = pd.read_csv(os.path.join(output, os.path.basename(output) + "_clusters.csv"))
    queryDF = totalDF.loc[totalDF['Taxon'] == "query"]
    queryDF = queryDF.reset_index(drop=True)
    query = str(queryDF["Cluster"][0])
    num_samples = len(totalDF["Taxon"])
    totalDF["Cluster"] = totalDF["Cluster"].astype(str)
    cluster_list = list(totalDF["Cluster"])

    totalDF["Prevalence"] = totalDF.apply(lambda row: calc_prevalence(row["Cluster"], cluster_list, num_samples), axis = 1)
    totalDF = totalDF.sort_values(by='Prevalence', ascending=False)

    uniquetotalDF = totalDF.drop_duplicates(subset=['Cluster'])
    clusters = list(uniquetotalDF['Cluster'])
    prevalences = list(uniquetotalDF["Prevalence"])
    query_prevalence = prevalences[clusters.index(query)]
    # write list of all isolates in cluster
    clusterDF = totalDF.loc[totalDF['Cluster'] == query]
    to_include = list(clusterDF['Taxon'])
    with open(os.path.join(output, "include.txt"), "w") as i:
        i.write("\n".join(to_include))
    # get aliases
    if os.path.isfile(os.path.join(species_db, "aliases.csv")):
        aliasDF = pd.read_csv(os.path.join(species_db, "aliases.csv"))
        alias_dict = get_aliases(aliasDF, list(clusterDF['Taxon']), species)
    else: 
        alias_dict = {"Aliases": "NA"}
    return query, query_prevalence, clusters, prevalences, alias_dict, to_include

@scheduler.task('interval', id='clean_tmp', hours=1, misfire_grace_time=900)
def clean_tmp():
    print("Cleaning up unused databases")
    for name in glob.glob("/tmp/" + db_prefix + "_*"):
        shutil.rmtree(name)

def main():
    # data cleanup
    scheduler.init_app(app)
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())
    app.run(debug=False,use_reloader=False)
