import json
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

class ArgsStructure:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def stringify(cluster):
    """Ensure Plotly.js interprets clusters as categorical variables"""
    cluster = '"' + str(cluster) + '"'
    return cluster

def default_options(species_db):
    """Default options for WebAPI"""
    with open(os.path.join(species_db, "args.txt")) as a:
        args_json = a.read()
    args_dict = json.loads(args_json)
    args = ArgsStructure(**args_dict)
    args.ref_db = species_db
    args.previous_clustering = species_db
    args.model_dir = species_db
    args.distances = os.path.join(species_db, species_db + ".dists")
    args.q_files = os.path.join(args.output, "queries.txt")
    return (args)

def get_colours(query, clusters):
    """Colour array for Plotly.js"""
    colours = []
    query = stringify(str(query))
    for clus in clusters:
        if not clus == query:
            colours.append('"blue"')
        else:
            colours.append('"rgb(255,128,128)"')
    return colours

def sketch_to_hdf5(sketch, output):
    """Convert JSON sketch to query hdf5 database"""
    kmers = []
    dists = []

    sketch_dict = json.loads(sketch)
    qNames = ["query"]
    queryDB = h5py.File(os.path.join(output, output + '.h5'), 'w')
    sketches = queryDB.create_group("sketches")
    sketch_props = sketches.create_group(qNames[0])

    for key, value in sketch_dict.items():
        try:
            kmers.append(int(key))
            dists.append(np.array(value))
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
            else:
                print(key + " not recognised")

    sketch_props.attrs['kmers'] = kmers
    for k_index in range(len(kmers)):
        k_spec = sketch_props.create_dataset(str(kmers[k_index]), data=dists[k_index], dtype='uint64')
        k_spec.attrs['kmer-size'] = kmers[k_index]
    queryDB.close()
    return qNames

def graphml_to_json(query, output):
    """Converts full GraphML file to JSON subgraph"""
    labels = []
    nodes_list = []
    edges_list = []
    G = gt.load_graph(os.path.join(output, "GPS_v3_references" + ".graphml"))
    components = gt.label_components(G)[0].a
    subgraph = gt.GraphView(G, vfilt=(components == int(query)))
    subgraph = gt.Graph(subgraph, prune=True)
    subgraph.save(os.path.join(output,"subgraph.graphml")) 

    G=nx.read_graphml(os.path.join(output,"subgraph.graphml"))
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

    with open(os.path.join(output,"subgraph.json"), 'w') as f:
        json.dump(network_dict, f, indent=4)

    return str(data).replace("'", "'").replace("False", "false")

def highlight_cluster(query, cluster):
    """Colour assigned cluster in Microreact output"""
    if str(cluster) == query:
        colour = "red"
    else:
        colour = "blue"
    return colour

def api(query, ref_db):
    """Post cluster and tree information to microreact"""
    url = "https://microreact.org/api/project/"
    microreactDF = pd.read_csv(os.path.join(ref_db, ref_db + "_microreact_clusters.csv"))
    microreactDF["Cluster"] = microreactDF['Cluster_Cluster__autocolour']
    microreactDF["CC__colour"] = microreactDF.apply(lambda row: highlight_cluster(query, row["Cluster_Cluster__autocolour"]), axis = 1)
    microreactDF = microreactDF.drop(columns=['Cluster_Cluster__autocolour'])
    clusters = microreactDF.to_csv()

    with open(os.path.join(ref_db, ref_db + ".nwk"), "r") as nwk:
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

def summarise_clusters(output, ref_db):
    """Retreieve assigned query and all cluster prevalences"""
    queryDF = pd.read_csv(os.path.join(output, output + "_clusters.csv"))
    queryDF = queryDF.loc[queryDF['Taxon'] == "query"]
    queryDF = queryDF.reset_index(drop=True)
    query = str(queryDF["Cluster"][0])
    queryDQ = stringify(query)
    clusterDF = pd.read_csv(os.path.join(ref_db, ref_db + "_clusters.csv"))
    clusterDF = clusterDF.append(queryDF)
    num_samples = len(clusterDF["Taxon"])
    clusterDF["Cluster"] = clusterDF["Cluster"].apply(stringify)
    cluster_list = list(clusterDF["Cluster"])

    clusterDF["Prevalence"] = clusterDF.apply(lambda row: calc_prevalence(row["Cluster"], cluster_list, num_samples), axis = 1)
    clusterDF = clusterDF.sort_values(by='Prevalence', ascending=False)

    uniqueclusterDF = clusterDF.drop_duplicates(subset=['Cluster'])
    clusters = list(uniqueclusterDF['Cluster'])
    prevalences = list(uniqueclusterDF["Prevalence"])
    query_prevalence = prevalences[clusters.index(queryDQ)]

    return query, query_prevalence, clusters, prevalences
    