import json
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
    with open(os.path.join(output, "include.txt"), "w") as i:
        i.write("\n".join(list(clusterDF['Taxon'])))
    # get aliases
    aliasDF = pd.read_csv(os.path.join(species_db, "aliases.csv"))
    alias_dict = get_aliases(aliasDF, list(clusterDF['Taxon']), species)
    return query, query_prevalence, clusters, prevalences, alias_dict
