from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import os 

from PopPUNK.web import get_colours, default_options, graphml_to_json, highlight_cluster, api, calc_prevalence, stringify, summarise_clusters
from PopPUNK.assign import assign_query
from PopPUNK.utils import setupDBFuncs

app = Flask(__name__)
CORS(app, expose_headers='Authorization')

@app.route('/network', methods=['GET', 'POST'])
@cross_origin()
def postNetwork():
    """Recieve request to produce and post network information"""
    if not request.json:
        return "not a json post"
    if request.json:
        species_db = "GPS_v3_references"
        args = default_options(species_db)
        with open(os.path.join(args.output, "subgraph.json")) as n:
            networkJson = n.read()
        networkResponse = '{"network":' + networkJson + '}'
        return jsonify(networkResponse)

@app.route('/upload', methods=['POST'])
@cross_origin()
def sketchAssign():
    """Recieve sketch and respond with clustering information"""
    if not request.json:
        return "not a json post"
    if request.json:
        json_sketch = request.json

        species_db = "GPS_v3_references"

        args = default_options(species_db)
        if not os.path.exists(args.output):
            os.mkdir(args.output)

        qc_dict = {'run_qc': False }
        dbFuncs = setupDBFuncs(args, args.min_kmer_count, qc_dict)

        ClusterResult =  assign_query(dbFuncs,
                                    args.ref_db,
                                    args.q_files,
                                    args.output,
                                    args.update_db,
                                    args.write_references,
                                    args.distances,
                                    args.threads,
                                    args.overwrite,
                                    args.plot_fit,
                                    args.graph_weights,
                                    args.max_a_dist,
                                    args.model_dir,
                                    args.strand_preserved,
                                    args.previous_clustering,
                                    args.external_clustering,
                                    args.core_only,
                                    args.accessory_only,
                                    args.web,
                                    json_sketch)

        species = 'Streptococcus pneumoniae'

        query, query_prevalence, clusters, prevalences = summarise_clusters(args.output, args.ref_db)
        colours = str(get_colours(query, clusters)).replace("'", "")
        url = api(query, args.ref_db)
        networkJson = graphml_to_json(query, args.output)

        response = {"species":species, 
                    "prev":str(query_prevalence) + '%', 
                    "query":str(query), 
                    "clusters":str(clusters).replace("'", ""), 
                    "prevalences":prevalences, 
                    "colours":colours, 
                    "microreactUrl":url, 
                    "network":networkJson}

        return jsonify(str(response).replace("'", '"').replace('"[','[').replace(']"',']').replace('"{','{').replace('}"','}'))

if __name__ == "__main__":
    app.secret_key = os.urandom(24)
    app.run(debug=False,use_reloader=False)
    