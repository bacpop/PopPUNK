from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import os 
import json

from PopPUNK.web import get_colours, default_options, graphml_to_json, api, summarise_clusters
from PopPUNK.assign import assign_query
from PopPUNK.utils import setupDBFuncs
from PopPUNK.visualise import generate_visualisations

app = Flask(__name__)
CORS(app, expose_headers='Authorization')

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
            species_db = "GPS_v3_references"
        args = default_options(species_db)

        if not os.path.exists(args.assign.output):
            os.mkdir(args.assign.output)

        qc_dict = {'run_qc': False }
        dbFuncs = setupDBFuncs(args.assign, args.assign.min_kmer_count, qc_dict)

        ClusterResult =  assign_query(dbFuncs,
                                    args.assign.ref_db,
                                    args.assign.q_files,
                                    args.assign.output,
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
                                    sketch_dict["sketch"])

        query, query_prevalence, clusters, prevalences = summarise_clusters(args.assign.output, args.assign.ref_db)
        colours = get_colours(query, clusters)
        url = api(query, args.assign.ref_db)

        response = {"species":species, 
                    "prev":str(query_prevalence) + '%', 
                    "query":query, 
                    "clusters":clusters, 
                    "prevalences":prevalences, 
                    "colours":colours, 
                    "microreactUrl":url}
        response = json.dumps(response)  
        
        return jsonify(response)

@app.route('/network', methods=['GET', 'POST'])
@cross_origin()
def postNetwork():
    """Recieve request to produce and post network information"""
    if not request.json:
        return "not a json post"
    if request.json:
        species_dict = request.json
        # determine database to use
        if species_dict["species"] == "S.pneumoniae":
            species_db = "GPS_v3_references"
        args = default_options(species_db)
        generate_visualisations(args.visualise.query_db,
                                args.visualise.ref_db,
                                args.visualise.distances,
                                args.visualise.threads,
                                args.visualise.output,
                                args.visualise.gpu_dist,
                                args.visualise.deviceid,
                                args.visualise.external_clustering,
                                args.visualise.microreact,
                                args.visualise.phandango,
                                args.visualise.grapetree,
                                args.visualise.cytoscape,
                                args.visualise.perplexity,
                                args.visualise.strand_preserved,
                                args.visualise.include_files,
                                args.visualise.model_dir,
                                args.visualise.previous_clustering,
                                args.visualise.previous_query_clustering,
                                args.visualise.info_csv,
                                args.visualise.rapidnj,
                                args.visualise.overwrite,
                                args.visualise.core_only,
                                args.visualise.accessory_only)
        networkJson = graphml_to_json(args.visualise.output)
        networkResponse = json.dumps({"network":networkJson})
        return jsonify(networkResponse)

if __name__ == "__main__":
    app.secret_key = os.urandom(24)
    app.run(debug=False,use_reloader=False)
    