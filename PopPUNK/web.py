import json
import h5py
import os 
import sys
import numpy as np

def sketch_to_hdf5(sketch, output):
    '''Convert JSON sketch to query hdf5 database'''

    kmers = []
    dists = []

    sketch_dict = json.loads(sketch)
    qNames = ["query.txt"]

    queryDB = h5py.File(os.path.join(output, output + '.h5'), 'w')
    sketches = queryDB.create_group("sketches")
    sketch_props = sketches.create_group(qNames[0])

    for key, value in sketch_dict.items():
        try:
            kmers.append(int(key))
            dists.append(np.array(value))
        except:
            if key == "bases":
                sketch_props.attrs['bases'] = value
            elif key == "bbits":
                sketch_props.attrs['bbits'] = value
            elif key == "length":
                sketch_props.attrs['length'] = value
            elif key == "missing_bases":
                sketch_props.attrs['missing_bases'] = value
            elif key == "sketchsize64":
                sketch_props.attrs['sketchsize64'] = value
            elif key == "version":
                sketch_props.attrs['version'] = value
            else:
                raise AttributeError(key + " Not recognised")
    
    sketch_props.attrs['kmers'] = kmers
    sketch_props.attrs['base_freq'] = [0.25, 0.25, 0.25, 0.25]

    for k_index in range(len(kmers)):
        k_spec = sketch_props.create_dataset(str(kmers[k_index]), data=dists[k_index], dtype='u8')
        k_spec.attrs['kmer-size'] = kmers[k_index]
    
    queryDB.close()

    return qNames