#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018-2020 John Lees and Nick Croucher

import sys
import argparse
import h5py

# command line parsing
def get_options():

    parser = argparse.ArgumentParser(description='Get information about a PopPUNK database',
                                     prog='poppunk_db_info')

    # input options
    parser.add_argument('db', help='Database file (.h5)')
    parser.add_argument('--list-samples', action='store_true', default=False,
                        help='List every sample in the database')

    return parser.parse_args()

# main code
if __name__ == "__main__":

    # Check input ok
    args = get_options()

    ref_db = h5py.File(args.db, 'r')
    print("PopPUNK database:\t\t" + args.db)

    sketch_version = ref_db['sketches'].attrs['sketch_version']
    print("Sketch version:\t\t\t" + sketch_version)

    if 'random' in ref_db.keys():
        has_random = True
    else:
        has_random = False
    print("Contains random matches:\t" + str(has_random))

    num_samples = len(ref_db['sketches'].keys())
    print("Number of samples:\t\t" + str(num_samples))

    first_sample = list(ref_db['sketches'].keys())[0]
    kmer_size = ref_db['sketches/' + first_sample].attrs['kmers']
    print("K-mer sizes:\t\t\t" + ",".join([str(x) for x in kmer_size]))

    sketch_size = int(ref_db['sketches/' + first_sample].attrs['sketchsize64']) * 64
    print("Sketch size:\t\t\t" + str(sketch_size))

    if args.list_samples:
        print("\nSamples:")
        for sample_name in list(ref_db['sketches'].keys()):
            print("Name:\t" + sample_name)

            base_freq = ref_db['sketches/' + sample_name].attrs['base_freq']
            print("\tBase frequencies:\t" + ",".join([base + ':' + str(x) for base, x in zip(['A', 'C', 'G', 'T'], base_freq)]))

            length = ref_db['sketches/' + sample_name].attrs['length']
            print("\tLength:\t\t\t" + str(length))

            missing = ref_db['sketches/' + sample_name].attrs['missing_bases']
            print("\tMissing bases:\t\t" + str(missing))


    sys.exit(0)