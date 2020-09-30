#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018-2020 John Lees and Nick Croucher

import sys
import argparse
import pickle

# command line parsing
def get_options():

    parser = argparse.ArgumentParser(description='Fix pickle files with incorrect label order',
                                     prog='pickle_fix')

    # input options
    parser.add_argument('pickle', help='Input pickle (.dists.pkl)')
    parser.add_argument('output', help='Prefix for output files')

    return parser.parse_args()

# main code
if __name__ == "__main__":

    # Check input ok
    args = get_options()

    with open(args.pickle, 'rb') as pickled_names:
        rNames, qNames, self = pickle.load(pickled_names)

    rNames = sorted(rNames)
    qNames = sorted(qNames)

    with open(args.output + ".dists.pkl", 'wb') as pickle_fixed:
        pickle.dump([rNames, qNames, self], pickle_fixed)

    sys.exit(0)
