#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2018-2020 John Lees and Nick Croucher

# universal
import os
import sys
# additional
import numpy as np
from tempfile import mkstemp

# import poppunk package
from .__init__ import __version__

from .sketchlib import removeFromDB

from .utils import storePickle
from .utils import readPickle
from .utils import iterDistRows

def prune_distance_matrix(refList, remove_seqs_in, distMat, output):
    """Rebuild distance matrix following selection of panel of references

    Args:
        refList (list)
            List of sequences used to generate distance matrix
        remove_seqs_in (list)
            List of sequences to be omitted
        distMat (numpy.array)
            nx2 matrix of core distances (column 0) and accessory
            distances (column 1)
        output (string)
            Prefix for new distance output files
    Returns:
        newRefList (list)
            List of sequences retained in distance matrix
        newDistMat (numpy.array)
            Updated version of distMat
    """
    # Find list items to remove
    remove_seqs_list = []
    removal_indices = []
    for to_remove in remove_seqs_in:
        found = False
        for idx, item in enumerate(refList):
            if item == to_remove:
                removal_indices.append(idx)
                remove_seqs_list.append(item)
                found = True
                break
        if not found:
            sys.stderr.write("Couldn't find " + to_remove + " in database\n")
    remove_seqs = frozenset(remove_seqs_list)

    if len(remove_seqs) > 0:
        sys.stderr.write("Removing " + str(len(remove_seqs)) + " sequences\n")

        numNew = len(refList) - len(remove_seqs)
        newDistMat = np.zeros((int(0.5 * numNew * (numNew - 1)), 2), dtype=distMat.dtype)

        # Create new reference list iterator
        removal_indices.sort()
        removal_indices.reverse()
        next_remove = removal_indices.pop()
        newRefList = []
        for idx, seq in enumerate(refList):
            if idx == next_remove:
                if len(removal_indices) > 0:
                    next_remove = removal_indices.pop()
            else:
                newRefList.append(seq)

        newRowNames = iter(iterDistRows(newRefList, newRefList, self=True))

        # Copy over rows which don't have an excluded sequence
        newIdx = 0
        for distRow, (ref1, ref2) in zip(distMat, iterDistRows(refList, refList, self=True)):
            if ref1 not in remove_seqs and ref2 not in remove_seqs:
                (newRef1, newRef2) = next(newRowNames)
                if newRef1 == ref1 and newRef2 == ref2:
                    newDistMat[newIdx, :] = distRow
                    newIdx += 1
                else:
                    raise RuntimeError("Row name mismatch. Old: " + ref1 + "," + ref2 + "\n"
                                       "New: " + newRef1 + "," + newRef2 + "\n")

        storePickle(newRefList, newRefList, True, newDistMat, output)
    else:
        newRefList = refList
        newDistMat = distMat

    # return new distance matrix and sequence lists
    return newRefList, newDistMat

# command line parsing
def get_options():

    import argparse

    parser = argparse.ArgumentParser(description='Remove sequences from a PopPUNK database',
                                     prog='poppunk_prune')

    # input options
    iGroup = parser.add_argument_group('Input files')
    iGroup.add_argument('--remove', required=True, help='File listing sequences to remove (required)')
    iGroup.add_argument('--distances', required=True, help='Prefix of input pickle of pre-calculated distances (required)')
    iGroup.add_argument('--ref-db', help='Location of reference db, if resketching')

    # output options
    oGroup = parser.add_argument_group('Output options')
    oGroup.add_argument('--output', required=True, help='Prefix for output files (required)')

    # processing
    other = parser.add_argument_group('Other options')
    other.add_argument('--threads', default=1, type=int, help='Number of threads to use [default = 1]')

    other.add_argument('--version', action='version',
                       version='%(prog)s '+__version__)

    return parser.parse_args()

def main():

    # Check input ok
    args = get_options()

    # Create the output directory
    if not os.path.isdir(args.output):
        try:
            os.makedirs(args.output)
        except OSError:
            sys.stderr.write("Cannot create output directory\n")
            sys.exit(1)

    # Read in old distances
    refList, queryList, self, distMat = readPickle(args.distances,
                                                   enforce_self=True)

    # Read in seqs to remove
    remove_seqs_in = []
    with open(args.remove, 'r') as remove_file:
        for line in remove_file:
            remove_seqs_in.append(line.rstrip())

    # reduce distance matrix
    newRefList = prune_distance_matrix(refList,
                                       remove_seqs_in,
                                       distMat,
                                       args.output + "/" + os.path.basename(args.output))[0]

    if len(refList) != len(newRefList):
        if args.ref_db is not None:
            sys.stderr.write("Resketching sequences\n")
            removeFromDB(args.ref_db, args.output, remove_seqs_in)
            os.rename(args.output + "/" + os.path.basename(args.output) + ".tmp.h5",
                      args.output + "/" + os.path.basename(args.output) + ".h5")
    else:
        sys.stderr.write("No sequences to remove\n")

    sys.exit(0)

