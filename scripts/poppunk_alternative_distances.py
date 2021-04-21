#! python

import os
import sys
import pickle
import numpy as np
import dendropy

#############
# Functions #
#############

def get_options():
    """Parse input arguments"""

    import argparse
    
    parser = argparse.ArgumentParser(description='Script for converting '
                ' Gubbins output to PopPUNK input',
                prog='make_lineage_assignment_input.py')

    parser.add_argument('--tree',
                        type = str,
                        help = 'Name of input newick file')

    parser.add_argument('--output',
                        type = str,
                        help = 'Prefix for output files')

    args = parser.parse_args()
    
    return args

def storePickle(rlist, qlist, self, X, pklName):
    """Saves core and accessory distances in a .npy file, names in a .pkl

    Called during ``--create-db``

    Args:
        rlist (list)
            List of reference sequence names (for :func:`~iterDistRows`)
        qlist (list)
            List of query sequence names (for :func:`~iterDistRows`)
        self (bool)
            Whether an all-vs-all self DB (for :func:`~iterDistRows`)
        X (numpy.array)
            n x 2 array of core and accessory distances
        pklName (str)
            Prefix for output files
    """
    with open(pklName + ".pkl", 'wb') as pickle_file:
        pickle.dump([rlist, qlist, self], pickle_file)
    np.save(pklName + ".npy", X)

def listDistInts(refSeqs, querySeqs, self=True):
    """Gets the ref and query ID for each row of the distance matrix

    Returns an iterable with ref and query ID pairs by row.

    Args:
        refSeqs (list)
            List of reference sequence names.
        querySeqs (list)
            List of query sequence names.
        self (bool)
            Whether a self-comparison, used when constructing a database.
            Requires refSeqs == querySeqs
            Default is True
    Returns:
        ref, query (str, str)
            Iterable of tuples with ref and query names for each distMat row.
    """
    num_ref = len(refSeqs)
    num_query = len(querySeqs)
    if self:
        if refSeqs != querySeqs:
            raise RuntimeError('refSeqs must equal querySeqs for db building (self = true)')
        for i in range(num_ref):
            for j in range(i + 1, num_ref):
                yield(j, i)
    else:
        comparisons = [(0,0)] * (len(refSeqs) * len(querySeqs))
        for i in range(num_query):
            for j in range(num_ref):
                yield(j, i)

#################
# Main function #
#################

def main():
    
    # Parse arguments from command line
    args = get_options()
    
    # Create directory
    if not os.path.isdir(args.output):
        try:
            os.makedirs(args.output)
        except OSError:
            sys.stderr.write("Cannot create output directory " + args.output + "\n")
            sys.exit(1)
    
    # Load tree
    tree = dendropy.Tree.get(path = args.tree, schema = 'newick')
    
    # Extract taxon names and save to pickle
    rlist = [taxon for taxon in tree.taxon_namespace]
    
    # Extract distances
    core_distances = list()
    pdc = tree.phylogenetic_distance_matrix()
    for seq1,seq2 in listDistInts(rlist, rlist, self = True):
        core_distances.append(pdc(rlist[seq1],rlist[seq2]))
    
    # Convert distances
    distMat = np.zeros((len(core_distances),2),
                        dtype = float)
    distMat[:,0] = np.asarray(core_distances)
    
    # Save output
    pkl_file_name = os.path.join(os.path.basename(args.output),
                                 os.path.basename(args.output))
    storePickle(rlist, rlist, True, distMat, pkl_file_name)

if __name__ == '__main__':
    main()
    sys.exit(0)
