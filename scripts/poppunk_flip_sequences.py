#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2020 John Lees and Nick Croucher

import os
import sys
import numpy as np
import argparse
import pp_sketchlib

#############
# functions #
#############

# command line parsing
def get_options():

    parser = argparse.ArgumentParser(description='Re-arrange a collection of sequences with mixed orientations', prog='flip_sequences')

    # input options
    parser.add_argument('--orientation-file', help='Tab separated file with name and file location'
                                         ' for a correctly-orientated reference sequence (required)',
                                         required=True, type = str)
    parser.add_argument('--unknown-file', help='Tab separated file with names and file locations'
                                        ' of sequences to be orientated (required)', required=True,
                                        type = str)
    parser.add_argument('--min-k', default = 13, type=int, help='Minimum kmer length [default = 9]')
    parser.add_argument('--max-k', default = 29, type=int, help='Maximum kmer length [default = 29]')
    parser.add_argument('--k-step', default = 4, type=int, help='K-mer step size [default = 4]')
    parser.add_argument('--sketch-size', default=10000, type=int, help='Kmer sketch size [default = 10000]')
    parser.add_argument('--threshold', default = 0.0, type=float, help='Amount by which reverse-complemented'
                                        ' accessory distance must exceed original to justify flipping')
    parser.add_argument('--prefix', help='Prefix to use for intermediate files [default = reorder]',
                                    default = 'reorder', type = str)
    parser.add_argument('--update-sequences', help='Replace original sequences with reverse complements '
                                    ' where appropriate', default = False, action = 'store_true')
    parser.add_argument('--keep', help='Keep intermediate files [default = false]',
                                    default = False, action = 'store_true')
    parser.add_argument('--threads', default = 1, type=int, help='Number of CPUs to use [default = 1]')
    parser.add_argument('--use-gpu', default=False, action='store_true', help='Use a GPU when calculating distances [default = False]')
    parser.add_argument('--deviceid', default=0, type=int, help='CUDA device ID, if using GPU [default = 0]')

    return parser.parse_args()

def readRfile(rFile):
    """Reads in files for sketching. Names and sequence, tab separated

    Args:
        rFile (str)
            File with locations of assembly files to be sketched

    Returns:
        names (list)
            Array of sequence names
        sequences (list of lists)
            Array of sequence files
    """
    names = []
    sequences = []
    with open(rFile, 'r') as refFile:
        for refLine in refFile:
            rFields = refLine.rstrip().split("\t")
            if len(rFields) < 2:
                sys.stderr.write("Input reference list is misformatted\n"
                                 "Must contain sample name and file, tab separated\n")
                sys.exit(1)

            names.append(rFields[0])
            sample_files = []
            for sequence in rFields[1:]:
                sample_files.append(sequence)

            # Take first of sequence list if using mash
            sequences.append(sample_files)

    if len(set(names)) != len(names):
        sys.stderr.write("Input contains duplicate names! All names must be unique\n")
        sys.exit(1)

    return (names, sequences)

# reverse complement sequence
complement = {'A': 'T',
            'C': 'G',
            'G': 'C',
            'T': 'A',
            'U': 'A',
            'N': 'N',
            '-': 'N'
}
# from https://stackoverflow.com/questions/25188968/reverse-complement-of-dna-strand-using-python
def reverse_complement(seq):
    bases = list(seq.upper())
    bases = reversed([complement.get(base,base) for base in bases])
    bases = ''.join(bases)
    return bases

# from https://www.hackerrank.com/challenges/text-wrap/forum
def wrap(string, max_width):
    return "\n".join([string[i:i+max_width] for i in range(0, len(string), max_width)])

def write_contig(h,b,n):
    n.write(h)
    rc_bases = reverse_complement(b)
    n.write(wrap(rc_bases, 60))


def reverse_complement_sequence(o,n):
    header_line = None
    bases = ''
    with open(o,'r') as original_file, open(n,'w') as new_file:
        # iterate through file
        for line in original_file.readlines():
            if line.startswith('>'):
                if header_line is not None:
                    write_contig(header_line,bases,new_file)
                header_line = line
                bases = ''
            else:
                bases = bases + line.rstrip()
        # write final contig
        write_contig(header_line,bases,new_file)

#################
# run main code #
#################

if __name__ == "__main__":

    # Check input ok
    args = get_options()
    
    # Get kmers
    klist = np.arange(args.min_k, args.max_k + 1, args.k_step)
    
    # Make directory and databases
    if not os.path.isdir(args.prefix):
        try:
            os.makedirs(args.prefix)
        except OSError:
            sys.stderr.write("Cannot create output directory\n")
            sys.exit(1)
    
    # Strand-specific sketch of reference sequence
    orientated_db_name = args.prefix + "/" + os.path.basename(args.prefix) + ".orientated"
    orientated_names, orientated_sequences = readRfile(args.orientation_file)
    # ensure only one sequence in the lists
    orientated_names = [orientated_names[0]]
    orientated_sequences = [orientated_sequences[0]]
    pp_sketchlib.constructDatabase(orientated_db_name,
                                    orientated_names,
                                    orientated_sequences,
                                    klist,
                                    args.sketch_size,
                                    True, # strand-specific
                                    0, # min_count
                                    False, # use_exact
                                    args.threads)
    
    # Strand-specific sketch of unknown sequences
    unknown_strand_db_name = args.prefix + "/" + os.path.basename(args.prefix) + ".unknown.ss"
    unknown_names, unknown_sequences = readRfile(args.unknown_file)
    pp_sketchlib.constructDatabase(unknown_strand_db_name,
                                    unknown_names,
                                    unknown_sequences,
                                    klist,
                                    args.sketch_size,
                                    True, # strand-specific
                                    0, # min_count
                                    False, # use_exact
                                    args.threads)
    
    # Canonical sketch of unknown sequences
    unknown_canonical_db_name = args.prefix + "/" + os.path.basename(args.prefix) + ".unknown.canonical"
    pp_sketchlib.constructDatabase(unknown_canonical_db_name,
                                    unknown_names,
                                    unknown_sequences,
                                    klist,
                                    args.sketch_size,
                                    False, # not strand-specific
                                    0, # min_count
                                    False, # use_exact
                                    args.threads)
    
    # Compare strand-specific sketches
    ss_distMat = pp_sketchlib.queryDatabase(unknown_strand_db_name,
                                            orientated_db_name,
                                            unknown_names,
                                            orientated_names,
                                            klist,
                                            True, # Correction
                                            False, # Jaccard
                                            args.threads,
                                            args.use_gpu,
                                            args.deviceid)
    
    # Compare canonical sketch and strand-specified sketch
    canonical_distMat = pp_sketchlib.queryDatabase(unknown_canonical_db_name,
                                            orientated_db_name,
                                            unknown_names,
                                            orientated_names,
                                            klist,
                                            True, # Correction
                                            False, # Jaccard
                                            args.threads,
                                            args.use_gpu,
                                            args.deviceid)

    # Compare output distances - use accessory as less susceptible to noise
    original_files = {}
    original_ss_accessory_distance = {}
    canonical_accessory_better_match = np.greater(ss_distMat[:,1],canonical_distMat[:,1])
    for i,(r,f) in enumerate(zip(unknown_names,unknown_sequences)):
        if canonical_accessory_better_match[i]:
            # store original distance
            original_ss_accessory_distance[r] = ss_distMat[i,1]
            original_files[r] = f
    
    # Reverse complement selected sequences and test whether matches improve
    if len(original_files.keys()) > 0:
        # write candidates for reorientation to new file
        rc_list_file = args.prefix + '.list'
        rc_names = []
        rc_sequences = []
        rc_files = {}
        rc_db_name = args.prefix + '/' + os.path.basename(args.prefix)
        with open(rc_list_file, 'w') as rc_list:
            for r in original_files:
                rc_files[r] = []
                rc_list.write(r)
                rc_names.append(r)
                rc_sequences.append(original_files[r])
                # Can be converted to multiprocessing pool if necessary
                for f in original_files[r]:
                    rc_file = args.prefix + '/rc.' + os.path.basename(f)
                    rc_list.write('\t' + rc_file)
                    reverse_complement_sequence(f,rc_file)
                    rc_files[r].append(rc_file)
                rc_list.write('\n')
                
        # Sketch reverse complemented sequences
        pp_sketchlib.constructDatabase(rc_db_name,
                                        rc_names,
                                        rc_sequences,
                                        klist,
                                        args.sketch_size,
                                        True, # strand-specific
                                        0, # min_count
                                        False, # use_exact
                                        args.threads)
                                        
        # Query against the original correctly-orientated sequence
        rc_ss_distMat = pp_sketchlib.queryDatabase(rc_db_name,
                                                orientated_db_name,
                                                rc_names,
                                                orientated_names,
                                                klist,
                                                True, # Correction
                                                False, # Jaccard
                                                args.threads,
                                                args.use_gpu,
                                                args.deviceid)
                                                
        # Test if reverse-complemented virus is closer to reference than original orientation
        report_file_name = args.prefix + '/' + os.path.basename(args.prefix) + '.reverse_complement.txt'
        with open(report_file_name, 'w') as report_file:
            report_file.write('Sequence\tOriginal distance\tReverse complement distance\tAction\n')
            for i,r in enumerate(rc_names):
                report_file.write(r + '\t' + str(original_ss_accessory_distance[r]) + '\t' + str(rc_ss_distMat[i,1]))
                # Replace original sequences
                if rc_ss_distMat[i,1] < (original_ss_accessory_distance[r] + args.threshold):
                    report_file.write('\t' + 'Use reverse-complement' + '\n')
                    if args.update_sequences:
                        for i,f in enumerate(original_files[r]):
                            os.rename(rc_files[r][i],f)
                else:
                    report_file.write('\t' + 'Use original' + '\n')
            
    else:
        sys.stderr.write('No evidence of sequences requiring reorientation\n')
    
    # Tidy up
    if not args.keep:
        os.rmdir(args.prefix)
    
    sys.exit(0)
