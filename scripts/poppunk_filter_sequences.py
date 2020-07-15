#!/Users/nicholascroucher/miniconda3/envs/poppunk/bin/python
# vim: set fileencoding=<utf-8> :
# Copyright 2020 John Lees and Nick Croucher

import pickle
import sys
import numpy as np
import argparse

# command line parsing
def get_options():

    parser = argparse.ArgumentParser(description='Filter input assemblies by quality', prog='filter_sequences')

    # input options
    parser.add_argument('--r-files',
        type = str,
        required=True,
        help='File listing reference input assemblies (required)')
    parser.add_argument('--min-length',
        type = float,
        default = 10000.0,
        help='Minimum length of assembly (default = 10000)')
    parser.add_argument('--max-length',
        type = float,
        default = 100000000.0,
        help='Maximum length of assembly (default = 100000000)')
    parser.add_argument('--max-undetermined',
        type = float,
        default = 0.05,
        help='Maximum proportion of undetermined bases (default = 0.05)')
    parser.add_argument('--max-contigs',
        type = float,
        default = 100000000.0,
        help='Maximum number of contigs within assembly (default = 100000000)')
    parser.add_argument('--output',
        type = str,
        required = True,
        help='Prefix for output files')
    
    return parser.parse_args()

# process each individual sequence
def process_assembly(fn) :
    """Generates some simple quality statistics from a FASTA input file

    Args:
        fn (str)
            Path to sequence file.

    Returns:
        s_length
            Total length of sequence (int)
        n_contig
            Number of contigs in sequence (int)
        p_undetermined
            Proportion of sequence that consists of ambiguous bases (float)
    """
    
    # data structures
    s_length = 0
    n_contig = 0
    n_undetermined = 0
    # iterate through file
    with open(fn, 'r') as s_file:
        for line in s_file.readlines():
            if line[0] == '>':
                n_contig += 1
            else:
                s_length += len(line.rstrip())
                n_undetermined += sum(map(line.count, ['n','N','x','X','-']))
    # calculate proportion undetermined
    p_undetermined = n_undetermined/s_length
    # return summaries
    return(s_length, n_contig, p_undetermined)
    
# main code
if __name__ == "__main__":
    
    # Check input ok
    args = get_options()

    # read input file
    file_name = {}
    with open(args.r_files, 'r') as rfile:
        for line in rfile.readlines():
            entry = line.rstrip().split('\t')
            file_name[entry[0]] = entry[1]
            
    # iterate through files and analyse
    with open(args.output + '.list', 'w') as filtered_list:
        with open(args.output + '.assembly_stats.csv', 'w') as stats_csv:
            stats_csv.write('Sequence,File,Length,NumContigs,ProportionUndetermined\n')
            for s_name in file_name:
                s_length, n_contig, p_undetermined = process_assembly(file_name[s_name])
                stats_csv.write(s_name + ',' + file_name[s_name] + ',' + str(s_length) + ',' + str(n_contig) + ',' + str(p_undetermined) + '\n')
                if s_length >= args.min_length and s_length <= args.max_length \
                    and n_contig <= args.max_contigs and p_undetermined <= args.max_undetermined:
                    filtered_list.write(s_name + '\t' + file_name[s_name] + '\n')
