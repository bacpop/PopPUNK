# vim: set fileencoding=<utf-8> :
# Copyright 2018-2021 John Lees and Nick Croucher

'''Print suggested citations and methods'''

import os

from .__init__ import __version__
import pp_sketchlib

from .sketchlib import readDBParams, getSeqsInDb


citation = \
"""1. Lees JA, Harris SR, Tonkin-Hill G, Gladstone RA, Lo SW, Weiser JN, Corander J, Bentley SD, Croucher NJ.
Fast and flexible bacterial genomic epidemiology with PopPUNK.
Genome Research 29:304-316 (2019).
doi:10.1101/gr.241455.118
2. Pedregosa, F. et al. Scikit-learn: Machine Learning in Python.
J. Mach. Learn. Res. 12, 2825–2830 (2011)
3. Virtanen, P. et al.
SciPy 1.0: fundamental algorithms for scientific computing in Python.3
Nat. Methods 17, 261–272 (2020)
4. Harris, C. R. et al.
Array programming with NumPy.
Nature 585, 357–362 (2020)
5. Peixoto, T. P.
The graph-tool python library. (2017)
doi:10.6084/m9.figshare.1164194"""

sketchlib_citation = \
"""6. Lees JA & Croucher NJ.
pp-sketchlib (2020).
doi:10.5281/zenodo.4531418
7. Zhao, X.
BinDash, software for fast genome distance estimation on a typical personal laptop.
Bioinformatics 35:671–673 (2019).
doi:10.1093/bioinformatics/bty651
8. Mohamadi, H., Chu, J., Vandervalk, B. P. & Birol, I.
ntHash: recursive nucleotide hashing.
Bioinformatics 32:3492–3494 (2016).
doi:10.1093/bioinformatics/btw397"""

poppunk_methods = "We built a database of %(number_samples)s isolates using " + \
"pp-sketchlib version %(sketchlib_version)s (doi:%(sketchlib_doi)s) with " + \
"sketch version %(sketchlib_hash)s, k-mer lengths %(kmin)s-%(kmax)s, a " + \
"sketch size of %(sketch_size)s and %(seed_type)s seeds [6-8]. We assigned " + \
"variable-length-k-mer clusters (VLKCs) using PopPUNK version %(poppunk_version)s " + \
"(doi:%(poppunk_doi)s) by fitting a %(model_mode)s with %(model_options)s [1-5].\n"

assign_methods = "We queried a database of %(number_samples)s isolates using pp-sketchlib version" + \
" %(sketchlib_version)s (doi:%(sketchlib_doi)s) with sketch version %(sketchlib_hash)s, " + \
"k-mer lengths %(kmin)s-%(kmax)s, a sketch size of %(sketch_size)s and %(seed_type)s seeds [6-8]. We assigned sequences to variable-length-k-mer clusters (VLKCs) using PopPUNK " + \
"version %(poppunk_version)s (doi:%(poppunk_doi)s) [1-5].\n"

def print_citation(args, assign=False):
    # Read values from DB
    try:
        if assign:
            db_prefix = args.db
        else:
            db_prefix = args.ref_db
        n_samples = str(len(getSeqsInDb(db_prefix + "/" + os.path.basename(db_prefix) + ".h5")))
        kmers, sketch_size, codon_phased = readDBParams(db_prefix)
        kmin = str(min(kmers))
        kmax = str(max(kmers))
        sketch_size = str(sketch_size * 64)
        seed_phasing = "codon-phased" if codon_phased else "dense"
    except:
        n_samples = "X"
        kmin = "X"
        kmax = "X"
        sketch_size = "X"
        seed_phasing = "X"

    try:
        pp_sketchlib_version = pp_sketchlib.version
        sketch_version = pp_sketchlib.version
    except AttributeError:
        pp_sketchlib_version = "X"
        sketch_version = "X"

    if assign:
        print(assign_methods % {"number_samples" : n_samples,
                                 "sketchlib_version" : pp_sketchlib_version,
                                 "sketchlib_doi" : "10.5281/zenodo.4531418",
                                 "sketchlib_hash" : sketch_version,
                                 "kmin" : kmin,
                                 "kmax" : kmax,
                                 "sketch_size" : sketch_size,
                                 "seed_type" : seed_phasing,
                                 "poppunk_version" : __version__,
                                 "poppunk_doi" : "10.1101/gr.241455.118"})
    else:
        if args.fit_model == "bgmm":
            model_mode = "BGMM"
            model_options = str(args.K) + " components"
        elif args.fit_model == "dbscan":
            model_mode = "HDBSCAN"
            model_options = str(args.D) + " maximum clusters and a minimum of "  + \
                    str(round((n_samples * (n_samples + 1))/(2 * args.min_cluster_prop))) + \
                    " points per cluster"
        elif args.fit_model == "refine":
            model_mode = "refined"
            model_options = "score " + str(args.score_idx) + " used to optimise the VLKCs"
        elif args.fit_model == "threshold":
            model_mode = "simple threshold"
            model_options = "a core-distance cutoff distance of " + args.threshold
        elif args.fit_model == "lineage":
            model_mode = "lineage"
            model_options = "ranks of " + args.ranks
        else:
            model_mode = "UNKNOWN"
            model_options = "UNKNOWN"

        print(poppunk_methods % {"number_samples" : n_samples,
                                 "sketchlib_version" : pp_sketchlib_version,
                                 "sketchlib_doi" : "10.5281/zenodo.4531418",
                                 "sketchlib_hash" : sketch_version,
                                 "kmin" : kmin,
                                 "kmax" : kmax,
                                 "sketch_size" : sketch_size,
                                 "seed_type" : seed_phasing,
                                 "poppunk_version" : __version__,
                                 "poppunk_doi" : "10.1101/gr.241455.118",
                                 "model_mode" : model_mode,
                                 "model_options" : model_options})

    print(citation)
    print(sketchlib_citation)
