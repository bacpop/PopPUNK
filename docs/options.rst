Options
=======

Usage::

   usage: PopPUNK [-h]
               (--easy-run | --create-db | --fit-model | --refine-model | --assign-query)
               [--ref-db REF_DB] [--r-files R_FILES] [--q-files Q_FILES]
               [--distances DISTANCES] --output OUTPUT [--plot-fit PLOT_FIT]
               [--full-db] [--update-db] [--overwrite] [--min-k MIN_K]
               [--max-k MAX_K] [--k-step K_STEP] [--sketch-size SKETCH_SIZE]
               [--K K] [--dbscan] [--D D]
               [--min-cluster-prop MIN_CLUSTER_PROP] [--pos-shift POS_SHIFT]
               [--neg-shift NEG_SHIFT] [--manual-start MANUAL_START]
               [--indiv-refine] [--no-local] [--model-dir MODEL_DIR]
               [--previous-clustering PREVIOUS_CLUSTERING] [--core-only]
               [--accessory-only] [--microreact] [--cytoscape]
               [--rapidnj RAPIDNJ] [--perplexity PERPLEXITY]
               [--info-csv INFO_CSV] [--mash MASH] [--threads THREADS]
               [--no-stream] [--version]

   PopPUNK (POPulation Partitioning Using Nucleotide Kmers)

Command line options

   optional arguments:
     -h, --help            show this help message and exit

   Mode of operation:
     --easy-run            Create clusters from assemblies with default settings
     --create-db           Create pairwise distances database between reference
                           sequences
     --fit-model           Fit a mixture model to a reference database
     --refine-model        Refine the accuracy of a fitted model
     --assign-query        Assign the cluster of query sequences without re-
                           running the whole mixture model

   Input files:
     --ref-db REF_DB       Location of built reference database
     --r-files R_FILES     File listing reference input assemblies
     --q-files Q_FILES     File listing query input assemblies
     --distances DISTANCES
                           Prefix of input pickle of pre-calculated distances

   Output options:
     --output OUTPUT       Prefix for output files (required)
     --plot-fit PLOT_FIT   Create this many plots of some fits relating k-mer to
                           core/accessory distances [default = 0]
     --full-db             Keep full reference database, not just representatives
     --update-db           Update reference database with query sequences
     --overwrite           Overwrite any existing database files

   Kmer comparison options:
     --min-k MIN_K         Minimum kmer length [default = 9]
     --max-k MAX_K         Maximum kmer length [default = 29]
     --k-step K_STEP       K-mer step size [default = 4]
     --sketch-size SKETCH_SIZE
                           Kmer sketch size [default = 10000]

   Model fit options:
     --K K                 Maximum number of mixture components [default = 2]
     --dbscan              Use DBSCAN rather than mixture model
     --D D                 Maximum number of clusters in DBSCAN fitting [default
                           = 100]
     --min-cluster-prop MIN_CLUSTER_PROP
                           Minimum proportion of points in a cluster in DBSCAN
                           fitting [default = 0.0001]

   Refine model options:
     --pos-shift POS_SHIFT
                           Maximum amount to move the boundary away from origin
                           [default = 0.2]
     --neg-shift NEG_SHIFT
                           Maximum amount to move the boundary towards the origin
                           [default = 0.4]
     --manual-start MANUAL_START
                           A file containing information for a start point. See
                           documentation for help.
     --indiv-refine        Also run refinement for core and accessory
                           individually
     --no-local            Do not perform the local optimization step (speed up
                           on very large datasets)

   Database querying options:
     --model-dir MODEL_DIR
                           Directory containing model to use for assigning
                           queries to clusters [default = reference database
                           directory]
     --previous-clustering PREVIOUS_CLUSTERING
                           Directory containing previous cluster definitions and
                           network [default = use that in the directory
                           containing the model]
     --core-only           Use a core-distance only model for assigning queries
                           [default = False]
     --accessory-only      Use an accessory-distance only model for assigning
                           queries [default = False]

   Further analysis options:
     --microreact          Generate output files for microreact visualisation
     --cytoscape           Generate network output files for Cytoscape
     --rapidnj RAPIDNJ     Path to rapidNJ binary to build NJ tree for Microreact
     --perplexity PERPLEXITY
                           Perplexity used to calculate t-SNE projection (with
                           --microreact) [default=20.0]
     --info-csv INFO_CSV   Epidemiological information CSV formatted for
                           microreact (with --microreact or --cytoscape)

   Other options:
     --mash MASH           Location of mash executable
     --threads THREADS     Number of threads to use [default = 1]
     --no-stream           Use temporary files for mash dist interfacing. Reduce
                           memory use/increase disk use for large datasets
     --version             show program's version number and exit

