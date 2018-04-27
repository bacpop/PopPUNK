Options
=======

Usage::

   usage: PopPUNK [-h]
               (--create-db | --fit-model | --create-query-db | --assign-query)
               [--ref-db REF_DB] [--r-files R_FILES] [--q-files Q_FILES]
               [--distances DISTANCES] --output OUTPUT [--save-distances]
               [--plot-fit PLOT_FIT] [--full-db] [--update-db] [--overwrite]
               [--min-k MIN_K] [--max-k MAX_K] [--k-step K_STEP]
               [--sketch-size SKETCH_SIZE] [--K K] [--priors PRIORS] [--bgmm]
               [--t-dist] [--microreact] [--cytoscape] [--rapidnj RAPIDNJ]
               [--perplexity PERPLEXITY] [--info-csv INFO_CSV] [--mash MASH]
               [--threads THREADS] [--version]

Command line options:

   optional arguments:
     -h, --help            show this help message and exit

   Mode of operation:
     --easy-run            Create clusters from assemblies with default settings
     --create-db           Create pairwise distances database between reference
                           sequences
     --fit-model           Fit a mixture model to a reference database
     --create-query-db     Create distances between query sequences and a
                           reference database
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
     --save-distances      Store pickle of calculated distances for query
                           sequences
     --plot-fit PLOT_FIT   Create this many plots of some fits relating k-mer to
                           core/accessory distances[default = 0]
     --full-db             Keep full reference database, not just representatives
     --update-db           Update reference database with query sequences
     --overwrite           Overwrite any existing database files

   Kmer comparison options:
     --min-k MIN_K         Minimum kmer length [default = 9]
     --max-k MAX_K         Maximum kmer length [default = 29]
     --k-step K_STEP       K-mer step size [default = 4]
     --sketch-size SKETCH_SIZE
                           Kmer sketch size [default = 10000]

   Mixture model options:
     --K K                 Maximum number of mixture components (EM only)
                           [default = 2]
     --priors PRIORS       File specifying model priors. See documentation for
                           help
     --bgmm                Use ADVI rather than EM to fit the mixture model
     --t-dist              Use a mixture of t distributions rather than Gaussians
                           (ADVI only)

   Further analysis options:
     --microreact          Generate output files for microreact visualisation
     --cytoscape           Generate network output files for Cytoscape
     --rapidnj RAPIDNJ     Path to rapidNJ binary to build NJ tree for Microreact
     --perplexity PERPLEXITY
                           Perplexity used to calculate t-SNE projection (with
                           --microreact) [default=5.0]
     --info-csv INFO_CSV   Epidemiological information CSV formatted for
                           microreact (with --microreact or --cytoscape)

   Other options:
     --mash MASH           Location of mash executable
     --threads THREADS     Number of threads to use during database querying
                           [default = 1]
     --version             show program's version number and exit


