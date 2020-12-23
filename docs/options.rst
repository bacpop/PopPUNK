Options
=======

Usage::

       poppunk [-h]
               (--create-db | --fit-model {bgmm,dbscan,refine,lineage,threshold} | --use-model)
               [--ref-db REF_DB] [--r-files R_FILES]
               [--distances DISTANCES]
               [--external-clustering EXTERNAL_CLUSTERING]
               [--output OUTPUT] [--plot-fit PLOT_FIT] [--overwrite]
               [--graph-weights] [--min-k MIN_K] [--max-k MAX_K]
               [--k-step K_STEP] [--sketch-size SKETCH_SIZE]
               [--codon-phased] [--min-kmer-count MIN_KMER_COUNT]
               [--exact-count] [--strand-preserved]
               [--qc-filter {stop,prune,continue}] [--retain-failures]
               [--max-a-dist MAX_A_DIST] [--length-sigma LENGTH_SIGMA]
               [--length-range LENGTH_RANGE LENGTH_RANGE]
               [--prop-n PROP_N] [--upper-n UPPER_N] [--K K] [--D D]
               [--min-cluster-prop MIN_CLUSTER_PROP]
               [--threshold THRESHOLD] [--pos-shift POS_SHIFT]
               [--neg-shift NEG_SHIFT] [--manual-start MANUAL_START]
               [--indiv-refine] [--no-local] [--model-dir MODEL_DIR]
               [--ranks RANKS] [--use-accessory] [--threads THREADS]
               [--gpu-sketch] [--gpu-dist] [--deviceid DEVICEID]
               [--version]

Command line options::

  optional arguments:
    -h, --help            show this help message and exit

  Mode of operation:
    --create-db           Create pairwise distances database between
                          reference sequences
    --fit-model {bgmm,dbscan,refine,lineage,threshold}
                          Fit a mixture model to a reference database
    --use-model           Apply a fitted model to a reference database to
                          restore database files

  Input files:
    --ref-db REF_DB       Location of built reference database
    --r-files R_FILES     File listing reference input assemblies
    --distances DISTANCES
                          Prefix of input pickle of pre-calculated distances
    --external-clustering EXTERNAL_CLUSTERING
                          File with cluster definitions or other labels
                          generated with any other method.

  Output options:
    --output OUTPUT       Prefix for output files
    --plot-fit PLOT_FIT   Create this many plots of some fits relating k-mer
                          to core/accessory distances [default = 0]
    --overwrite           Overwrite any existing database files
    --graph-weights       Save within-strain Euclidean distances into the
                          graph

  Create DB options:
    --min-k MIN_K         Minimum kmer length [default = 13]
    --max-k MAX_K         Maximum kmer length [default = 29]
    --k-step K_STEP       K-mer step size [default = 4]
    --sketch-size SKETCH_SIZE
                          Kmer sketch size [default = 10000]
    --codon-phased        Used codon phased seeds X--X--X [default = False]
    --min-kmer-count MIN_KMER_COUNT
                          Minimum k-mer count when using reads as input
                          [default = 0]
    --exact-count         Use the exact k-mer counter with reads [default =
                          use countmin counter]
    --strand-preserved    Treat input as being on the same strand, and
                          ignore reverse complement k-mers [default = use
                          canonical k-mers]

  Quality control options:
    --qc-filter {stop,prune,continue}
                          Behaviour following sequence QC step: "stop"
                          [default], "prune" (analyse data passing QC), or
                          "continue" (analyse all data)
    --retain-failures     Retain sketches of genomes that do not pass QC
                          filters in separate database [default = False]
    --max-a-dist MAX_A_DIST
                          Maximum accessory distance to permit [default =
                          0.5]
    --length-sigma LENGTH_SIGMA
                          Number of standard deviations of length
                          distribution beyond which sequences will be
                          excluded [default = 5]
    --length-range LENGTH_RANGE LENGTH_RANGE
                          Allowed length range, outside of which sequences
                          will be excluded [two values needed - lower and
                          upper bounds]
    --prop-n PROP_N       Threshold ambiguous base proportion above which
                          sequences will be excluded [default = 0.1]
    --upper-n UPPER_N     Threshold ambiguous base count above which
                          sequences will be excluded

  Model fit options:
    --K K                 Maximum number of mixture components [default = 2]
    --D D                 Maximum number of clusters in DBSCAN fitting
                          [default = 100]
    --min-cluster-prop MIN_CLUSTER_PROP
                          Minimum proportion of points in a cluster in
                          DBSCAN fitting [default = 0.0001]
    --threshold THRESHOLD
                          Cutoff if using --fit-model threshold

  Refine model options:
    --pos-shift POS_SHIFT
                          Maximum amount to move the boundary away from
                          origin [default = to between-strain mean]
    --neg-shift NEG_SHIFT
                          Maximum amount to move the boundary towards the
                          origin [default = to within-strain mean]
    --manual-start MANUAL_START
                          A file containing information for a start point.
                          See documentation for help.
    --indiv-refine        Also run refinement for core and accessory
                          individually
    --no-local            Do not perform the local optimization step (speed
                          up on very large datasets)
    --model-dir MODEL_DIR
                          Directory containing model to use for assigning
                          queries to clusters [default = reference database
                          directory]

  Lineage analysis options:
    --ranks RANKS         Comma separated list of ranks used in lineage
                          clustering [default = 1,2,3]
    --use-accessory       Use accessory distances for lineage definitions
                          [default = use core distances]

  Other options:
    --threads THREADS     Number of threads to use [default = 1]
    --gpu-sketch          Use a GPU when calculating sketches (read data
                          only) [default = False]
    --gpu-dist            Use a GPU when calculating distances [default =
                          False]
    --deviceid DEVICEID   CUDA device ID, if using GPU [default = 0]
    --version             show program's version number and exit
