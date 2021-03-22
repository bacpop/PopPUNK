Citing PopPUNK
==============

If you use PopPUNK, PopPIPE or pp-sketchlib in a scientific paper, we would appreciate
a citation. As a minimum, please cite the following paper(s):

Lees JA, Harris SR, Tonkin-Hill G, Gladstone RA, Lo SW, Weiser JN, Corander J, Bentley SD, Croucher NJ. Fast and flexible
bacterial genomic epidemiology with PopPUNK. *Genome Research* **29**:1-13 (2019).
doi:`10.1101/gr.241455.118 <https://dx.doi.org/10.1101/gr.241455.118>`__

Generating citations and methods
--------------------------------
You can add ``--citation`` to your PopPUNK command to generate a full list of papers
to cite. This will also produce a basic methods paragraph for you to edit and include. You
can do this after running ``poppunk_assign`` or ``poppunk --fit-model``::

    poppunk --citation --fit-model bgmm --ref-db example_db --K 4

gives:

    We built a database of 28 isolates using pp-sketchlib version 1.7.0 (doi:10.5281/zenodo.4531418)
    with sketch version 88ee3ff83ba294c928505f991e20078691ed090e, k-mer lengths 13-28, a sketch size of 9984 and dense seeds [6-8].
    We assigned variable-length-k-mer clusters (VLKCs) using PopPUNK version 2.4.0
    (doi:10.1101/gr.241455.118) by fitting a BGMM with 4 components [1-5].

or::

    poppunk_assign --citation --query some_queries.txt --db example_db

gives:

    We queried a database of 28 isolates and their pre-assigned variable-length-k-mer
    clusters (VLKCs) using pp-sketchlib version 1.7.0 (doi:10.5281/zenodo.4531418) with
    sketch version 88ee3ff83ba294c928505f991e20078691ed090e, k-mer lengths 13-28,
    a sketch size of 9984 and dense seeds [6-8]. We assigned the VLKCs using PopPUNK
    version 2.4.0 (doi:10.1101/gr.241455.118) [1-5].

If your journal requires versions for all software packages, you may find running
``conda list`` helpful. The ``poppunk_db_info.py`` script (:ref:`db-info`) can be
run on your ``.h5`` files to give useful information too.