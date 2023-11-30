Roadmap
====================================

.. |nbsp| unicode:: 0xA0
   :trim:

This document describes our future plans for additions to PopPUNK, `pp-sketchlib <https://github.com/bacpop/pp-sketchlib>`__ and `BeeBOP <https://github.com/bacpop/beebop/>`__
and BeeBOP. Tasks are in order of priority.

PopPUNK
-------
1. Containerise the workflow. See `#193 <https://github.com/bacpop/PopPUNK/issues/193>`__, `#277 <https://github.com/bacpop/PopPUNK/issues/277>`__, `#278 <https://github.com/bacpop/PopPUNK/issues/278>`__.
2. Add full worked tutorials back to the documentation `#275 <https://github.com/bacpop/PopPUNK/issues/275>`__.
3. Make the update pipeline more robust. See `#273 <https://github.com/bacpop/PopPUNK/issues/273>`__.
4. Codebase optimsation and refactoring
    - Modularisation of the network code `#249 <https://github.com/bacpop/PopPUNK/issues/249>`__.
    - Removing old functions `#103 <https://github.com/bacpop/PopPUNK/issues/103>`__
5. Add more species databases:
    - N. meningitidis `#267 <https://github.com/bacpop/PopPUNK/issues/267>`__.
    - H. influenzae `#276 <https://github.com/bacpop/PopPUNK/issues/276>`__.
6. Stable names for lineage/subclustering modes.

Other enhancements listed on the `issue page <https://github.com/bacpop/pp-sketchlib/issues>`__ are currently not planned.

pp-sketchlib
------------

1. Update installation in package managers
    - Update for new macOS `#92 <https://github.com/bacpop/ska.rust#planned-features>`__
    - Rebuild conda recipe for CUDA12 and newer HDF5 `#46 <https://github.com/conda-forge/pp-sketchlib-feedstock/pull/46>`__
2. Allow amino-acids as input `#89 <https://github.com/bacpop/pp-sketchlib/issues/89>`__.

Other enhancements listed on the `issue page <https://github.com/bacpop/pp-sketchlib/issues>`__ are currently not planned.

BeeBOP
------

1. Update backend database to v8 `#42 <https://github.com/bacpop/beebop/pull/42>`__.
2. Update CI images.
3. Add more info on landing page.
    - News page.
    - About page.
    - Methods description.
4. Add custom cluster names (e.g. GPSCs)
5. Integration tests for error logging.
6. Persist user data.
    - Persist microreact tokens `#41 <https://github.com/bacpop/beebop/pull/41>`__.
    - Allow user to change or delete tokens
7. Add linage/subclusters to results page `#23 <https://github.com/bacpop/beebop/pull/23>`__.
8. Report sample quality to user.
9. Front-end update for large numbers of input files.
10. Add serotype prediction for *S. pneumoniae*.
11. Add multiple species databases.
