# CPU values-only tridiagonal D&C (BR)
- `src/{dstedc_dc_br,dlaed0_br,dlaed4_br,dlaed7_br,dlaed9_br}.f`: the BR solver. Entry point: `DSTEDC_DC_BR(N, D, E, WORK, LWORK, IWORK, LIWORK, INFO)`. Pass LWORK=-1 for a workspace query. OpenMP parallel. These files link against any stock BLAS/LAPACK (OpenBLAS, MKL, reference LAPACK).
- `src/openblas_get_num_nodes.c`: NUMA-node query helper. The job scripts compile it, but the BR objects don't need it.
- `bench/`: benchmark sources used by the paper jobs. Most benchmark programs are embedded as heredocs in `jobs/*.sh`. `bench/deflation_recheck/` is the instrumented BR variant used for the Table 5 deflation statistics.
- `jobs/`: the authors' paper job scripts, ported to repo-relative paths. Outputs go to `results/cpu/`. `module load` is optional. Set `MKLROOT` (MKL 2025.3) and, for PDSTEBZ, `I_MPI_ROOT`.
- `patches/worktree-vs-1eed268.patch`: working-tree changes on top of the pinned fork commit.

Provenance (the paper version): fork https://github.com/uestc-ndsl-hpc/OpenBLAS, branch `brdc-svd-style-wprod`, commit `1eed268`, plus `patches/worktree-vs-1eed268.patch`. The files in `src/` are byte-identical to that working tree. The full fork is only needed to rebuild libopenblas with BR inside it (it exports `dstedc_dc_br_`); the AE does not need it.
