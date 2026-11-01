# GPU values-only tridiagonal D&C (BR)
These are the paper sources, trimmed from SyncFlowEVD branch `bench/PPoPP-bench` @ 9acae31 (with its 4 uncommitted edits). That is the paper version.
- `src/tridiag_solver.cu`, `src/tridiag_solver_*.cuh`: the GPU BR values-only D&C. `src/handle.cu`, `src/handle_impl.hpp`: the handle. `src/tridiag_solver_cpu.cpp`: CPU LAPACK fallback/reference backend.
- `bench/tridiag_dc_bench.cu`: BR vs cuSOLVER `cusolverDnXstedc` (compz=N), with an optional CPU LAPACK reference.
- `src/ae_stage_stubs.cu` (AE only): stubs for the dense-reduction, back-transform and eigenvector-sign stages of the full EVD library. The paper's experiments start from tridiagonal input and compute eigenvalues only. If dense input or eigenvectors are requested, the stubs return EVD_STATUS_NOT_IMPLEMENTED.
Build: `cmake -S gpu -B build-gpu -DCMAKE_CUDA_ARCHITECTURES=90 [-DAE_USE_MKL=ON | -DBLA_VENDOR=OpenBLAS] && cmake --build build-gpu -j`
