# Artifact for PPoPP 2027 Paper #234
**Revisiting Eigenvalue-Only Divide-and-Conquer Tridiagonal Eigensolver from Algorithmic Perspective**

Badges requested: **Artifact Available**, **Artifact Functional**, **Results Reproduced** (CPU and GPU).
DOI: TODO (Zenodo). License: Apache-2.0 (see `LICENSE` and `NOTICE`).

The artifact contains the boundary-row (BR) values-only divide-and-conquer (D&C) tridiagonal eigensolver, in two implementations:
- **CPU** (`cpu/`): a LAPACK-style Fortran/OpenMP prototype (`DSTEDC_DC_BR`) and the paper's CPU benchmark jobs (Tables 1–5, Figs. 4–5).
- **GPU** (`gpu/`): the CUDA prototype and `tridiag_dc_bench`, which compares BR against cuSOLVER `cusolverDnXstedc` (compz=N) (Fig. 6, Sec. 5.6).

Part of this repository has been upstreamed to Reference-LAPACK: the BR solver is merged as the new `DSTEBR`/`SSTEBR` routines (with `xLAED0_BR`, `xLAED4_BR`, `xLAED7_BR`, `xLAED8_BR`, `xLAED9_BR`, LAPACKE wrappers and tests) in [Reference-LAPACK/lapack#1424](https://github.com/Reference-LAPACK/lapack/pull/1424). The `cpu/` sources here are the exact versions used for the paper's experiments.

Reviewers are expected to use their own hardware. Section 2 lists what is needed and Section 6 gives scaled-down options.

## 1. Layout
| Path | Contents |
|---|---|
| `cpu/src/` | The 5 BR routines (`dstedc_dc_br.f`, `dlaed0_br.f`, `dlaed4_br.f`, `dlaed7_br.f`, `dlaed9_br.f`) and a NUMA helper. They link against any stock BLAS/LAPACK. |
| `cpu/bench/`, `cpu/jobs/` | Paper benchmark sources and job scripts (repo-relative; `module load` optional) |
| `cpu/patches/` | Working-tree diff on top of the pinned OpenBLAS fork commit (provenance, see `cpu/README.md`) |
| `gpu/` | Trimmed values-only GPU D&C plus `tridiag_dc_bench`, with a standalone CMake (see `gpu/README.md`) |
| `scripts/` | `getting_started.sh`; `cpu/` (build, smoke, scaled-down, run-all); `gpu/` (Fig. 6, Sec. 5.6); `gpu/original/` holds the authors' Slurm scripts for reference |
| `data/inputs/reduced_dense_49152.de.bin` | The only shipped input: the tridiagonal (D, E) of a reduced dense 49,152 matrix (768 KiB), used by Table 3 (reduced dense row), Fig. 5 (right) and PDSTEBZ. All other inputs are generated (Section 5). |
| `results/` | All outputs go here |

Provenance: the CPU code comes from https://github.com/uestc-ndsl-hpc/OpenBLAS, branch `brdc-svd-style-wprod`, commit `1eed268`, plus `cpu/patches/worktree-vs-1eed268.patch`. The GPU code comes from SyncFlowEVD branch `bench/PPoPP-bench`, commit `9acae31`, plus working-tree edits. Both are the versions used for the paper.

## 2. Hardware and software requirements
**Reference platforms (paper, Sec. 5.1):**
- CPU: one dual-socket Intel Xeon Gold 6348 node, 56 physical cores. Intel oneAPI MKL 2025.3 with MKL dynamic threading disabled. Every baseline and BR are linked against the same MKL.
- GPU: one NVIDIA H100-PCIe 80 GB, CUDA 13.2, 8 host cores.

**Required for Results Reproduced:**
| Part | Hardware | Software |
|---|---|---|
| GPU (Fig. 6, Sec. 5.6) | An H100- or 80 GB-class NVIDIA GPU (sm_90 reference; sm_80 also builds). At N=49,152 cuSOLVER's workspace is about 54 GiB, and at N=65,536 it is 96.0 GiB (which is the point of Sec. 5.6). | CUDA 13.2 (cuBLAS, cuSOLVER with `cusolverDnXstedc`, CUB, NVTX), CMake ≥ 3.18, a C++17 host compiler, LAPACK (OpenBLAS or MKL) for the optional CPU reference |
| CPU (Tables 1–5, Figs. 4–5) | A 56-core-class x86-64 node (2 sockets). RAM: TODO (confirm the reference node size). The DLAED0 internal D&C baseline queries about 6.0 GiB at N=16,384, and lazy replay at N=65,536 runs out of memory at more than 96 GiB on the reference node (Table 2). | gfortran with OpenMP, MKL 2025.3 (`MKLROOT`), MPFR (Table 1), Intel MPI with MKL ScaLAPACK (PDSTEBZ; `I_MPI_ROOT`), a C compiler |

## 3. Getting started (Artifact Functional, < 30 min)
```
# CPU (any x86-64 Linux, gfortran + OpenBLAS or MKL), then GPU (NVIDIA + CUDA 13.2)
./scripts/getting_started.sh
SKIP_GPU=1 ./scripts/getting_started.sh                         # CPU only
LAPACK_LIBS="-L$MKLROOT/lib -Wl,-rpath,$MKLROOT/lib -lmkl_rt" ./scripts/getting_started.sh   # use MKL
CUDA_ARCH=80 GPU_CMAKE_ARGS="-DAE_USE_MKL=ON" ./scripts/getting_started.sh                    # A100/A800, MKL
```
- **CPU part:** builds the BR routines and `scripts/cpu/smoke_br.f90`, which is an AE smoke test and not a paper experiment. It solves a random N=4096 tridiagonal with BR and DSTERF. It prints the workspace sizes, both times, and `max_rel_diff_vs_dsterf`, then `PASS` if that difference is below 1e-10. On our test box it reports differences around 1e-14.
- **GPU part:** builds `gpu/` and runs N=4096 on the normal, toeplitz and clustered generators, with BR, cuSOLVER and the CPU LAPACK reference. It prints one row per input (columns include `self_ms`, `cslv_ms`, `cslv/self` and BR error vs. the CPU reference).

## 4. Reproducing the paper (Results Reproduced)
Step 0 (inputs): `bash cpu/jobs/run_spectrum_de_cache_job.sh` writes `results/data/{uniform,normal,toeplitz,clustered}_{8192,16384,32768,49152,65536}.de.bin`. CPU and GPU both use these files. The generator is single-threaded and takes tens of minutes on a slow machine.

| Paper item | Command | Platform |
|---|---|---|
| Fig. 6 | `scripts/gpu/fig6_h100.sh` (alternative using built-in generators: `scripts/gpu/fig6_builtin_generators.sh`) | GPU |
| Sec. 5.6, N=65,536 | `scripts/gpu/sec56_oom_65k.sh` | GPU |
| Footnote 2 (optional) | `scripts/gpu/original/run_h100_cusolver_toeplitz_32k_49k_profile_job.sh` (needs Nsight; uses the authors' paths) | GPU |
| Table 1 | `scripts/cpu/run_paper_cpu.sh table1` | CPU + MPFR |
| Tables 2–3 | `scripts/cpu/run_paper_cpu.sh table2_3` | CPU + MKL, MPI/ScaLAPACK |
| Table 4 | `scripts/cpu/run_paper_cpu.sh table4` | CPU |
| Table 5 | `scripts/cpu/run_paper_cpu.sh table5` | CPU |
| Fig. 4 | `scripts/cpu/run_paper_cpu.sh fig4` (threads 1, 2, 4, 8, 16, 28, 56; N=16,384 and 65,536) | CPU |
| Fig. 5 | `scripts/cpu/run_paper_cpu.sh fig5` | CPU |
CPU outputs (CSV/logs) go to `results/cpu/`, and GPU logs go to `results/fig6/` and `results/sec56/`. Full-run times: TODO.

## 5. Expected results (values quoted from the paper)
Absolute times depend on the machine. Reviewers should check the ratios and trends.
- **Table 1** (N=16,384; e_MPFR = max abs error / max(1, ‖λ_MPFR‖∞); columns QR/QL, internal D&C, BR):
  - Uniform: 2.99e-14, 6.07e-16, 3.91e-16
  - Normal: 4.65e-14, 5.94e-16, 2.92e-16
  - Toeplitz: 8.99e-15, 7.68e-16, 7.77e-16
  - Clustered: 2.99e-17, 8.11e-17, 8.11e-17
  - Glued Wilkinson: 7.40e-15, 4.31e-16, 6.10e-16
  - Expected trend: BR is at the same scale as internal D&C, and both are closer to the reference than QR/QL on the non-trivial rows.
- **Table 2** (uniform N=65,536): QR/QL 56.35 s; bisection over 10 min; PDSTEBZ 18.34 s; MRRR 88.85 s; lazy-replay D&C out of memory (over 96 GiB); BR 168 ms.
- **Table 3** (columns: DSTERF time, then DSTERF/BR speedup at 1 thread and at 56 threads):
  - Uniform 16K: 4.03 s, 65.8×, 612.0×
  - Uniform 65K: 56.35 s, 140.0×, 335.6×
  - Normal 16K: 3.50 s, 58.0×, 505.2×
  - Normal 65K: 47.29 s, 144.1×, 470.0×
  - Clustered 16K: 42.1 ms, 2.39×, 10.30×
  - Clustered 65K: 257 ms, 3.17×, 14.96×
  - Toeplitz 16K: 3.40 s, 1.58×, 20.11×
  - Toeplitz 65K: 51.96 s, 1.59×, 19.93×
  - Reduced dense 49,152: 29.85 s, 0.95×, 9.14×
- **Table 4** (N=16,384; columns: internal D&C / BR / ratio at 1 thread, then the same at 56 threads):
  - Uniform: 170 ms / 61.2 ms / 2.77×; 17.5 ms / 6.56 ms / 2.66×
  - Normal: 169 / 60.3 ms / 2.80×; 17.0 / 6.92 ms / 2.46×
  - Toeplitz: 2.60 s / 2.16 s / 1.21×; 2.47 s / 169 ms / 14.61×
  - Clustered: 69.6 / 17.7 ms / 3.94×; 6.71 / 4.09 ms / 1.64×
- **Table 5** (N=16,384, 56 threads; columns: r, root K, DSTERF/BR):
  - Gauss–Legendre: 0.99, 16,384, 11.0×
  - 1D Laplacian: 0.60, 8,192, 20.5×
  - Anderson: 0.47, 234, 523.6×
  - The paper also reports 2868× for Anderson at N=65,536.
- **Fig. 4**: BR self-speedup T1/Tp at N=65,536 tracks deflation. Clustered reaches its best time before 56 threads. The per-point values are in the figure (TODO: copy them from the source data).
- **Fig. 5**: Static secular-root parallelism is 2.12–3.16× slower than adaptive on uniform, normal and clustered. Task-level is 4.79× slower on Toeplitz. Adaptive costs 2–3% probe overhead. On reduced dense N=49,152, BR state gives 6.6× and the adaptive scheduler 21.4× over lazy replay.
- **Fig. 6** (H100; cuSOLVER/BR at N = 8K / 16K / 32K / 49K):
  - Normal: 1.1 / 1.6 / 2.3 / 3.5×
  - Uniform: 1.1 / 1.6 / 2.3 / 3.4×
  - Toeplitz: 12.4 / 20.2 / 25.8 / 11.4×
  - Clustered: 1.2 / 2.0 / 2.5 / 4.1×
  - Text: 1.14–25.82× through N=32,768 and 3.43–11.39× at 49,152.
- **Sec. 5.6**: at N=65,536 cuSOLVER's queried workspace is 96.0 GiB, which does not fit on an 80 GB H100. BR returns the uniform row in 60.5 ms.

## 6. Scaled-down options (no reference hardware)
- CPU without 56 cores or MKL: `scripts/cpu/scaled_down.sh` (BR vs DSTERF over `SIZES` and `THREADS`, linked with OpenBLAS). It shows the thread-scaling trend. The paper jobs also run on fewer cores, but the 56-thread columns will not match.
- GPU with less than 80 GB: `SIZES="8192 16384" scripts/gpu/fig6_h100.sh`. A 24 GB GPU cannot run cuSOLVER at N=32,768 (about 24 GiB workspace), while BR still runs. The memory-ceiling effect of Sec. 5.6 then shows up at a smaller N.
- GPU sm_80 (A100/A800): `CUDA_ARCH=80`.

## 7. Claims
**Supported by this artifact:**
1. BR keeps O(n) persistent auxiliary state, where values-only D&C with lazy replay needs O(n^2) (Sec. 3.3, Table 2). The bench workspace queries show this.
2. Accuracy: BR is within the internal D&C error scale against a 256-bit MPFR reference (Table 1).
3. CPU performance against DSTERF, internal DLAED0(ICOMPQ=0), MRRR, bisection and PDSTEBZ (Tables 2–5). Self-speedup tracks deflation (Fig. 4). The adaptive scheduler tracks the better static policy (Fig. 5).
4. GPU BR is faster than cuSOLVER Xstedc(compz=N) wherever both fit (Fig. 6). cuSOLVER cannot run at N=65,536 on an 80 GB GPU, while BR completes (Sec. 5.6).

**Not supported / out of scope:**
- Identical absolute timings on other hardware or library versions.
- Dense reduction and back-transformation, which the paper excludes. The GPU dense-input and eigenvector paths are stubbed out in this artifact.
- Distributed solver (future work).
- ELPA probes (supplemental; `cpu/jobs/install_elpa_cpu_mpi.sh` and `run_elpa_tridiagonal_job.sh` are provided as-is).
- Formal (Lean 4) proofs are not part of this artifact.
