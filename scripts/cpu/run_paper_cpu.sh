#!/usr/bin/env bash
# Runs the paper's CPU experiments in order (reference: 2x Xeon Gold 6348, 56 cores, MKL 2025.3).
# Requires MKLROOT; MPFR (Table 1); Intel MPI + MKL ScaLAPACK (PDSTEBZ, I_MPI_ROOT). Outputs go to results/cpu/.
# Pass a subset as arguments, e.g.: run_paper_cpu.sh fig4 fig5
set -euo pipefail
AE=$(cd "$(dirname "$0")/../.." && pwd); J=$AE/cpu/jobs; export AE_ROOT=$AE
: "${MKLROOT:?set MKLROOT to an MKL 2025.3 installation}"
declare -A JOBS=(
  [inputs]="run_spectrum_de_cache_job.sh"
  [table1]="run_mpfr_main_families_probe_job.sh run_mpfr_value_probe_job.sh"
  [table2_3]="run_dsterf_large_job.sh run_mrrr_bisection_job.sh run_extended_mrrr_sweep_job.sh run_extended_pdstebz_sweep_job.sh run_pdstebz_table_complete_job.sh run_cache_accuracy_16k_65k_job.sh"
  [table4]="run_cache_accuracy_16k_49152_internal_job.sh run_cache_accuracy_16k_49152_internal_cases_job.sh"
  [table5]="run_application_tridiag_bench_job.sh run_application_tridiag_deflation_job.sh run_application_tridiag_fast_job.sh"
  [fig4]="run_cpu_strong_scaling_job.sh"
  [fig5]="run_scheduler_ablation_job.sh run_state_scheduler_ablation_49152_job.sh"
)
for k in ${@:-inputs table1 table2_3 table4 table5 fig4 fig5}; do
  for j in ${JOBS[$k]}; do echo "=== [$k] $j"; bash "$J/$j" 2>&1 | tee "$AE/results/cpu_${k}_${j%.sh}.log"; done
done
