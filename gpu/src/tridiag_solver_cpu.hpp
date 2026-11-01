#ifndef EVD_LIBRARY_INTERNAL_TRIDIAG_SOLVER_CPU_HPP
#define EVD_LIBRARY_INTERNAL_TRIDIAG_SOLVER_CPU_HPP

#include "evd/handle.h"
#include "evd/status.h"
#include "syev2stage_pipeline.hpp"
#include "tridiag_solver_dc.cuh"

namespace evd::internal {

evd_status run_tridiag_solver_cpu(evd_handle handle,
                                  const Syev2StageState &state,
                                  const TridiagSolverWorkspaceView &view);

} // namespace evd::internal

#endif // EVD_LIBRARY_INTERNAL_TRIDIAG_SOLVER_CPU_HPP
