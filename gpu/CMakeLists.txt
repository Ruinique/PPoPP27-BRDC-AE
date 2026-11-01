# Minimal standalone build for the PPoPP'27 #234 AE: values-only GPU tridiagonal
# D&C (BR) + tridiag_dc_bench (BR vs cuSOLVER cusolverDnXstedc, optional CPU LAPACK reference).
cmake_minimum_required(VERSION 3.18)
project(br_tridiag_gpu LANGUAGES CXX CUDA)
option(AE_USE_MKL "Use Intel MKL as the CPU LAPACK reference (default: any LAPACK via FindLAPACK)" OFF)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 90)   # H100; use 80 for A100/A800
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
find_package(CUDAToolkit REQUIRED)
if(AE_USE_MKL)
  find_package(MKL CONFIG REQUIRED)
  set(AE_LAPACK MKL::MKL)
  set(AE_LAPACK_DEFS EVD_ENABLE_MKL=1)
else()
  find_package(LAPACK REQUIRED)
  set(AE_LAPACK ${LAPACK_LIBRARIES})
endif()

add_library(br_tridiag
  src/handle.cu
  src/tridiag_solver.cu
  src/tridiag_solver_cpu.cpp
  src/ae_stage_stubs.cu)
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA"
   AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.2
   AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.3)
  # Kept from the original build: CUDA 13.2 ptxas O3 miscompiles this TU.
  set_source_files_properties(src/tridiag_solver.cu PROPERTIES COMPILE_OPTIONS "-Xptxas=-O2")
endif()
target_include_directories(br_tridiag PUBLIC include PRIVATE src)
target_compile_definitions(br_tridiag PRIVATE ${AE_LAPACK_DEFS})
target_link_libraries(br_tridiag PUBLIC CUDA::cudart PRIVATE CUDA::cublas CUDA::cusolver ${AE_LAPACK})
set_target_properties(br_tridiag PROPERTIES CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)

add_executable(tridiag_dc_bench bench/tridiag_dc_bench.cu)
target_include_directories(tridiag_dc_bench PRIVATE src)
target_link_libraries(tridiag_dc_bench PRIVATE br_tridiag CUDA::cudart CUDA::cublas CUDA::cusolver ${AE_LAPACK})
set_target_properties(tridiag_dc_bench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
