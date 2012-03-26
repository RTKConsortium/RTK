# - Wrapper around FindCUDA

if (MINGW)
  # Cuda doesn't work with mingw at all
  set (CUDA_FOUND FALSE)
elseif (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} LESS 2.8)
  # FindCuda is included with CMake 2.8
  set (CUDA_FOUND FALSE)
else ()
  find_package (CUDA QUIET)

  # GCS 2011.03.16
  # Make nvcc less whiny
  if (CMAKE_COMPILER_IS_GNUCC)
    set (CUDA_PROPAGATE_HOST_FLAGS OFF)
    if (CMAKE_C_FLAGS)
        set (CUDA_NVCC_FLAGS "--compiler-options \"${CMAKE_C_FLAGS}\"")
    endif ()
  endif ()

  # JAS 08.25.2010
  #   Check to make sure nvcc has gcc-4.3 for compiling.
  #   This script will modify CUDA_NVCC_FLAGS if system default is not gcc-4.3
  include (nvcc-check)
endif ()

set (CUDA_FOUND ${CUDA_FOUND} CACHE BOOL "Did we find cuda?")

if (CUDA_FOUND)
  cuda_include_directories (${CMAKE_CURRENT_SOURCE_DIR})
endif ()

message (STATUS "CUDA Build Level: ALL Compute Capabilities")
set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
      -gencode arch=compute_10,code=sm_10
      -gencode arch=compute_11,code=sm_11
      -gencode arch=compute_12,code=sm_12
      -gencode arch=compute_13,code=sm_13
      -gencode arch=compute_20,code=sm_20
    )

