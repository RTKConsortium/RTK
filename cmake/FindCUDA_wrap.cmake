# - Wrapper around FindCUDA

if (MINGW)
  # Cuda doesn't work with mingw at all
  set (CUDA_FOUND FALSE)
elseif (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} LESS 2.8)
  # FindCuda is included with CMake 2.8
  set (CUDA_FOUND FALSE)
else ()
  # GCS 2011.03.16
  # Make nvcc less whiny
  if (CMAKE_COMPILER_IS_GNUCC)
    set (CUDA_PROPAGATE_HOST_FLAGS OFF)
  endif ()

  # GCS 2012-05-11:  We need to propagate cxx flags to nvcc, but 
  # the flag -ftest-coverage causes nvcc to barf, so exclude that one
  if (CMAKE_COMPILER_IS_GNUCC)
    string (REPLACE "-ftest-coverage" "" TMP "${CMAKE_CXX_FLAGS}")
    string (REPLACE "-ftemplate-depth=50" "" TMP "${TMP}")
    string (REPLACE " " "," TMP "${TMP}")
    set (CUDA_CXX_FLAGS ${CUDA_CXX_FLAGS} ${TMP})
  endif ()

  # GCS 2012-05-07: Workaround for poor, troubled FindCUDA
  set (CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE FALSE)
  find_package (CUDA QUIET)
endif ()

# GCS 2012-09-25 - Seems this is needed too
if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
  set (CUDA_CXX_FLAGS "${CUDA_CXX_FLAGS},-fPIC")
endif ()


set (CUDA_FOUND ${CUDA_FOUND} CACHE BOOL "Did we find cuda?")

if (CUDA_FOUND)
  cuda_include_directories (${CMAKE_CURRENT_SOURCE_DIR})
endif ()

# JAS 08.25.2010
#   Check to make sure nvcc has gcc-4.3 for compiling.
#   This script will modify CUDA_NVCC_FLAGS if system default is not gcc-4.3
include (nvcc-check)

set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
      -gencode arch=compute_10,code=sm_10
      -gencode arch=compute_11,code=sm_11
      -gencode arch=compute_12,code=sm_12
      -gencode arch=compute_13,code=sm_13
    )

if(CUDA_VERSION_MAJOR GREATER "2")
  set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
        -gencode arch=compute_20,code=sm_20
    )
endif()

