# - Wrapper around FindCUDA

IF (MINGW)
  # Cuda doesn't work with mingw at all
  SET (CUDA_FOUND FALSE)
ELSEIF (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} EQUAL 2.6
    AND ${CMAKE_PATCH_VERSION} LESS 2)
  # FindCuda requires CMake 2.6.2
  SET (CUDA_FOUND FALSE)
ELSE (MINGW)
  # In cmake 2.8, use stock findcuda.  Otherwise, use included one.
  IF (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} LESS 2.8)
    SET (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/FindCUDA/CMake/cuda" 
      ${CMAKE_MODULE_PATH})
  ENDIF (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} LESS 2.8)
  FIND_PACKAGE (CUDA QUIET)
ENDIF (MINGW)

SET (CUDA_FOUND ${CUDA_FOUND} CACHE BOOL "Did we find cuda?")

IF (CUDA_FOUND)
  CUDA_INCLUDE_DIRECTORIES (
    ${CMAKE_CURRENT_SOURCE_DIR}
    )
ENDIF(CUDA_FOUND)

# JAS 08.25.2010
#   Check to make sure nvcc has gcc-4.3 for compiling.
#   This script will modify CUDA_NVCC_FLAGS if system default is not gcc-4.3
#INCLUDE (nvcc-check.cmake)
INCLUDE (nvcc-check)

# JAS 08.25.2010
#   Build code for all known compute capabilities
SET (CUDA_NVCC_FLAGS
  ${CUDA_NVCC_FLAGS}
  -gencode arch=compute_10,code=sm_10
  -gencode arch=compute_11,code=sm_11
  -gencode arch=compute_12,code=sm_12
  -gencode arch=compute_20,code=sm_20
  )
