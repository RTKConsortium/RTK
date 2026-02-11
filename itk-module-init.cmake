#
# Find the packages required by this module
#
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

# Set default Cuda architecture if not provided. The first case allows for
# backward compatibility with cmake versions before 3.20 which did not handle
# CUDAARCHS environment variable.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(DEFINED ENV{CUDAARCHS})
    set(
      CMAKE_CUDA_ARCHITECTURES
      "$ENV{CUDAARCHS}"
      CACHE STRING
      "CUDA architectures"
    )
  else()
    set(CMAKE_CUDA_ARCHITECTURES "52" CACHE STRING "CUDA architectures")
  endif()
endif()
include(CheckLanguage)
check_language(CUDA)

# Determine default value for RTK_USE_CUDA
set(RTK_USE_CUDA_DEFAULT OFF)
if(CMAKE_CUDA_COMPILER)
  set(CUDAToolkit_NVCC_EXECUTABLE ${CMAKE_CUDA_COMPILER})
  find_package(CUDAToolkit)
  if(NOT CUDAToolkit_FOUND)
    message(
      WARNING
      "CUDAToolkit not found (available since CMake v3.17). RTK_USE_CUDA set to OFF."
    )
  elseif(
    DEFINED
      CUDAToolkit_VERSION
    AND
      "${CUDAToolkit_VERSION}"
        VERSION_LESS
        8.0
  )
    message(
      WARNING
      "CUDA version ${CUDAToolkit_VERSION} is not supported by RTK, version 8 is required. RTK_USE_CUDA set to OFF."
    )
  else()
    set(RTK_USE_CUDA_DEFAULT ON)
  endif()
endif()
option(RTK_USE_CUDA "Use CUDA for RTK" ${RTK_USE_CUDA_DEFAULT})

if(RTK_USE_CUDA)
  # Configure CUDA compilation options
  if(NOT RTK_CUDA_VERSION)
    set(
      RTK_CUDA_VERSION
      ${CUDAToolkit_VERSION}
      CACHE STRING
      "Specify the exact CUDA version that must be used for RTK"
    )
  else()
    set(
      RTK_CUDA_VERSION
      ${RTK_CUDA_VERSION}
      CACHE STRING
      "Specify the exact CUDA version that must be used for RTK"
    )
  endif()
  mark_as_advanced(RTK_CUDA_VERSION)

  enable_language(CUDA)
  set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  if(RTK_CUDA_VERSION)
    find_package(CUDAToolkit EXACT ${RTK_CUDA_VERSION} REQUIRED)
  else()
    find_package(CUDAToolkit REQUIRED 8.0)
  endif()

  set(
    RTK_CUDA_PROJECTIONS_SLAB_SIZE
    "16"
    CACHE STRING
    "Number of projections processed simultaneously in CUDA forward and back projections"
  )
endif()
