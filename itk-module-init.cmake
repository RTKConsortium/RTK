#
# Find the packages required by this module
#
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

include(CheckLanguage)
check_language(CUDA)

# Determine default value for RTK_USE_CUDA
set(RTK_USE_CUDA_DEFAULT OFF)
if (CMAKE_CUDA_COMPILER)
  find_package(CUDAToolkit)
  if(NOT CUDAToolkit_FOUND)
    message(WARNING "CUDAToolkit not found (available since CMake v3.17). RTK_USE_CUDA set to OFF.")
  elseif(DEFINED CUDAToolkit_VERSION AND "${CUDAToolkit_VERSION}" VERSION_LESS 8.0)
    message(WARNING "CUDA version ${CUDAToolkit_VERSION} is not supported by RTK, version 8 is required. RTK_USE_CUDA set to OFF.")
  else()
    set(RTK_USE_CUDA_DEFAULT ON)
  endif()
endif()
option(RTK_USE_CUDA "Use CUDA for RTK" ${RTK_USE_CUDA_DEFAULT})

# Configure CUDA compilation options
if(RTK_USE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  find_package(CUDAToolkit REQUIRED 8.0)

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-declarations")
  set(RTK_CUDA_PROJECTIONS_SLAB_SIZE "16" CACHE STRING "Number of projections processed simultaneously in CUDA forward and back projections")
endif()
