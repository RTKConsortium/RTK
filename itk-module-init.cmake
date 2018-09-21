#
# Find the packages required by this module
#
list(APPEND CMAKE_MODULE_PATH ${RTK_SOURCE_DIR}/cmake) 

find_package(CUDA_wrap QUIET)
if(CUDA_FOUND)
  if(${CUDA_VERSION} VERSION_LESS 8.0)
    message(WARNING "CUDA version ${CUDA_VERSION} is not supported by RTK.")
    set(RTK_USE_CUDA_DEFAULT OFF)
  else()
    set(RTK_USE_CUDA_DEFAULT ON)
  endif()
else()
  set(RTK_USE_CUDA_DEFAULT OFF)
endif()
option(RTK_USE_CUDA "Use CUDA for RTK" ${RTK_USE_CUDA_DEFAULT})

if(RTK_USE_CUDA)
  if(NOT CUDA_FOUND)
    find_package(CUDA_wrap REQUIRED)
  endif()
  set(RTK_CUDA_PROJECTIONS_SLAB_SIZE "16" CACHE STRING "Number of projections processed simultaneously in CUDA forward and back projections")
endif()

