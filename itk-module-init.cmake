#
# Find the packages required by this module
#
list(APPEND CMAKE_MODULE_PATH ${RTK_SOURCE_DIR}/cmake) 

find_package(CUDA_wrap QUIET)
if(CUDA_FOUND)
  set(RTK_USE_CUDA_DEFAULT ON)
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

# Propagate cmake options in a header file
configure_file(${RTK_SOURCE_DIR}/rtkConfiguration.h.in
  ${RTK_BINARY_DIR}/rtkConfiguration.h)
