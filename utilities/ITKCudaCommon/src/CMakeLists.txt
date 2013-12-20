set(ITKCudaCommon_SRC
    itkCudaContextManager.cxx
    itkCudaDataManager.cxx
    itkCudaKernelManager.cxx
    itkCudaUtil.cxx
  )

# Avoid issue with command line too long for nvcc
SET_DIRECTORY_PROPERTIES(PROPERTIES INCLUDE_DIRECTORIES "" )

# This is necessary to include the win32 header from RTK
# These lines should be removed if ITKCudaCommon is put as
# an ITK module
include_directories(${RTK_SOURCE_DIR}/code)
include_directories(${RTK_BINARY_DIR})

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${ITKCudaCommon_SOURCE_DIR}/include)
include_directories(${ITK_INCLUDE_DIRS})
add_library(ITKCudaCommon ${ITKCudaCommon_SRC})
target_link_libraries(ITKCudaCommon ${ITKCommon_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_LIBRARIES})
