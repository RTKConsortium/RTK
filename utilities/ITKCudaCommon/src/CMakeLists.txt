set(ITKCudaCommon_SRC
    itkCudaContextManager.cxx
    itkCudaDataManager.cxx
    itkCudaKernelManager.cxx
    itkCudaUtil.cxx
  )

set(ITKCudaCommon_Kernels 
    CudaImageOps.cu
    CudaReduction.cu
  )

# Avoid issue with command line too long for nvcc
SET_DIRECTORY_PROPERTIES(PROPERTIES INCLUDE_DIRECTORIES "" )

generate_cuda_ptx_wrappers(ITKCudaCommon_KernelsPTX ITKCudaCommon_KernelsWrappers ${ITKCudaCommon_Kernels})

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${ITKCudaCommon_SOURCE_DIR}/include)
include_directories(${ITK_INCLUDE_DIRS})
add_library(ITKCudaCommon ${ITKCudaCommon_SRC} ${ITKCudaCommon_KernelsWrappers} ${ITKCudaCommon_Kernels})
target_link_libraries(ITKCudaCommon ${ITKCommon_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_LIBRARIES})
