set(ITKCudaCommon_SRCS
    itkCudaContextManager.cxx
    itkCudaDataManager.cxx
    itkCudaUtil.cxx
    itkCudaMemoryProbe.cxx
  )

itk_module_add_library(ITKCudaCommon ${ITKCudaCommon_SRCS})
target_link_libraries(ITKCudaCommon PUBLIC ${CUDA_CUDA_LIBRARY} ${CUDA_LIBRARIES} ${CUDA_cufft_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})
target_include_directories(ITKCudaCommon PUBLIC ${CUDA_INCLUDE_DIRS})
