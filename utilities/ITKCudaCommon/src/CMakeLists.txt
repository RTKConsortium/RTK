set(ITKCudaCommon_SRC
    itkCudaContextManager.cxx
    itkCudaDataManager.cxx
    itkCudaKernelManager.cxx
    itkCudaUtil.cxx
  )

add_library(ITKCudaCommon ${ITKCudaCommon_SRC})
target_link_libraries(ITKCudaCommon LINK_PRIVATE ${ITKCommon_LIBRARIES})

itk_module_link_dependencies()
target_link_libraries(ITKCudaCommon LINK_PUBLIC ${CUDA_CUDA_LIBRARY} ${CUDA_LIBRARIES})
itk_module_target(ITKCudaCommon)