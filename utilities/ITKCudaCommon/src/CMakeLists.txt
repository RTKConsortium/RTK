set(ITKCudaCommon_SRC
    itkCudaContextManager.cxx
    itkCudaDataManager.cxx
    itkCudaKernelManager.cxx
    itkCudaUtil.cxx
  )

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${ITKCudaCommon_SOURCE_DIR}/include)
include_directories(${ITK_INCLUDE_DIRS})
add_library(ITKCudaCommon ${ITK_LIBRARY_BUILD_TYPE} ${ITKCudaCommon_SRC})
target_link_libraries(ITKCudaCommon LINK_PUBLIC ${ITKCommon_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_LIBRARIES})


# if(ITK_SOURCE_DIR)
#   itk_module_link_dependencies()
#   itk_module_target(ITKCudaCommon)
# endif()
