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

IF(NOT RTK_INSTALL_NO_LIBRARIES)
  target_include_directories(ITKCudaCommon PUBLIC $<INSTALL_INTERFACE:${RTK_INSTALL_INCLUDE_DIR}/ITKCudaCommon>)
  install(TARGETS ITKCudaCommon EXPORT RTK
    RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
    LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
    ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development
    PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_PREFIX}/${RTK_INSTALL_INCLUDE_DIR}/ITKCudaCommon"
  )
ENDIF(NOT RTK_INSTALL_NO_LIBRARIES)

# if(ITK_SOURCE_DIR)
#   itk_module_link_dependencies()
#   itk_module_target(ITKCudaCommon)
# endif()
