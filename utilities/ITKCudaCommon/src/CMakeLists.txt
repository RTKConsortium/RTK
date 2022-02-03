set(ITKCudaCommon_SRC
    itkCudaContextManager.cxx
    itkCudaDataManager.cxx
    itkCudaUtil.cxx
    itkCudaMemoryProbe.cxx
  )

include_directories(${ITKCudaCommon_SOURCE_DIR}/include)
include_directories(${ITK_INCLUDE_DIRS})
add_library(ITKCudaCommon ${ITK_LIBRARY_BUILD_TYPE} ${ITKCudaCommon_SRC})
target_link_libraries(ITKCudaCommon LINK_PUBLIC ${ITKCommon_LIBRARIES} CUDA::cudart CUDA::cuda_driver)

if (NOT ITK_SOURCE_DIR)
  # Export to RTK build targets must be done after RTK has been loaded by ITK.
  # When building externally we must do this after including ITKModuleExternal.
  # See RTK/CMakeLists.txt.
else()
    itk_module_target_export(ITKCudaCommon)
endif()

#=========================================================
if(NOT RTK_INSTALL_NO_LIBRARIES)
  target_include_directories(ITKCudaCommon PUBLIC $<INSTALL_INTERFACE:${ITKCudaCommon_INSTALL_INCLUDE_DIR}>)
  install(TARGETS ITKCudaCommon EXPORT RTK
    RUNTIME DESTINATION ${ITKCudaCommon_INSTALL_RUNTIME_DIR} COMPONENT Runtime
    LIBRARY DESTINATION ${ITKCudaCommon_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
    ARCHIVE DESTINATION ${ITKCudaCommon_INSTALL_ARCHIVE_DIR} COMPONENT Development
    PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_PREFIX}/${ITKCudaCommon_INSTALL_INCLUDE_DIR}"
  )
  install(TARGETS ITKCudaCommon EXPORT ITKTargets
    RUNTIME DESTINATION ${ITKCudaCommon_INSTALL_RUNTIME_DIR} COMPONENT Runtime
    LIBRARY DESTINATION ${ITKCudaCommon_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
    ARCHIVE DESTINATION ${ITKCudaCommon_INSTALL_ARCHIVE_DIR} COMPONENT Development
    PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_PREFIX}/${ITKCudaCommon_INSTALL_INCLUDE_DIR}"
  )
endif()
