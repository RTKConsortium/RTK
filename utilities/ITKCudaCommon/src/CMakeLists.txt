set(ITKCudaCommon_SRC
    itkCudaContextManager.cxx
    itkCudaDataManager.cxx
    itkCudaKernelManager.cxx
    itkCudaUtil.cxx
  )

# Avoid issue with command line too long for nvcc
set_directory_properties(PROPERTIES INCLUDE_DIRECTORIES "" )

add_library(ITKCudaCommon ${ITKCudaCommon_SRC})
target_link_libraries(ITKCudaCommon ${ITKCommon_LIBRARIES})
target_link_libraries(ITKCudaCommon "$<BUILD_INTERFACE:${CUDA_CUDA_LIBRARY};${CUDA_LIBRARIES}>")

# This is necessary to include the win32 header from RTK
# These lines should be removed if ITKCudaCommon is put as
# an ITK module
target_include_directories(ITKCudaCommon PUBLIC "$<BUILD_INTERFACE:${RTK_SOURCE_DIR}/code>")
target_include_directories(ITKCudaCommon PUBLIC "$<BUILD_INTERFACE:${RTK_BINARY_DIR}>")

target_include_directories(ITKCudaCommon PUBLIC "$<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>")
target_include_directories(ITKCudaCommon PUBLIC "$<BUILD_INTERFACE:${ITKCudaCommon_SOURCE_DIR}/include>")
FOREACH(dir IN LISTS ITK_INCLUDE_DIRS)
  target_include_directories(ITKCudaCommon PUBLIC "$<BUILD_INTERFACE:${dir}>")
ENDFOREACH()

# for the install interface, others should be automatic through include inheritance
target_include_directories(ITKCudaCommon PUBLIC "$<INSTALL_INTERFACE:${RTK_INSTALL_INCLUDE_DIR}>")

if(NOT RTK_INSTALL_NO_LIBRARIES)
  install(TARGETS ITKCudaCommon EXPORT RTK
          RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
          LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
          ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
endif() 