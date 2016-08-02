project(ITKCudaCommon)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/CMake)

include(itkCUDA)

set(ITKCudaCommon_LIBRARIES ITKCudaCommon)
set(ITKCudaCommon_SYSTEM_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
set(ITKCudaCommon_SYSTEM_LIBRARY_DIRS ${CUDA_LIBRARIES})

include_directories(${ITKCudaCommon_SOURCE_DIR}/include)
add_subdirectory(src)

# Installation
if(NOT RTK_INSTALL_NO_DEVELOPMENT)
  # Include .h and .hxx files
  file(GLOB __files1 "${ITKCudaCommon_SOURCE_DIR}/include/*.h")
  file(GLOB __files2 "${ITKCudaCommon_SOURCE_DIR}/include/*.hxx")
  install(FILES ${__files1} ${__files2}
          DESTINATION ${RTK_INSTALL_INCLUDE_DIR}
          COMPONENT Development)
endif()
