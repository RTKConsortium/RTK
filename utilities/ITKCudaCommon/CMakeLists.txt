cmake_minimum_required(VERSION 2.8.4 FATAL_ERROR)

project(ITKCudaCommon)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/CMake)

include(itkCUDA)

set(ITKCudaCommon_LIBRARIES ITKCudaCommon)

list(APPEND ITKCudaCommon_INCLUDE_DIRS
  ${ITKCudaCommon__SOURCE_DIR}/include
  ${ITKCommon_BINARY_DIR}
  ${CUDA_TOOLKIT_INCLUDE}
  ${CUDA_INCLUDE_DIRS}
  ${ITK_INCLUDE_DIRS})

include_directories(${ITKCudaCommon_INCLUDE_DIRS})

set(ITKCudaCommon_SYSTEM_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
set(ITKCudaCommon_SYSTEM_LIBRARY_DIRS ${CUDA_LIBRARIES} )

if(NOT ITK_SOURCE_DIR)
  find_package(ITK REQUIRED)   
  list(APPEND CMAKE_MODULE_PATH ${ITK_CMAKE_DIR}) 
  include(ITKModuleExternal)
else()
  itk_module_impl()
endif()

# Installation
if(NOT RTK_INSTALL_NO_DEVELOPMENT)
  # Include .h and .hxx files
  file(GLOB __files1 "${ITKCudaCommon_SOURCE_DIR}/include/*.h")
  file(GLOB __files2 "${ITKCudaCommon_SOURCE_DIR}/include/*.hxx")
  install(FILES ${__files1} ${__files2}
          DESTINATION ${RTK_INSTALL_INCLUDE_DIR}
          COMPONENT Development)
endif()
