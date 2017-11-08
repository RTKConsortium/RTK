project(ITKCudaCommon)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/CMake)

include(itkCUDA)

set(ITKCudaCommon_LIBRARIES ITKCudaCommon)
set(ITKCudaCommon_SYSTEM_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
set(ITKCudaCommon_SYSTEM_LIBRARY_DIRS ${CUDA_LIBRARIES})

include_directories(${ITKCudaCommon_SOURCE_DIR}/include)
if(NOT ITK_SOURCE_DIR)
  add_subdirectory(src)
else()
  itk_module_impl()
endif()
