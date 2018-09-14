project(ITKCudaCommon)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/CMake)

include(itkCUDA)

set(ITKCudaCommon_LIBRARIES ITKCudaCommon)
set(ITKCudaCommon_SYSTEM_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
set(ITKCudaCommon_SYSTEM_LIBRARY_DIRS ${CUDA_LIBRARIES})

set(ITKCudaCommon_INCLUDE_DIRS ${ITKCudaCommon_SYSTEM_INCLUDE_DIRS}
  ${ITKCudaCommon_SOURCE_DIR}/include PARENT_SCOPE)

#=========================================================
# Installation variables
#=========================================================
if(NOT ITKCudaCommon_INSTALL_RUNTIME_DIR)
  set(ITKCudaCommon_INSTALL_RUNTIME_DIR bin)
endif()
if(NOT ITKCudaCommon_INSTALL_LIB_DIR)
  set(ITKCudaCommon_INSTALL_LIB_DIR lib)
endif()
if(NOT ITKCudaCommon_INSTALL_ARCHIVE_DIR)
  set(ITKCudaCommon_INSTALL_ARCHIVE_DIR lib)
endif()
if(NOT ITKCudaCommon_INSTALL_INCLUDE_DIR)
  set(ITKCudaCommon_INSTALL_INCLUDE_DIR include/ITKCudaCommon)
endif()
if(NOT ITKCudaCommon_INSTALL_PACKAGE_DIR)
  set(ITKCudaCommon_INSTALL_PACKAGE_DIR "lib/cmake/ITKCudaCommon")
endif()

include_directories(${ITKCudaCommon_INCLUDE_DIRS})

if(NOT ITK_SOURCE_DIR)
  add_subdirectory(src)
else()
  itk_module_impl()
endif()
