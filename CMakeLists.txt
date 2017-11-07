#=========================================================
# RTK : Reconstruction Toolkit
#=========================================================
cmake_minimum_required(VERSION 2.8.4 FATAL_ERROR)
if(WIN32 OR NOT EXISTS /dev/urandom)
  cmake_minimum_required(VERSION 2.8.5)
endif()

## Tell CMake to be quiet
cmake_policy(SET CMP0003 NEW)
if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()

project(RTK)

## RTK Version
set(RTK_VERSION_MAJOR "1")
set(RTK_VERSION_MINOR "3")
set(RTK_VERSION_PATCH "0")

set(RTK_LIBRARIES RTK)

## Default to release
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif ()

#=========================================================
# Installation variables
#=========================================================
if(NOT RTK_INSTALL_RUNTIME_DIR)
  set(RTK_INSTALL_RUNTIME_DIR bin)
endif()
if(NOT RTK_INSTALL_LIB_DIR)
  set(RTK_INSTALL_LIB_DIR lib)
endif()
if(NOT RTK_INSTALL_ARCHIVE_DIR)
  set(RTK_INSTALL_ARCHIVE_DIR lib)
endif()
if(NOT RTK_INSTALL_INCLUDE_DIR)
  set(RTK_INSTALL_INCLUDE_DIR include/RTK)
endif()
if(NOT RTK_INSTALL_PACKAGE_DIR)
  set(RTK_INSTALL_PACKAGE_DIR "lib/cmake/RTK")
endif()

#Set position independent code for Unix (-fPIC)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#=========================================================
# Remove some MS Visual c++ flags
if(MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS -D_CRT_NONSTDC_NO_DEPRECATE -D_SCL_SECURE_NO_WARNINGS)
endif()

#=========================================================
# Remove some Intel compiler warnings
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  string(REPLACE "-Wno-unused-parameter" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
  if(WIN32)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Qwd1268")
  else()
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -wd1268")
  endif()
endif()

# --------------------------------------------------------
# Shared libraries option
set(RTK_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})

# --------------------------------------------------------
# Initialization
if(NOT ITK_SOURCE_DIR)
    include(itk-module-init.cmake)
endif()

#=========================================================
# If choose to build documentation, then search for Doxygen executables.
option(BUILD_DOXYGEN "Build Doxygen Documentation" OFF)
if(BUILD_DOXYGEN)
  add_subdirectory(documentation/Doxygen)
endif()

#=========================================================
# lp_solve library
#=========================================================
add_subdirectory(utilities/lp_solve)

set(LPSOLVE_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}/utilities/lp_solve
  ${PROJECT_SOURCE_DIR}/utilities/lp_solve/shared
  ${PROJECT_SOURCE_DIR}/utilities/lp_solve/bfp
  ${PROJECT_SOURCE_DIR}/utilities/lp_solve/bfp/bfp_LUSOL
  ${PROJECT_SOURCE_DIR}/utilities/lp_solve/bfp/bfp_LUSOL/LUSOL
  ${PROJECT_SOURCE_DIR}/utilities/lp_solve/colamd
  )

list(APPEND RTK_INCLUDE_DIRS
  "${LPSOLVE_INCLUDE_DIRS}")

# --------------------------------------------------------
# Find ITK (required)
if(NOT ITK_SOURCE_DIR)
  find_package(ITK 4.2.0 REQUIRED)
endif()

if(ITK_VERSION_MINOR LESS "4")
  include_directories(BEFORE utilities/itkImageScanlineConstIterator)
endif()
if(ITK_VERSION_MINOR LESS "5")
  include_directories(BEFORE utilities/itkBinShrinkImageFilter)
endif()

#=========================================================
# ITKCudaCommon
#=========================================================
if(RTK_USE_CUDA)
  if(NOT TARGET ITKCudaCommon)
    add_subdirectory(utilities/ITKCudaCommon)
  endif()
endif()

#=========================================================
# Include directories
#=========================================================
list(APPEND RTK_INCLUDE_DIRS
  ${RTK_BINARY_DIR})

# Export library linking directories (used by LIBPATH)
set(RTK_SYSTEM_LIBRARY_DIRS ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(RTK_SYSTEM_INCLUDE_DIRS ${RTK_INCLUDE_DIRS})

#=========================================================
# Configure and build ITK external module
#=========================================================
if(NOT ITK_SOURCE_DIR)
  list(APPEND CMAKE_MODULE_PATH ${ITK_CMAKE_DIR})
  include(ITKModuleExternal)
else()
  itk_module_impl()
endif()

#=========================================================
# Generate RTKConfig.cmake for the build tree.
set (RTK_USE_FILE "${RTK_SOURCE_DIR}/cmake/UseRTK.cmake")
set (RTK_LIBRARY_DIRS ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
configure_file(cmake/RTKConfig.cmake.in RTKConfig.cmake @ONLY)
configure_file(cmake/RTKConfigVersion.cmake.in RTKConfigVersion.cmake @ONLY)

#-----------------------------------------------------------------------------
# The subdirectories added below this line should use only the public
# interface with find_package(ITK). Set ITK_DIR to use this ITK build.
if(ITK_SOURCE_DIR)
  set(ITK_DIR "${ITK_BINARY_DIR}")
endif()
#=========================================================
# Build applications
#=========================================================
option(BUILD_APPLICATIONS "Build RTK applications" ON)
if(BUILD_APPLICATIONS)
  add_subdirectory(applications)
endif()
