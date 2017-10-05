#=========================================================
cmake_minimum_required(VERSION 2.8.4 FATAL_ERROR)
if(WIN32 OR NOT EXISTS /dev/urandom)
  cmake_minimum_required(VERSION 2.8.5)
endif()

#=========================================================
#RTK = Reconstruction toolkit
project(RTK)
#=========================================================

#=========================================================
#Support for the CTest dashboard testing system
include(CTest)
include(cmake/RTKExternalData.cmake)
#=========================================================

## Tell CMake to be quiet
cmake_policy (SET CMP0003 NEW)
if(POLICY CMP0054)
  cmake_policy (SET CMP0054 NEW)
endif()

## Default to release
if (NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE Release)
endif ()

if(BUILD_TESTING)
  enable_testing()
endif()
#=========================================================

#=========================================================
set(RTK_VERSION_MAJOR "1")
set(RTK_VERSION_MINOR "3")
set(RTK_VERSION_PATCH "0")

#=========================================================

#=========================================================
# Help function to debug CMake
macro (DD in)
  message(${in}=[${${in}}])
endmacro()
#=========================================================

#=========================================================
# Output directories.
if(NOT LIBRARY_OUTPUT_PATH)
  set (LIBRARY_OUTPUT_PATH ${RTK_BINARY_DIR}/bin CACHE INTERNAL "Single output directory for building all libraries.")
endif()
if(NOT EXECUTABLE_OUTPUT_PATH)
  set (EXECUTABLE_OUTPUT_PATH ${RTK_BINARY_DIR}/bin CACHE INTERNAL "Single output directory for building all executables.")
endif()
mark_as_advanced(LIBRARY_OUTPUT_PATH EXECUTABLE_OUTPUT_PATH)

set(RTK_LIBRARY_PATH "${LIBRARY_OUTPUT_PATH}")
set(RTK_EXECUTABLE_PATH "${EXECUTABLE_OUTPUT_PATH}")
set(CXX_TEST_PATH ${EXECUTABLE_OUTPUT_PATH})

#=========================================================
# Remove some MS Visual c++ flags
if(MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS -D_CRT_NONSTDC_NO_DEPRECATE -D_SCL_SECURE_NO_WARNINGS)
endif()

# Flag required for shared libraries
if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif ()
#=========================================================

#=========================================================
# Find ITK (required)
find_package(ITK 4.2.0 REQUIRED)
if(ITK_FOUND)
  include(${ITK_USE_FILE})
else()
  message(FATAL_ERROR "Cannot build without ITK. Set ITK_DIR.")
endif()
set(ITK_LIBRARIES_SANS_VTK)
foreach(lib ${ITK_LIBRARIES})
  if(NOT lib MATCHES "ITKVtkGlue")
    list(APPEND ITK_LIBRARIES_SANS_VTK ${lib})
  endif()
endforeach()
set(ITK_LIBRARIES ${ITK_LIBRARIES_SANS_VTK})

# GPU optional dependencies
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(CUDA_wrap)
if(CUDA_FOUND)
    if(${CUDA_VERSION} LESS 4.1)
    message(WARNING "CUDA version ${CUDA_VERSION} is not supported by RTK.")
    set(RTK_USE_CUDA_DEFAULT OFF)
  else()
    set(RTK_USE_CUDA_DEFAULT ON)
  endif()
else()
  set(RTK_USE_CUDA_DEFAULT OFF)
endif()
option(RTK_USE_CUDA "Use CUDA for RTK" ${RTK_USE_CUDA_DEFAULT})

if(RTK_USE_CUDA)
  if(NOT CUDA_FOUND)
    find_package(CUDA_wrap REQUIRED)
  endif()
  include_directories(${CUDA_INCLUDE_DIRS})
  set(RTK_CUDA_PROJECTIONS_SLAB_SIZE "16" CACHE STRING "Number of projections processed simultaneously in CUDA forward and back projections")
endif()
#=========================================================

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
#=========================================================

#=========================================================
# Shared libraries option
option(BUILD_SHARED_LIBS "Build RTK with shared libraries." ${ITK_BUILD_SHARED})
set(RTK_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
#=========================================================

#=========================================================
# RTK uses KWStyle for checking the coding style
include(${RTK_SOURCE_DIR}/cmake/KWStyle/KWStyle.cmake)
#=========================================================


#=========================================================
# If choose to build documentation, then search for Doxygen executables.
option(BUILD_DOXYGEN "Build Doxygen Documentation" OFF)
if(BUILD_DOXYGEN)
  add_subdirectory(documentation/Doxygen)
endif()
#=========================================================

#=========================================================
# Installation variables
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
  set(RTK_INSTALL_INCLUDE_DIR include/RTK-${RTK_VERSION_MAJOR}.${RTK_VERSION_MINOR})
endif()
if(NOT RTK_INSTALL_PACKAGE_DIR)
  set(RTK_INSTALL_PACKAGE_DIR "lib/cmake/RTK-${RTK_VERSION_MAJOR}.${RTK_VERSION_MINOR}")
endif()
#=========================================================
# Utilities
list(APPEND RTK_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/code")
if(RTK_USE_CUDA)
  list(APPEND RTK_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/utilities/ITKCudaCommon/include/")
  list(APPEND RTK_INCLUDE_DIRS ${CUDA_TOOLKIT_INCLUDE})
endif()
list(APPEND RTK_INCLUDE_DIRS "${PROJECT_BINARY_DIR}")
include_directories(${RTK_INCLUDE_DIRS})

if(ITK_VERSION_MINOR LESS "4")
  include_directories(BEFORE utilities/itkImageScanlineConstIterator)
endif()
if(ITK_VERSION_MINOR LESS "5")
  include_directories(BEFORE utilities/itkBinShrinkImageFilter)
endif()

if(RTK_USE_CUDA)
  add_subdirectory(utilities/ITKCudaCommon)
  include_directories(utilities/ITKCudaCommon/include)
endif()
add_subdirectory(utilities/lp_solve)
include_directories(
        utilities/lp_solve
        utilities/lp_solve/shared
        utilities/lp_solve/bfp
        utilities/lp_solve/bfp/bfp_LUSOL
        utilities/lp_solve/bfp/bfp_LUSOL/LUSOL
        utilities/lp_solve/colamd)
list(APPEND RTK_INCLUDE_DIRS
    "${PROJECT_SOURCE_DIR}/utilities/lp_solve"
    "${PROJECT_SOURCE_DIR}/utilities/lp_solve/shared"
    "${PROJECT_SOURCE_DIR}/utilities/lp_solve/bfp"
    "${PROJECT_SOURCE_DIR}/utilities/lp_solve/bfp/bfp_LUSOL"
    "${PROJECT_SOURCE_DIR}/utilities/lp_solve/bfp/bfp_LUSOL/LUSOL"
    "${PROJECT_SOURCE_DIR}/utilities/lp_solve/colamdcode")
add_subdirectory(code)

#=========================================================
# Generate RTKConfig.cmake for the build tree.
set (RTK_USE_FILE "${RTK_SOURCE_DIR}/cmake/UseRTK.cmake")
set (RTK_LIBRARY_DIRS ${RTK_LIBRARY_PATH})
configure_file(cmake/RTKConfig.cmake.in RTKConfig.cmake @ONLY)
configure_file(cmake/RTKConfigVersion.cmake.in RTKConfigVersion.cmake @ONLY)
#=========================================================

option(RTK_TIME_EACH_FILTER "Time each RTK filter in a global object and report times in RTK applications" OFF)
if(ITK_VERSION LESS "4.6")
    set(RTK_TIME_EACH_FILTER OFF) 
endif()

option(BUILD_APPLICATIONS "Build RTK applications" ON)
if(BUILD_APPLICATIONS)
  add_subdirectory(applications)
endif()

if(BUILD_TESTING)
  add_subdirectory(testing)
endif()

option(BUILD_EXAMPLES "Build RTK examples" ON)
if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

ExternalData_Add_Target(rtkData)

option(BUILD_SIMPLERTK "Build SimpleRTK Wrapping" OFF)
if(BUILD_SIMPLERTK)
  # SimpleRTK is meant to be build separately, so we set the variables
  set(RTK_DIR ${CMAKE_BINARY_DIR})
  include(utilities/SimpleRTK/SimpleRTKWithSwig.cmake)
endif()
