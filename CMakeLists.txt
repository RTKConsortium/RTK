#=========================================================
# RTK : Reconstruction Toolkit
#=========================================================
# Respect the CMAKE_CXX_STANDARD flags when building for
# ITKv5 or C++11.
cmake_minimum_required(VERSION 3.9.5 FATAL_ERROR)

## Only policies introduced after the cmake_minimum_required
## version need to explicitly be set to NEW.
## Refer to https://cmake.org/cmake/help/v3.11/manual/cmake-policies.7.html
set(CMAKE_POLICIES
  CMP0070
  CMP0071
  CMP0072)

foreach(p ${CMAKE_POLICIES})
  if(POLICY ${p})
    cmake_policy(SET ${p} NEW)
  endif()
endforeach()

## Set the default target properties for RTK
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 11) # Supported values are ``11``, ``14``, and ``17``.
endif()
if(NOT CMAKE_CXX_STANDARD_REQUIRED)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(NOT CMAKE_CXX_EXTENSIONS)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

#=========================================================
# Help function to debug CMake
macro (DD in)
  message(${in}=[${${in}}])
endmacro()
#=========================================================

project(RTK)

## RTK Version
set(RTK_VERSION_MAJOR "2")
set(RTK_VERSION_MINOR "1")
set(RTK_VERSION_PATCH "0")
set(RTK_VERSION_STRING "${RTK_VERSION_MAJOR}.${RTK_VERSION_MINOR}")

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

#=========================================================
# Allow for multiple CL.EXE to write to the same .PDB file
if(RTK_USE_CUDA)
  if(WIN32)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /FS")
  endif()
endif()

# --------------------------------------------------------
# Find ITK (required)
if(NOT ITK_SOURCE_DIR)
  find_package(ITK 5.0.1 REQUIRED)
endif()

# --------------------------------------------------------
# Shared libraries option
if(NOT ITK_SOURCE_DIR)
  set(RTK_BUILD_SHARED_LIBS ${ITK_BUILD_SHARED})
else()
  set(RTK_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
endif()

# ----------------------------------------------------------------------------
# Set RTK_DATA_ROOT
# Internally used by XRadRawToAttenuationImageFilter to set the path for its
# flat and dark headers.
if(NOT ITK_SOURCE_DIR)
  set(RTK_DATA_ROOT ${CMAKE_BINARY_DIR}/ExternalData/test CACHE PATH "Path of the data root" FORCE)
else()
  file(RELATIVE_PATH RTK_RELATIVE_PATH ${ITK_SOURCE_DIR} ${RTK_SOURCE_DIR})
  set(RTK_DATA_ROOT ${ExternalData_BINARY_ROOT}/${RTK_RELATIVE_PATH}/test CACHE PATH "Path of the data root" FORCE)
endif()
mark_as_advanced(RTK_DATA_ROOT)

# --------------------------------------------------------
# Initialization
if(NOT ITK_SOURCE_DIR)
  include(itk-module-init.cmake)
endif()

# Propagate cmake options in a header file
configure_file(${RTK_SOURCE_DIR}/rtkConfiguration.h.in
  ${RTK_BINARY_DIR}/rtkConfiguration.h)

#=========================================================
# If choose to build documentation, then search for Doxygen executables.
option(RTK_BUILD_DOXYGEN "Build Doxygen Documentation" OFF)
if(RTK_BUILD_DOXYGEN)
  add_subdirectory(documentation/Doxygen)
endif()

# Setup build locations.
if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${RTK_BINARY_DIR}/bin)
endif()
if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${RTK_BINARY_DIR}/lib)
endif()
if(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${RTK_BINARY_DIR}/lib)
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
# Setup KWStyle from ITK
if(ITK_USE_KWSTYLE)
  set(WORKING_DIR "${CMAKE_SOURCE_DIR}")
  if(ITK_SOURCE_DIR)
    set(WORKING_DIR "${ITK_SOURCE_DIR}")
  endif()

  find_package( Git )
  if(GIT_FOUND AND EXISTS "${WORKING_DIR}/.git/config")
    execute_process( COMMAND ${GIT_EXECUTABLE} config hooks.KWStyle.path
      "${KWSTYLE_EXECUTABLE}"
      WORKING_DIRECTORY ${WORKING_DIR} )
  endif()
endif()

#=========================================================
# ITKCudaCommon
#=========================================================
if(RTK_USE_CUDA)
  if(NOT TARGET ITKCudaCommon)
    add_subdirectory(utilities/ITKCudaCommon)
  endif()
endif()
list(APPEND RTK_INCLUDE_DIRS
  ${ITKCudaCommon_INCLUDE_DIRS})
#=========================================================
# Include directories
#=========================================================
list(APPEND RTK_INCLUDE_DIRS
  ${RTK_BINARY_DIR})

# Export library linking directories (used by LIBPATH)
set(RTK_SYSTEM_LIBRARY_DIRS ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(RTK_SYSTEM_INCLUDE_DIRS ${RTK_INCLUDE_DIRS})

#=========================================================
# Generate RTKConfig.cmake for the build tree.
set(RTK_MODULE_PATH_CONFIG ${CMAKE_MODULE_PATH})
set(RTK_LIBRARY_DIRS ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

set(RTK_EXPORT_CODE_BUILD "
# The RTK version number
set(RTK_VERSION_MAJOR ${RTK_VERSION_MAJOR})
set(RTK_VERSION_MINOR ${RTK_VERSION_MINOR})
set(RTK_VERSION_PATCH ${RTK_VERSION_PATCH})

# Whether the compiled version of RTK uses CUDA
set(RTK_USE_CUDA ${RTK_USE_CUDA})

if(${RTK_USE_CUDA})
  set(CMAKE_MODULE_PATH ${RTK_MODULE_PATH_CONFIG} ${CMAKE_MODULE_PATH})
  # if we are using CUDA, make sure CUDA libraries are available
  find_package(CUDA_wrap REQUIRED)
  include_directories(${CUDA_INCLUDE_DIRS})
  set(RTK_CUDA_PROJECTIONS_SLAB_SIZE \"16\" CACHE STRING \"Number of projections processed simultaneously in CUDA forward and back projections\")
endif()
")

set(RTK_EXPORT_CODE_INSTALL "
# The RTK version number
set(RTK_VERSION_MAJOR ${RTK_VERSION_MAJOR})
set(RTK_VERSION_MINOR ${RTK_VERSION_MINOR})
set(RTK_VERSION_PATCH ${RTK_VERSION_PATCH})

# Whether the compiled version of RTK uses CUDA
set(RTK_USE_CUDA ${RTK_USE_CUDA})

if(${RTK_USE_CUDA})
  # if we are using CUDA, make sure CUDA libraries are available
  find_package(CUDA_wrap REQUIRED)
  set(RTK_CUDA_PROJECTIONS_SLAB_SIZE \"16\" CACHE STRING \"Number of projections processed simultaneously in CUDA forward and back projections\")
endif()
")

#=========================================================
# Configure and build ITK external module
#=========================================================
if(NOT ITK_SOURCE_DIR)
  if(NOT EXISTS ${ITK_CMAKE_DIR}/ITKModuleMacros.cmake)
    message(FATAL_ERROR "Modules can only be built against an ITK build tree; they cannot be built against an ITK install tree.")
  endif()

  list(APPEND CMAKE_MODULE_PATH ${ITK_CMAKE_DIR})
  include(ITKModuleExternal)
else()
  itk_module_impl()
endif()

# Install configuration file
install(FILES ${RTK_BINARY_DIR}/rtkConfiguration.h DESTINATION ${RTK_INSTALL_INCLUDE_DIR})
install(FILES ${RTK_SOURCE_DIR}/cmake/FindCUDA_wrap.cmake
              ${RTK_SOURCE_DIR}/cmake/FindGengetopt.cmake
              ${RTK_SOURCE_DIR}/cmake/GetGitRevisionDescription.cmake
              ${RTK_SOURCE_DIR}/cmake/GetGitRevisionDescription.cmake.in
              ${RTK_SOURCE_DIR}/cmake/has_cuda_gpu.cxx
        DESTINATION ${ITK_INSTALL_PACKAGE_DIR})
if(NOT ITK_SOURCE_DIR)
  install(CODE "MESSAGE(FATAL_ERROR \"Cannot install, RTK is compiled separately from ITK. Installation is only functional if RTK is compiled within ITK.\")")
endif()

#=========================================================
# Build applications
#=========================================================
option(RTK_PROBE_EACH_FILTER "Probe each RTK filter in a global object and report times and memory used in RTK applications" OFF)
option(RTK_BUILD_APPLICATIONS "Build RTK applications" ON)
if(RTK_BUILD_APPLICATIONS)
  add_subdirectory(applications)
endif()
