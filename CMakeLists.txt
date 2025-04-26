#=========================================================
# RTK : Reconstruction Toolkit
#=========================================================
# Respect the CMAKE_CXX_STANDARD flags when building for
# ITKv5 or C++11.
cmake_minimum_required(VERSION 3.16.3 FATAL_ERROR)

## Only policies introduced after the cmake_minimum_required
## version need to explicitly be set to NEW.
## Refer to https://cmake.org/cmake/help/v3.11/manual/cmake-policies.7.html
set(CMAKE_POLICIES
  CMP0135)

foreach(p ${CMAKE_POLICIES})
  if(POLICY ${p})
    cmake_policy(SET ${p} NEW)
  endif()
endforeach()

#=========================================================
# Help function to debug CMake
macro (DD in)
  message(${in}=[${${in}}])
endmacro()
#=========================================================

project(RTK)

## RTK Version
set(RTK_VERSION_MAJOR "2")
set(RTK_VERSION_MINOR "7")
set(RTK_VERSION_PATCH "0")
set(RTK_VERSION_STRING "${RTK_VERSION_MAJOR}.${RTK_VERSION_MINOR}")

set(RTK_LIBRARIES RTK)

#=========================================================
# Installation variables
#=========================================================
if(NOT CMAKE_INSTALL_LIBDIR)
  set(CMAKE_INSTALL_LIBDIR lib)
endif()
if(NOT RTK_INSTALL_RUNTIME_DIR)
  set(RTK_INSTALL_RUNTIME_DIR bin)
endif()
if(NOT RTK_INSTALL_LIB_DIR)
  set(RTK_INSTALL_LIB_DIR ${CMAKE_INSTALL_LIBDIR})
endif()
if(NOT RTK_INSTALL_ARCHIVE_DIR)
  set(RTK_INSTALL_ARCHIVE_DIR ${CMAKE_INSTALL_LIBDIR})
endif()
if(NOT RTK_INSTALL_INCLUDE_DIR)
  set(RTK_INSTALL_INCLUDE_DIR include/RTK)
endif()
if(NOT RTK_INSTALL_PACKAGE_DIR)
  set(RTK_INSTALL_PACKAGE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/RTK")
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

include(cmake/rtkCompilerFlags.cmake)

# --------------------------------------------------------
# Find ITK (required)
if(NOT ITK_SOURCE_DIR)
  find_package(ITK 5.1 REQUIRED)
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

  if(RTK_BUILD_SHARED_LIBS)
    set(ITK_LIBRARY_BUILD_TYPE "SHARED")
  else()
    set(ITK_LIBRARY_BUILD_TYPE "STATIC")
  endif()
endif()

#=========================================================
# If choose to build documentation, then search for Doxygen/Sphinx executables.
option(RTK_BUILD_DOXYGEN "Build Doxygen Documentation" OFF)
if(RTK_BUILD_DOXYGEN)
  add_subdirectory(documentation/Doxygen)
endif()

option(RTK_BUILD_SPHINX "Build Sphinx Documentation" OFF)
if(RTK_BUILD_SPHINX)
  add_subdirectory(documentation/docs)
endif()

# Setup build locations.
if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${RTK_BINARY_DIR}/bin)
endif()
if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${RTK_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
endif()
if(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
  # Python builds CI workflows expects archives to be placed in the ITK lib
  # directory. This makes sure that other remote modules that depends on RTK
  # can link against the RTK libraries.
  if (WRAP_ITK_INSTALL_COMPONENT_IDENTIFIER STREQUAL "PythonWheel")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${ITK_DIR}/${CMAKE_INSTALL_LIBDIR})
  else()
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${RTK_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
  endif()
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

#=========================================================
# Include directories
#=========================================================
list(APPEND RTK_INCLUDE_DIRS
  ${RTK_BINARY_DIR}/include)

#=========================================================
# Generate RTKConfig.cmake for the build tree.
set(RTK_MODULE_PATH_CONFIG ${CMAKE_MODULE_PATH})

# Add option to control the generation of RTK applications
option(RTK_BUILD_APPLICATIONS "Build RTK applications" ON)

set(RTK_EXPORT_CODE_BUILD "
# The RTK version number
set(RTK_VERSION_MAJOR ${RTK_VERSION_MAJOR})
set(RTK_VERSION_MINOR ${RTK_VERSION_MINOR})
set(RTK_VERSION_PATCH ${RTK_VERSION_PATCH})

# Whether the compiled version of RTK uses CUDA
set(RTK_USE_CUDA ${RTK_USE_CUDA})

# Whether RTK applications were built
set(RTK_BUILD_APPLICATIONS ${RTK_BUILD_APPLICATIONS})

if(RTK_USE_CUDA)
  find_package(CUDAToolkit)
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

# Whether RTK applications were built
set(RTK_BUILD_APPLICATIONS ${RTK_BUILD_APPLICATIONS})

if(RTK_USE_CUDA)
  find_package(CUDAToolkit)
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

  # Add third party to RTK build targets.
  # This must be done after RTK has been loaded by ITK to make sure
  # ${itk-module} variables are defined for RTK.
  itk_module_target(lpsolve55)

  if(${ITK_VERSION} VERSION_LESS 5.3)
    ## Set the default target properties for RTK
    if(NOT CMAKE_CXX_STANDARD)
      set(CMAKE_CXX_STANDARD 14) # Supported values are ``11``, ``14``, and ``17``.
    endif()
    if(NOT CMAKE_CXX_STANDARD_REQUIRED)
      set(CMAKE_CXX_STANDARD_REQUIRED ON)
    endif()
    if(NOT CMAKE_CXX_EXTENSIONS)
      set(CMAKE_CXX_EXTENSIONS OFF)
    endif()

    ## Default to release
    if (NOT CMAKE_BUILD_TYPE)
      set(CMAKE_BUILD_TYPE Release)
    endif()
  endif()
else()
  itk_module_impl()
endif()

# Propagate cmake options in a header file
# Must be done after the external module configuration to make sure CudaCommon_VERSION is defined
configure_file(${RTK_SOURCE_DIR}/rtkConfiguration.h.in
  ${RTK_BINARY_DIR}/include/rtkConfiguration.h)

# Install lpsolve headers
install(FILES ${RTK_SOURCE_DIR}/utilities/lp_solve/lp_bit.h
              ${RTK_SOURCE_DIR}/utilities/lp_solve/lp_crash.h
              ${RTK_SOURCE_DIR}/utilities/lp_solve/lp_explicit.h
              ${RTK_SOURCE_DIR}/utilities/lp_solve/lp_fortify.h
              ${RTK_SOURCE_DIR}/utilities/lp_solve/lp_Hash.h
              ${RTK_SOURCE_DIR}/utilities/lp_solve/lp_lib.h
              ${RTK_SOURCE_DIR}/utilities/lp_solve/lp_matrix.h
              ${RTK_SOURCE_DIR}/utilities/lp_solve/lp_MDO.h
              ${RTK_SOURCE_DIR}/utilities/lp_solve/lp_mipbb.h
              ${RTK_SOURCE_DIR}/utilities/lp_solve/lp_MPS.h
              ${RTK_SOURCE_DIR}/utilities/lp_solve/lp_presolve.h
              ${RTK_SOURCE_DIR}/utilities/lp_solve/lp_price.h
              ${RTK_SOURCE_DIR}/utilities/lp_solve/lp_pricePSE.h
              ${RTK_SOURCE_DIR}/utilities/lp_solve/lp_report.h
              ${RTK_SOURCE_DIR}/utilities/lp_solve/lp_rlp.h
              ${RTK_SOURCE_DIR}/utilities/lp_solve/lp_scale.h
              ${RTK_SOURCE_DIR}/utilities/lp_solve/lp_simplex.h
              ${RTK_SOURCE_DIR}/utilities/lp_solve/lp_solveDLL.h
              ${RTK_SOURCE_DIR}/utilities/lp_solve/lp_SOS.h
              ${RTK_SOURCE_DIR}/utilities/lp_solve/lp_types.h
              ${RTK_SOURCE_DIR}/utilities/lp_solve/lp_utils.h
              ${RTK_SOURCE_DIR}/utilities/lp_solve/lp_wlp.h
        DESTINATION ${RTK_INSTALL_INCLUDE_DIR}/lpsolve
        COMPONENT Development)
target_include_directories(lpsolve55 PUBLIC $<INSTALL_INTERFACE:${RTK_INSTALL_INCLUDE_DIR}/lpsolve>)

# Install configuration file
install(FILES ${RTK_BINARY_DIR}/include/rtkConfiguration.h DESTINATION ${RTK_INSTALL_INCLUDE_DIR})
install(FILES ${RTK_SOURCE_DIR}/cmake/FindGengetopt.cmake
              ${RTK_SOURCE_DIR}/cmake/GetGitRevisionDescription.cmake
              ${RTK_SOURCE_DIR}/cmake/GetGitRevisionDescription.cmake.in
        DESTINATION ${ITK_INSTALL_PACKAGE_DIR})
if(NOT ITK_SOURCE_DIR)
  install(CODE "MESSAGE(FATAL_ERROR \"Cannot install, RTK is compiled separately from ITK. Installation is only functional if RTK is compiled within ITK.\")")
endif()

#=========================================================
# Build applications
#=========================================================
option(RTK_PROBE_EACH_FILTER "Probe each RTK filter in a global object and report times and memory used in RTK applications" OFF)

if(RTK_BUILD_APPLICATIONS)
  add_subdirectory(applications)
endif()

# --------------------------------------------------------
# Setup KWStyle from ITK
if(ITK_USE_KWSTYLE)
  find_package( Git )
  if(GIT_FOUND AND EXISTS "${RTK_SOURCE_DIR}/.git/config")
    execute_process( COMMAND ${GIT_EXECUTABLE} config hooks.KWStyle.path
      "${KWSTYLE_EXECUTABLE}"
      WORKING_DIRECTORY ${RTK_SOURCE_DIR} )
  endif()
endif()

# --------------------------------------------------------
# Setup ClangFormat from ITK
if(ITK_USE_CLANGFORMAT)
  find_package( Git )
  if(GIT_FOUND AND EXISTS "${RTK_SOURCE_DIR}/.git/config")
    execute_process(COMMAND ${GIT_EXECUTABLE} config clangFormat.binary
      "${CLANG_FORMAT_EXECUTABLE}"
      WORKING_DIRECTORY ${RTK_SOURCE_DIR})
  endif()
endif()

#=========================================================
# Install pre-commit hook
#=========================================================
if(EXISTS "${RTK_SOURCE_DIR}/.git/config" AND
    NOT EXISTS "${RTK_SOURCE_DIR}/.git/hooks/pre-commit")
  # Silently ignore the error if the hooks directory is read-only.
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy ${RTK_SOURCE_DIR}/cmake/Hooks/pre-commit
                                     ${RTK_SOURCE_DIR}/.git/hooks/pre-commit
    OUTPUT_VARIABLE _output
    ERROR_VARIABLE  _output
    RESULT_VARIABLE _result
    )
  if(_result AND NOT "${_output}" MATCHES "Error copying file")
    message("${_output}")
  endif()
endif()
