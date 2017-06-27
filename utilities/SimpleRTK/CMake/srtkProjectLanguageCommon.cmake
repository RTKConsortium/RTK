
foreach(p
    CMP0042 # CMake 3.0
    CMP0063 # CMake 3.3.2
    )
  if(POLICY ${p})
    cmake_policy(SET ${p} NEW)
  endif()
endforeach()


#
# Project setup
#

if (NOT CMAKE_PROJECT_NAME STREQUAL "SimpleRTK" )

  set( SimpleRTK_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." )
  list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_LIST_DIR}")

  find_package(SimpleRTK REQUIRED)
  include(${SimpleRTK_USE_FILE})

  # Add compiler flags needed to use SimpleRTK.
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SimpleRTK_REQUIRED_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SimpleRTK_REQUIRED_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${SimpleRTK_REQUIRED_LINK_FLAGS}")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${SimpleRTK_REQUIRED_LINK_FLAGS}")
  set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${SimpleRTK_REQUIRED_LINK_FLAGS}")

endif()


# Setup build locations to the wrapping language sub directories
if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY OR CMAKE_PROJECT_NAME STREQUAL "SimpleRTK" )
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
endif()
if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY OR CMAKE_PROJECT_NAME STREQUAL "SimpleRTK" )
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
endif()
if(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY OR CMAKE_PROJECT_NAME STREQUAL "SimpleRTK" )
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
endif()


# Since most language libraries are not installed with CMake, the
# RPATH does not get fixed up during installation. So skip the RPATH
if(NOT DEFINED CMAKE_SKIP_BUILD_RPATH)
  set(CMAKE_SKIP_BUILD_RPATH 1)
endif()

if(NOT TARGET dist)
  add_custom_target( dist cmake -E echo "Finished generating wrapped packages for distribution..." )
endif()

# TODO these should be moved into UseSimpleRTK
if(NOT SimpleRTK_DOC_FILES)
  set ( SimpleRTK_DOC_FILES
    "${SimpleRTK_SOURCE_SOURCE_DIR}/LICENSE"
    "${SimpleRTK_SOURCE_SOURCE_DIR}/NOTICE"
    "${SimpleRTK_SOURCE_SOURCE_DIR}/Readme.md"
  )
endif()

#
# General SWIG configuration
#

find_package ( SWIG 2 REQUIRED )

include (srtkUseSWIG)

set(SimpleRTK_WRAPPING_COMMON_DIR
  ${SimpleRTK_SOURCE_DIR}/Wrapping/Common)

if ( CMAKE_PROJECT_NAME STREQUAL "SimpleRTK" )
  file(GLOB SWIG_EXTRA_DEPS
    "${SimpleRTK_SOURCE_DIR}/Code/Common/include/*.h"
    "${SimpleRTK_SOURCE_DIR}/Code/Registration/include/*.h"
    "${SimpleRTK_SOURCE_DIR}/Code/IO/include/*.h")
  list( APPEND SWIG_EXTRA_DEPS
    "${SimpleRTK_BINARY_DIR}/Code/BasicFilters/include/SimpleRTKBasicFiltersGeneratedHeaders.h"
    ${SimpleRTKBasicFiltersGeneratedHeader} )
else()
  find_file( _file
    NAMES SimpleRTKBasicFiltersGeneratedHeaders.h
    PATHS ${SimpleRTK_INCLUDE_DIRS}
    NO_DEFAULT_PATH )
  list( APPEND SWIG_EXTRA_DEPS ${_file} )
endif()

# make a manual list of dependencies for the Swig.i files
list( APPEND SWIG_EXTRA_DEPS
  "${SimpleRTK_WRAPPING_COMMON_DIR}/SimpleRTK_Common.i"
  )

# check if uint64_t is the same as unsigned long
try_compile(SRTK_ULONG_SAME_AS_UINT64
  ${PROJECT_BINARY_DIR}/CMakeTmp
  ${SimpleRTK_SOURCE_DIR}/CMake/same_uint64_ulong.cxx )

# when "-DSWIGWORDSIZE64" is defined SWIG used unsigned long for uint64_t types
if(${SRTK_ULONG_SAME_AS_UINT64} )
  set ( CMAKE_SWIG_GLOBAL_FLAGS "-DSWIGWORDSIZE64" )
endif()

set ( CMAKE_SWIG_GLOBAL_FLAGS -I${SimpleRTK_WRAPPING_COMMON_DIR} ${CMAKE_SWIG_GLOBAL_FLAGS} )

include(srtkTargetLinkLibrariesWithDynamicLookup)
include(srtkStripOption)
include(srtkForbidDownloadsOption)
