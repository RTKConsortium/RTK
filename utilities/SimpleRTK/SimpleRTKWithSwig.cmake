cmake_minimum_required ( VERSION 2.8.4 )

#includes the language options
set(CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_LIST_DIR}/CMake
  ${CMAKE_BINARY_DIR}/CMake
  ${CMAKE_CURRENT_LIST_DIR}/Wrapping
  ${CMAKE_CURRENT_LIST_DIR}/SuperBuild
  ${CMAKE_MODULE_PATH}
  )

include(srtkLanguageOptions)

# We need SWIG
include(ExternalProject)
#------------------------------------------------------------------------------
# Swig
#------------------------------------------------------------------------------
option ( USE_SYSTEM_SWIG "Use a pre-compiled version of SWIG 2.0 previously configured for your system" OFF )
mark_as_advanced(USE_SYSTEM_SWIG)
if(USE_SYSTEM_SWIG)
  find_package ( SWIG 2 REQUIRED )
  include ( UseSWIGLocal )
else()
  include(${CMAKE_CURRENT_LIST_DIR}/SuperBuild/External_Swig.cmake)
  list(APPEND ${CMAKE_PROJECT_NAME}_DEPENDENCIES Swig)
endif()

# We need to add SimpleRTK as an external project depending on SWIG
# so that the configure step happens after swig has been built
ExternalProject_Add(SimpleRTK
  DOWNLOAD_COMMAND ""
  SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}
  BINARY_DIR SimpleRTK-build
#  INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
  CMAKE_ARGS
    --no-warn-unused-cli
    -DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED_LIBS}
    -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}\ ${CXX_ADDITIONAL_WARNING_FLAGS}
    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY:PATH=<BINARY_DIR>/lib
    -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY:PATH=<BINARY_DIR>/lib
    -DCMAKE_RUNTIME_OUTPUT_DIRECTORY:PATH=<BINARY_DIR>/bin
    -DCMAKE_BUNDLE_OUTPUT_DIRECTORY:PATH=<BINARY_DIR>/bin
    ${ep_languages_args}
    # ITK
    -DITK_DIR:PATH=${ITK_DIR}
	# RTK
    -DRTK_DIR:PATH=${RTK_DIR}
    # Swig
    -DSWIG_DIR:PATH=${SWIG_DIR}
    -DSWIG_EXECUTABLE:PATH=${SWIG_EXECUTABLE}
    -DBUILD_TESTING:BOOL=${BUILD_TESTING}
    -DWRAP_LUA:BOOL=${WRAP_LUA}
    -DWRAP_PYTHON:BOOL=${WRAP_PYTHON}
    -DWRAP_RUBY:BOOL=${WRAP_RUBY}
    -DWRAP_JAVA:BOOL=${WRAP_JAVA}
    -DWRAP_TCL:BOOL=${WRAP_TCL}
    -DWRAP_CSHARP:BOOL=${WRAP_CSHARP}
    -DWRAP_R:BOOL=${WRAP_R}
    -DBUILD_EXAMPLES:BOOL=${BUILD_TESTING}
  DEPENDS ${${CMAKE_PROJECT_NAME}_DEPENDENCIES} RTK
)
