cmake_minimum_required ( VERSION 2.8.11 )

#includes the language options
set(CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_LIST_DIR}/CMake
  ${CMAKE_BINARY_DIR}/CMake
  ${CMAKE_CURRENT_LIST_DIR}/Wrapping
  ${CMAKE_CURRENT_LIST_DIR}/SuperBuild
  ${CMAKE_MODULE_PATH}
  )
  
include(srtkPreventInSourceBuilds)
include(srtkPreventInBuildInstalls)
include(VariableList)
include(srtkExternalData)
include(ExternalProject)

add_custom_target( SuperBuildSimpleRTKSource )

#
# srtkSourceDownload( <output variable> <filename> <md5 hash> )
#
# A function to get a filename for an ExternalData source file used in
# a superbuild. Adds a target which downloads all source code
# needed for superbuild projects. The source file is cached with in
# the build tree, and can be locally cache with other ExternalData
# controlled environment variables.
#
# The "SuperBuildSimpleRTKSource" target needs to be manually added as
# a dependencies to the ExternalProject.
#
#   add_dependencies( PROJ "SuperBuildSimpleRTKSource" )
#
# Note: Hash files are created under the SOURCE directory in the
# .ExternalSource sub-directory during configuration.
#
function(srtkSourceDownload outVar filename hash)
  set(link_file "${CMAKE_CURRENT_SOURCE_DIR}/.ExternalSource/${filename}")
  file(WRITE  "${link_file}.md5" ${hash} )
  ExternalData_Expand_arguments(
    SuperBuildSimpleRTKSourceReal
    link
    DATA{${link_file}}
    )
  set(${outVar} "${link}" PARENT_SCOPE)
endfunction()

function(srtkSourceDownloadDependency proj)
  if (CMAKE_VERSION VERSION_LESS 3.2)
    add_dependencies(${proj}  "SuperBuildSimpleRTKSource")
  else()
    ExternalProject_Add_StepDependencies(${proj} download "SuperBuildSimpleRTKSource")
  endif()
endfunction()

include(srtkLanguageOptions)

#------------------------------------------------------------------------------
# Lua
#------------------------------------------------------------------------------
option ( USE_SYSTEM_LUA "Use a pre-compiled version of LUA 5.1 previously configured for your system" OFF )
mark_as_advanced(USE_SYSTEM_LUA)
if ( USE_SYSTEM_LUA )
  find_package( LuaInterp REQUIRED 5.1 )
  set( SRTK_LUA_EXECUTABLE ${LUA_EXECUTABLE} CACHE PATH "Lua executable used for code generation." )
  mark_as_advanced( SRTK_LUA_EXECUTABLE )
  unset( LUA_EXECUTABLE CACHE )
else()
  include(External_Lua)
  list(APPEND ${CMAKE_PROJECT_NAME}_DEPENDENCIES Lua)
  set( SRTK_LUA_EXECUTABLE ${SRTK_LUA_EXECUTABLE} CACHE PATH "Lua executable used for code generation." )
  mark_as_advanced( SRTK_LUA_EXECUTABLE )
  list(APPEND SimpleRTK_VARS SRTK_LUA_EXECUTABLE)
endif()

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

#------------------------------------------------------------------------------
# Google Test
#------------------------------------------------------------------------------
option( USE_SYSTEM_GTEST "Use a pre-compiled version of GoogleTest. " OFF )
mark_as_advanced(USE_SYSTEM_GTEST)
if ( BUILD_TESTING )
  if (USE_SYSTEM_GTEST)
    find_package( GTest REQUIRED )
    list(APPEND SimpleRTK_VARS GTEST_LIBRARIES GTEST_INCLUDE_DIRS GTEST_MAIN_LIBRARIES)
  else()
    include(External_GTest)
    set( GTEST_ROOT ${GTEST_ROOT} )
    list(APPEND SimpleRTK_VARS GTEST_ROOT)
    list(APPEND ${CMAKE_PROJECT_NAME}_DEPENDENCIES GTest)
  endif()
endif()

# We need python virtualenv
#------------------------------------------------------------------------------
# Python virtualenv
#------------------------------------------------------------------------------
option( USE_SYSTEM_VIRTUALENV "Use a system version of Python's virtualenv. " OFF )
mark_as_advanced(USE_SYSTEM_VIRTUALENV)
if( NOT DEFINED SRTK_PYTHON_USE_VIRTUALENV OR SRTK_PYTHON_USE_VIRTUALENV )
  if ( USE_SYSTEM_VIRTUALENV )
    find_package( PythonVirtualEnv REQUIRED)
  else()
    include(External_virtualenv)
    if ( WRAP_PYTHON )
      list(APPEND ${CMAKE_PROJECT_NAME}_DEPENDENCIES virtualenv)
    endif()
  endif()
  list(APPEND SimpleRTK_VARS PYTHON_VIRTUALENV_SCRIPT)
endif()


VariableListToCache( SimpleRTK_VARS  ep_simplertk_cache )
VariableListToArgs( SimpleRTK_VARS  ep_simplertk_args )
VariableListToCache( SRTK_LANGUAGES_VARS  ep_languages_cache )
VariableListToArgs( SRTK_LANGUAGES_VARS  ep_languages_args )

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/SimpleRTK-build/CMakeCacheInit.txt" "${ep_simplertk_cache}${ep_common_cache}\n${ep_languages_cache}" )

set(proj SimpleRTK)
ExternalProject_Add(${proj}
  DOWNLOAD_COMMAND ""
  SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}
  BINARY_DIR SimpleRTK-build
#  INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    --no-warn-unused-cli
    -C "${CMAKE_CURRENT_BINARY_DIR}/SimpleRTK-build/CMakeCacheInit.txt"
    ${ep_simplertk_args}
    ${ep_common_args}
    -DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED_LIBS}
    -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}\ ${CXX_ADDITIONAL_WARNING_FLAGS}
	-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
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
	-DSRTK_4D_IMAGES:BOOL=ON
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
	-DSimpleRTK_PYTHON_WHEEL=1
  DEPENDS ${${CMAKE_PROJECT_NAME}_DEPENDENCIES} RTK
)



if(COMMAND ExternalData_Add_Target)
  ExternalData_Add_Target(SuperBuildSimpleRTKSourceReal)
  add_dependencies(SuperBuildSimpleRTKSource SuperBuildSimpleRTKSourceReal)
endif()


