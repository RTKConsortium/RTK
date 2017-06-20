
#-----------------------------------------------------------------------------
# Get and build RTK

get_cmake_property( _varNames VARIABLES )

foreach (_varName ${_varNames})
  if(_varName MATCHES "^RTK_" OR _varName MATCHES "FFTW")
    message( STATUS "Passing variable \"${_varName}=${${_varName}}\" to RTK external project.")
    list(APPEND RTK_VARS ${_varName})
  endif()
endforeach()

list(APPEND RTK_VARS
  PYTHON_EXECUTABLE
  PYTHON_INCLUDE_DIR
  PYTHON_LIBRARY
  PYTHON_DEBUG_LIBRARY
  )

VariableListToCache( RTK_VARS  ep_rtk_cache )
VariableListToArgs( RTK_VARS  ep_rtk_args )


set(proj RTK)  ## Use RTK convention of calling it RTK
set(RTK_REPOSITORY git://github.com/SimonRit/RTK)

# NOTE: it is very important to update the RTK_DIR path with the RTK version
# For now using master/head
#set(RTK_TAG_COMMAND GIT_TAG v4.5.1)

if( ${BUILD_SHARED_LIBS} )
  set( RTK_BUILD_SHARED_LIBS ON )
else()
  set( RTK_BUILD_SHARED_LIBS OFF )
endif()


file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${proj}-build/CMakeCacheInit.txt" "${ep_rtk_cache}\n${ep_common_cache}" )

ExternalProject_Add(${proj}
  GIT_REPOSITORY ${RTK_REPOSITORY}
  ${RTK_TAG_COMMAND}
  UPDATE_COMMAND ""
  SOURCE_DIR ${proj}
  BINARY_DIR ${proj}-build
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
  --no-warn-unused-cli
  -C "${CMAKE_CURRENT_BINARY_DIR}/${proj}-build/CMakeCacheInit.txt"
  ${ep_rtk_args}
  ${ep_common_args}
  -DRTK_DIR:PATH=${RTK_DIR}
  -DBUILD_APPLICATIONS:BOOL=OFF
  -DBUILD_EXAMPLES:BOOL=OFF
  -DBUILD_TESTING:BOOL=OFF
  -DBUILD_SHARED_LIBS:BOOL=${RTK_BUILD_SHARED_LIBS}
  -DCMAKE_SKIP_RPATH:BOOL=ON
  -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
  BUILD_COMMAND ${BUILD_COMMAND_STRING}
  DEPENDS
  ${RTK_DEPENDENCIES}
  )

ExternalProject_Get_Property(RTK install_dir)
set(RTK_DIR "${install_dir}/lib/cmake/RTK-Master" )
