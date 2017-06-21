cmake_minimum_required ( VERSION 2.8.1 FATAL_ERROR )

project( SimpleRTK_Lua )

include(../../CMake/srtkProjectLanguageCommon.cmake NO_POLICY_SCOPE)


if (CMAKE_VERSION VERSION_LESS "3")
  find_package ( Lua51 ${_QUIET} )
  if ( NOT LUA_FOUND )
    find_package ( Lua50 ${_QUIET} )
  endif()
else()
  find_package ( Lua ${_QUIET} )
endif()

set( LUA_ADDITIONAL_LIBRARIES "" CACHE STRING "Additional libraries which may be needed for lua executable such as readline.")
mark_as_advanced( LUA_ADDITIONAL_LIBRARIES )

include_directories ( ${LUA_INCLUDE_DIR} )

set_source_files_properties ( SimpleRTK.i PROPERTIES CPLUSPLUS ON )

# Run swig
set(CMAKE_SWIG_FLAGS ${CMAKE_SWIG_GLOBAL_FLAGS})
set(CMAKE_SWIG_OUTDIR ${CMAKE_CURRENT_BINARY_DIR})
set(SWIG_MODULE_SimpleRTKLua_EXTRA_DEPS ${SWIG_EXTRA_DEPS}
  ${CMAKE_CURRENT_SOURCE_DIR}/Lua.i )
SWIG_module_initialize ( SimpleRTKLua lua )
SWIG_add_source_to_module ( SimpleRTKLua  swig_generated_source SimpleRTK.i ${SWIG_EXTRA_DEPS} )
set_source_files_properties(${swig_generated_file_fullname} PROPERTIES COMPILE_FLAGS "-w" )

# Create a standalone Lua executable that includes SimpleRTK.
option( SRTK_LUA_BUILD_EXECUTABLE "Build a standalone SimpleRTK-Lua executable" OFF)
if ( SRTK_LUA_BUILD_EXECUTABLE )
  add_executable ( SimpleRTKLua SimpleRTKLuaMain.cxx ${swig_generated_file_fullname} )
  target_link_libraries ( SimpleRTKLua ${SimpleRTK_LIBRARIES} ${LUA_LIBRARIES} ${LUA_ADDITIONAL_LIBRARIES} )
  srtk_strip_target( SimpleRTKLua )
endif ()

# Create a SimpleRTK Lua module that can be loaded into any Lua executable.
SWIG_add_module( SimpleRTKLuaModule lua SimpleRTK.i )
target_link_libraries( ${SWIG_MODULE_SimpleRTKLuaModule_TARGET_NAME} ${SimpleRTK_LIBRARIES} )
set_target_properties( ${SWIG_MODULE_SimpleRTKLuaModule_TARGET_NAME} PROPERTIES PREFIX "" OUTPUT_NAME  "SimpleRTK" )
srtk_target_link_libraries_with_dynamic_lookup( ${SWIG_MODULE_SimpleRTKLuaModule_TARGET_NAME} ${LUA_LIBRARIES} )
set_source_files_properties( ${swig_generated_file_fullname} PROPERTIES COMPILE_FLAGS "-w")
srtk_strip_target( ${SWIG_MODULE_SimpleRTKLuaModule_TARGET_NAME} )

