cmake_minimum_required ( VERSION 2.8.1 FATAL_ERROR )

project( SimpleRTK_Lua )

include(../../CMake/srtkProjectLanguageCommon.cmake NO_POLICY_SCOPE)

find_package( Ruby REQUIRED )
include_directories( ${RUBY_INCLUDE_DIRS} )

set_source_files_properties( SimpleRTK.i PROPERTIES CPLUSPLUS ON )

# Run swig
set(CMAKE_SWIG_FLAGS -autorename -module simpleitk ${CMAKE_SWIG_GLOBAL_FLAGS})
set(CMAKE_SWIG_OUTDIR ${CMAKE_CURRENT_BINARY_DIR})
set(SWIG_MODULE_simplertk_EXTRA_DEPS ${SWIG_EXTRA_DEPS}
  ${CMAKE_CURRENT_SOURCE_DIR}/Ruby.i)

SWIG_add_module( simpleitk ruby SimpleRTK.i  )
target_link_libraries( ${SWIG_MODULE_simplertk_TARGET_NAME} ${SimpleRTK_LIBRARIES} )
srtk_target_link_libraries_with_dynamic_lookup( ${SWIG_MODULE_simplertk_TARGET_NAME} ${RUBY_LIBRARY} )
set_source_files_properties( ${swig_generated_file_fullname} PROPERTIES COMPILE_FLAGS "-w")
srtk_strip_target( ${SWIG_MODULE_simplertk_TARGET_NAME} )
