cmake_minimum_required ( VERSION 2.8.1 FATAL_ERROR )

project( SimpleRTK_TCL )

include(../../CMake/srtkProjectLanguageCommon.cmake NO_POLICY_SCOPE)

find_package ( TCL REQUIRED )
include_directories ( ${TCL_INCLUDE_PATH} )

set_source_files_properties ( SimpleRTK.i PROPERTIES CPLUSPLUS ON )

# Run swig
set(CMAKE_SWIG_FLAGS "-nosafe" ${CMAKE_SWIG_GLOBAL_FLAGS})
set(CMAKE_SWIG_OUTDIR ${CMAKE_CURRENT_BINARY_DIR})
set(SWIG_MODULE_SimpleRTKTCL_EXTRA_DEPS ${SWIG_EXTRA_DEPS} ${CMAKE_CURRENT_SOURCE_DIR}/Tcl.i )
set(SWIG_MODULE_SimpleRTKTcl_EXTRA_DEPS  ${SWIG_MODULE_SimpleRTKTCL_EXTRA_DEPS})
#  SWIG_add_module ( SimpleRTKTcl tcl SimpleRTK.i SimpleRTKTCL_wrap.cxx )
#  target_link_libraries ( ${SWIG_MODULE_SimpleRTKTcl_TARGET_NAME} ${SimpleRTK_LIBRARIES} )
#  srtk_target_link_libraries_with_dynamic_lookup( ${SWIG_MODULE_SimpleRTKTcl_TARGET_NAME} ${TCL_LIBRARY} )
#  set_source_files_properties(${swig_generated_file_fullname}  PROPERTIES COMPILE_FLAGS "-w")
#  add_executable ( SimpleRTKTclsh ${swig_generated_file_fullname} )
#  target_link_libraries ( SimpleRTKTclsh ${SimpleRTK_LIBRARIES} ${TCL_LIBRARY})

SWIG_module_initialize ( SimpleRTKTCL tcl )
SWIG_add_source_to_module ( SimpleRTKTCL swig_generated_source SimpleRTK.i ${SWIG_EXTRA_DEPS} )
set_source_files_properties(${swig_generated_file_fullname} PROPERTIES COMPILE_FLAGS "-w" )
add_executable ( SimpleRTKTclsh ${swig_generated_file_fullname} )
target_link_libraries ( SimpleRTKTclsh ${SimpleRTK_LIBRARIES} ${TCL_LIBRARY} )
srtk_strip_target( SimpleRTKTclsh )
