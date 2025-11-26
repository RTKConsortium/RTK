wrap_ggo(rtkprojectshepploganphantom_GGO_C rtkprojectshepploganphantom.ggo ../rtk3Doutputimage_section.ggo)
add_executable(
  rtkprojectshepploganphantom
  rtkprojectshepploganphantom.cxx
  ${rtkprojectshepploganphantom_GGO_C}
)
target_link_libraries(rtkprojectshepploganphantom RTK)

set_target_properties(
  rtkprojectshepploganphantom
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtkprojectshepploganphantom
    RUNTIME
      DESTINATION ${RTK_INSTALL_RUNTIME_DIR}
      COMPONENT Runtime
    LIBRARY
      DESTINATION ${RTK_INSTALL_LIB_DIR}
      COMPONENT RuntimeLibraries
    ARCHIVE
      DESTINATION ${RTK_INSTALL_ARCHIVE_DIR}
      COMPONENT Development
  )
endif()
