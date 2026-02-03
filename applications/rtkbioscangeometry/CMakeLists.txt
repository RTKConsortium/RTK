wrap_ggo(rtkbioscangeometry_GGO_C rtkbioscangeometry.ggo)
add_executable(
  rtkbioscangeometry
  rtkbioscangeometry.cxx
  ${rtkbioscangeometry_GGO_C}
)
target_link_libraries(rtkbioscangeometry ${RTK_APPLICATION_TARGETS})

set_target_properties(
  rtkbioscangeometry
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtkbioscangeometry
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
