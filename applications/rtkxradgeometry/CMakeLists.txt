wrap_ggo(rtkxradgeometry_GGO_C rtkxradgeometry.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
add_executable(
  rtkxradgeometry
  rtkxradgeometry.cxx
  ${rtkxradgeometry_GGO_C}
)
target_link_libraries(rtkxradgeometry RTK)

set_target_properties(
  rtkxradgeometry
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${RTK_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtkxradgeometry
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
