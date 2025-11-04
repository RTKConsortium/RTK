wrap_ggo(rtkdigisensgeometry_GGO_C rtkdigisensgeometry.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
add_executable(
  rtkdigisensgeometry
  rtkdigisensgeometry.cxx
  ${rtkdigisensgeometry_GGO_C}
)
target_link_libraries(rtkdigisensgeometry RTK)

set_target_properties(
  rtkdigisensgeometry
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${RTK_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtkdigisensgeometry
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
