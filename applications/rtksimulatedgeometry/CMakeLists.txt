wrap_ggo(rtksimulatedgeometry_GGO_C rtksimulatedgeometry.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
add_executable(
  rtksimulatedgeometry
  rtksimulatedgeometry.cxx
  ${rtksimulatedgeometry_GGO_C}
)
target_link_libraries(rtksimulatedgeometry RTK)

set_target_properties(
  rtksimulatedgeometry
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtksimulatedgeometry
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
