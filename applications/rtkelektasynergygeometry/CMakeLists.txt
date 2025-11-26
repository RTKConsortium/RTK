wrap_ggo(rtkelektasynergygeometry_GGO_C rtkelektasynergygeometry.ggo)
add_executable(
  rtkelektasynergygeometry
  rtkelektasynergygeometry.cxx
  ${rtkelektasynergygeometry_GGO_C}
)
target_link_libraries(rtkelektasynergygeometry RTK)

set_target_properties(
  rtkelektasynergygeometry
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtkelektasynergygeometry
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
