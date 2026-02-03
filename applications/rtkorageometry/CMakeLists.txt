wrap_ggo(rtkorageometry_GGO_C rtkorageometry.ggo)
add_executable(
  rtkorageometry
  rtkorageometry.cxx
  ${rtkorageometry_GGO_C}
)
target_link_libraries(rtkorageometry ${RTK_APPLICATION_TARGETS})

set_target_properties(
  rtkorageometry
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtkorageometry
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
