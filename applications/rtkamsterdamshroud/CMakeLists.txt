wrap_ggo(rtkamsterdamshroud_GGO_C rtkamsterdamshroud.ggo ../rtkinputprojections_section.ggo)
add_executable(
  rtkamsterdamshroud
  rtkamsterdamshroud.cxx
  ${rtkamsterdamshroud_GGO_C}
)
target_link_libraries(rtkamsterdamshroud ${RTK_APPLICATION_TARGETS})

set_target_properties(
  rtkamsterdamshroud
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtkamsterdamshroud
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
