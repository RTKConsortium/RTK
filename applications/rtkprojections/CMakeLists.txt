wrap_ggo(rtkprojections_GGO_C rtkprojections.ggo ../rtkinputprojections_section.ggo ../rtknoise_section.ggo)
add_executable(
  rtkprojections
  rtkprojections.cxx
  ${rtkprojections_GGO_C}
)
target_link_libraries(rtkprojections RTK)

set_target_properties(
  rtkprojections
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtkprojections
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
