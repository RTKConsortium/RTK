wrap_ggo(rtklagcorrection_GGO_C rtklagcorrection.ggo ../rtkinputprojections_section.ggo)
add_executable(
  rtklagcorrection
  rtklagcorrection.cxx
  ${rtklagcorrection_GGO_C}
)
target_link_libraries(rtklagcorrection RTK)

set_target_properties(
  rtklagcorrection
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtklagcorrection
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
