wrap_ggo(rtksubselect_GGO_C rtksubselect.ggo ../rtkinputprojections_section.ggo)
add_executable(
  rtksubselect
  rtksubselect.cxx
  ${rtksubselect_GGO_C}
)
target_link_libraries(rtksubselect RTK)

set_target_properties(
  rtksubselect
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtksubselect
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
