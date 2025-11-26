wrap_ggo(rtkfieldofview_GGO_C rtkfieldofview.ggo ../rtkinputprojections_section.ggo)
add_executable(
  rtkfieldofview
  rtkfieldofview.cxx
  ${rtkfieldofview_GGO_C}
)
target_link_libraries(rtkfieldofview RTK)

set_target_properties(
  rtkfieldofview
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtkfieldofview
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
