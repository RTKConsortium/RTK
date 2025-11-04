wrap_ggo(rtkfourdsart_GGO_C rtkfourdsart.ggo ../rtkinputprojections_section.ggo ../rtk4Doutputimage_section.ggo ../rtkprojectors_section.ggo ../rtkiterations_section.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
add_executable(
  rtkfourdsart
  rtkfourdsart.cxx
  ${rtkfourdsart_GGO_C}
)
target_link_libraries(rtkfourdsart RTK)

set_target_properties(
  rtkfourdsart
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${RTK_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtkfourdsart
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
