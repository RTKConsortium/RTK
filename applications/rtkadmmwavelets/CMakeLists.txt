wrap_ggo(rtkadmmwavelets_GGO_C rtkadmmwavelets.ggo ../rtkinputprojections_section.ggo ../rtk3Doutputimage_section.ggo ../rtkprojectors_section.ggo ../rtkiterations_section.ggo)
add_executable(
  rtkadmmwavelets
  rtkadmmwavelets.cxx
  ${rtkadmmwavelets_GGO_C}
)
target_link_libraries(rtkadmmwavelets ${RTK_APPLICATION_TARGETS})

set_target_properties(
  rtkadmmwavelets
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtkadmmwavelets
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
