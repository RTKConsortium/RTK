wrap_ggo(rtkspectralonestep_GGO_C rtkspectralonestep.ggo ../rtk3Doutputimage_section.ggo ../rtkprojectors_section.ggo ../rtkiterations_section.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
add_executable(
  rtkspectralonestep
  rtkspectralonestep.cxx
  ${rtkspectralonestep_GGO_C}
)
target_link_libraries(rtkspectralonestep RTK)

set_target_properties(
  rtkspectralonestep
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtkspectralonestep
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
