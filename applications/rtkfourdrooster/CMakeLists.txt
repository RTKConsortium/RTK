wrap_ggo(rtkfourdrooster_GGO_C rtkfourdrooster.ggo ../rtkinputprojections_section.ggo ../rtk4Doutputimage_section.ggo ../rtkprojectors_section.ggo ../rtkiterations_section.ggo)
add_executable(
  rtkfourdrooster
  rtkfourdrooster.cxx
  ${rtkfourdrooster_GGO_C}
)
target_link_libraries(rtkfourdrooster RTK)

set_target_properties(
  rtkfourdrooster
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtkfourdrooster
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
