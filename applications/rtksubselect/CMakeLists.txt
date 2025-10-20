wrap_ggo(rtksubselect_GGO_C rtksubselect.ggo ../rtkinputprojections_section.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
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
      "${RTK_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  foreach(EXE_NAME rtksubselect)
    install(
      TARGETS
        ${EXE_NAME}
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
  endforeach()
endif()
