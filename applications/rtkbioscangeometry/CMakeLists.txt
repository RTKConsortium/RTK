wrap_ggo(rtkbioscangeometry_GGO_C rtkbioscangeometry.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
add_executable(
  rtkbioscangeometry
  rtkbioscangeometry.cxx
  ${rtkbioscangeometry_GGO_C}
)
target_link_libraries(rtkbioscangeometry RTK)

set_target_properties(
  rtkbioscangeometry
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${RTK_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  foreach(EXE_NAME rtkbioscangeometry)
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
