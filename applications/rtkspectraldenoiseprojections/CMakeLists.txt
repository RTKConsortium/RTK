wrap_ggo(rtkspectraldenoiseprojections_GGO_C rtkspectraldenoiseprojections.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
add_executable(
  rtkspectraldenoiseprojections
  rtkspectraldenoiseprojections.cxx
  ${rtkspectraldenoiseprojections_GGO_C}
)
target_link_libraries(rtkspectraldenoiseprojections RTK)

set_target_properties(
  rtkspectraldenoiseprojections
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${RTK_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtkspectraldenoiseprojections
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
