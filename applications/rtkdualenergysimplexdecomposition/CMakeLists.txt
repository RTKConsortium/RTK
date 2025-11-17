wrap_ggo(rtkdualenergysimplexdecomposition_GGO_C rtkdualenergysimplexdecomposition.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
add_executable(
  rtkdualenergysimplexdecomposition
  rtkdualenergysimplexdecomposition.cxx
  ${rtkdualenergysimplexdecomposition_GGO_C}
)
target_link_libraries(rtkdualenergysimplexdecomposition RTK)

set_target_properties(
  rtkdualenergysimplexdecomposition
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtkdualenergysimplexdecomposition
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
