wrap_ggo(rtkspectralsimplexdecomposition_GGO_C rtkspectralsimplexdecomposition.ggo)
add_executable(
  rtkspectralsimplexdecomposition
  rtkspectralsimplexdecomposition.cxx
  ${rtkspectralsimplexdecomposition_GGO_C}
)
target_link_libraries(
  rtkspectralsimplexdecomposition
  ${RTK_APPLICATION_TARGETS}
)

set_target_properties(
  rtkspectralsimplexdecomposition
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtkspectralsimplexdecomposition
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
