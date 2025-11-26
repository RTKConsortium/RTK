wrap_ggo(rtkwaveletsdenoising_GGO_C rtkwaveletsdenoising.ggo)
add_executable(
  rtkwaveletsdenoising
  rtkwaveletsdenoising.cxx
  ${rtkwaveletsdenoising_GGO_C}
)
target_link_libraries(rtkwaveletsdenoising RTK)

set_target_properties(
  rtkwaveletsdenoising
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtkwaveletsdenoising
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
