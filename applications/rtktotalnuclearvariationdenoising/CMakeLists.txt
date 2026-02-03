wrap_ggo(rtktotalnuclearvariationdenoising_GGO_C rtktotalnuclearvariationdenoising.ggo)
add_executable(
  rtktotalnuclearvariationdenoising
  rtktotalnuclearvariationdenoising.cxx
  ${rtktotalnuclearvariationdenoising_GGO_C}
)
target_link_libraries(
  rtktotalnuclearvariationdenoising
  ${RTK_APPLICATION_TARGETS}
)

set_target_properties(
  rtktotalnuclearvariationdenoising
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtktotalnuclearvariationdenoising
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
