wrap_ggo(rtktutorialapplication_GGO_C rtktutorialapplication.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
add_executable(
  rtktutorialapplication
  rtktutorialapplication.cxx
  ${rtktutorialapplication_GGO_C}
)
target_link_libraries(rtktutorialapplication RTK)

set_target_properties(
  rtktutorialapplication
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtktutorialapplication
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
