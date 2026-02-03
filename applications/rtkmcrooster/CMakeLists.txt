wrap_ggo(rtkmcrooster_GGO_C rtkmcrooster.ggo ../rtkinputprojections_section.ggo ../rtk4Doutputimage_section.ggo)
add_executable(
  rtkmcrooster
  rtkmcrooster.cxx
  ${rtkmcrooster_GGO_C}
)
target_link_libraries(rtkmcrooster ${RTK_APPLICATION_TARGETS})

set_target_properties(
  rtkmcrooster
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtkmcrooster
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
