wrap_ggo(rtkfdk_GGO_C rtkfdk.ggo ../rtkinputprojections_section.ggo ../rtk3Doutputimage_section.ggo)
add_executable(
  rtkfdk
  rtkfdk.cxx
  ${rtkfdk_GGO_C}
)
target_link_libraries(rtkfdk RTK)

set_target_properties(
  rtkfdk
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtkfdk
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
