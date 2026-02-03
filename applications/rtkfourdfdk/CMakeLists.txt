wrap_ggo(rtkfourdfdk_GGO_C rtkfourdfdk.ggo ../rtkinputprojections_section.ggo ../rtk4Doutputimage_section.ggo)
add_executable(
  rtkfourdfdk
  rtkfourdfdk.cxx
  ${rtkfourdfdk_GGO_C}
)
target_link_libraries(rtkfourdfdk ${RTK_APPLICATION_TARGETS})

set_target_properties(
  rtkfourdfdk
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtkfourdfdk
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
