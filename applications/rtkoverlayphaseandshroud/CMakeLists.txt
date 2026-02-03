wrap_ggo(rtkoverlayphaseandshroud_GGO_C rtkoverlayphaseandshroud.ggo)
add_executable(
  rtkoverlayphaseandshroud
  rtkoverlayphaseandshroud.cxx
  ${rtkoverlayphaseandshroud_GGO_C}
)
target_link_libraries(rtkoverlayphaseandshroud ${RTK_APPLICATION_TARGETS})

set_target_properties(
  rtkoverlayphaseandshroud
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtkoverlayphaseandshroud
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
