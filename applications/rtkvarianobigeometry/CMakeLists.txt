wrap_ggo(rtkvarianobigeometry_GGO_C rtkvarianobigeometry.ggo ../rtkinputprojections_section.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
add_executable(
  rtkvarianobigeometry
  rtkvarianobigeometry.cxx
  ${rtkvarianobigeometry_GGO_C}
)
target_link_libraries(rtkvarianobigeometry RTK)

set_target_properties(
  rtkvarianobigeometry
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtkvarianobigeometry
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
