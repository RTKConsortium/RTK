wrap_ggo(rtkprojectionmatrix_GGO_C rtkprojectionmatrix.ggo ../rtkinputprojections_section.ggo ../rtk3Doutputimage_section.ggo)
add_executable(
  rtkprojectionmatrix
  rtkprojectionmatrix.cxx
  ${rtkprojectionmatrix_GGO_C}
)
target_link_libraries(rtkprojectionmatrix RTK)

set_target_properties(
  rtkprojectionmatrix
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtkprojectionmatrix
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
