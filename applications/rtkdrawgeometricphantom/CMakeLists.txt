wrap_ggo(rtkdrawgeometricphantom_GGO_C rtkdrawgeometricphantom.ggo ../rtk3Doutputimage_section.ggo)
add_executable(
  rtkdrawgeometricphantom
  rtkdrawgeometricphantom.cxx
  ${rtkdrawgeometricphantom_GGO_C}
)
target_link_libraries(rtkdrawgeometricphantom ${RTK_APPLICATION_TARGETS})

set_target_properties(
  rtkdrawgeometricphantom
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtkdrawgeometricphantom
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
