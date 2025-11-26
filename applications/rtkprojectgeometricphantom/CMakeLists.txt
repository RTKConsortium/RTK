wrap_ggo(rtkprojectgeometricphantom_GGO_C rtkprojectgeometricphantom.ggo ../rtk3Doutputimage_section.ggo)
add_executable(
  rtkprojectgeometricphantom
  rtkprojectgeometricphantom.cxx
  ${rtkprojectgeometricphantom_GGO_C}
)
target_link_libraries(rtkprojectgeometricphantom RTK)

set_target_properties(
  rtkprojectgeometricphantom
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtkprojectgeometricphantom
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
