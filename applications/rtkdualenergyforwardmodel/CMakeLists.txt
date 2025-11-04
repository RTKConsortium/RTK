wrap_ggo(rtkdualenergyforwardmodel_GGO_C rtkdualenergyforwardmodel.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
add_executable(
  rtkdualenergyforwardmodel
  rtkdualenergyforwardmodel.cxx
  ${rtkdualenergyforwardmodel_GGO_C}
)
target_link_libraries(rtkdualenergyforwardmodel RTK)

set_target_properties(
  rtkdualenergyforwardmodel
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${RTK_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtkdualenergyforwardmodel
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
