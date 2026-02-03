wrap_ggo(rtkspectralforwardmodel_GGO_C rtkspectralforwardmodel.ggo)
add_executable(
  rtkspectralforwardmodel
  rtkspectralforwardmodel.cxx
  ${rtkspectralforwardmodel_GGO_C}
)
target_link_libraries(rtkspectralforwardmodel ${RTK_APPLICATION_TARGETS})

set_target_properties(
  rtkspectralforwardmodel
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtkspectralforwardmodel
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
