wrap_ggo(rtkcheckimagequality_GGO_C rtkcheckimagequality.ggo)
add_executable(
  rtkcheckimagequality
  rtkcheckimagequality.cxx
  ${rtkcheckimagequality_GGO_C}
)
target_link_libraries(rtkcheckimagequality ${RTK_APPLICATION_TARGETS})

set_target_properties(
  rtkcheckimagequality
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtkcheckimagequality
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
