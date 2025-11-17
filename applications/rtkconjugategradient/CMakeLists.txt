wrap_ggo(rtkconjugategradient_GGO_C rtkconjugategradient.ggo ../rtkinputprojections_section.ggo ../rtk3Doutputimage_section.ggo ../rtkprojectors_section.ggo ../rtkiterations_section.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
add_executable(
  rtkconjugategradient
  rtkconjugategradient.cxx
  ${rtkconjugategradient_GGO_C}
)
target_link_libraries(rtkconjugategradient RTK)

set_target_properties(
  rtkconjugategradient
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtkconjugategradient
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
