wrap_ggo(rtkfourdconjugategradient_GGO_C rtkfourdconjugategradient.ggo ../rtkinputprojections_section.ggo ../rtk4Doutputimage_section.ggo ../rtkprojectors_section.ggo ../rtkiterations_section.ggo)
add_executable(
  rtkfourdconjugategradient
  rtkfourdconjugategradient.cxx
  ${rtkfourdconjugategradient_GGO_C}
)
target_link_libraries(rtkfourdconjugategradient ${RTK_APPLICATION_TARGETS})

set_target_properties(
  rtkfourdconjugategradient
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtkfourdconjugategradient
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
