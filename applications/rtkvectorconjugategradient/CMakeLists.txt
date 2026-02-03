wrap_ggo(rtkvectorconjugategradient_GGO_C rtkvectorconjugategradient.ggo ../rtk3Doutputimage_section.ggo ../rtkprojectors_section.ggo)
add_executable(
  rtkvectorconjugategradient
  rtkvectorconjugategradient.cxx
  ${rtkvectorconjugategradient_GGO_C}
)
target_link_libraries(rtkvectorconjugategradient ${RTK_APPLICATION_TARGETS})

set_target_properties(
  rtkvectorconjugategradient
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtkvectorconjugategradient
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
