wrap_ggo(rtkmcfourdconjugategradient_GGO_C rtkmcfourdconjugategradient.ggo ../rtkinputprojections_section.ggo ../rtk4Doutputimage_section.ggo ../rtkiterations_section.ggo)
add_executable(
  rtkmcfourdconjugategradient
  rtkmcfourdconjugategradient.cxx
  ${rtkmcfourdconjugategradient_GGO_C}
)
target_link_libraries(rtkmcfourdconjugategradient RTK)

set_target_properties(
  rtkmcfourdconjugategradient
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtkmcfourdconjugategradient
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
