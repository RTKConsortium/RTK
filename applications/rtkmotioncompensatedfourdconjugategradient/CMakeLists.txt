wrap_ggo(rtkmotioncompensatedfourdconjugategradient_GGO_C rtkmotioncompensatedfourdconjugategradient.ggo ../rtkinputprojections_section.ggo ../rtk4Doutputimage_section.ggo ../rtkiterations_section.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
add_executable(
  rtkmotioncompensatedfourdconjugategradient
  rtkmotioncompensatedfourdconjugategradient.cxx
  ${rtkmotioncompensatedfourdconjugategradient_GGO_C}
)
target_link_libraries(rtkmotioncompensatedfourdconjugategradient RTK)

set_target_properties(
  rtkmotioncompensatedfourdconjugategradient
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
      "${RTK_BINARY_DIR}/bin"
)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  install(
    TARGETS
      rtkmotioncompensatedfourdconjugategradient
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
