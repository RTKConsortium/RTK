WRAP_GGO(rtkkatsevichforwardbinning_GGO_C rtkkatsevichforwardbinning.ggo ../rtkinputprojections_section.ggo  ${RTK_BINARY_DIR}/rtkVersion.ggo)
add_executable(rtkkatsevichforwardbinning rtkkatsevichforwardbinning.cxx ${rtkkatsevichforwardbinning_GGO_C})
target_link_libraries(rtkkatsevichforwardbinning RTK)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  foreach(EXE_NAME rtkkatsevichforwardbinning)
    install(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  endforeach()
endif()

