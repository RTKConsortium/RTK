WRAP_GGO(rtkaddgaussiannoise_GGO_C rtkaddgaussiannoise.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
add_executable(rtkaddgaussiannoise rtkaddgaussiannoise.cxx ${rtkaddgaussiannoise_GGO_C})
target_link_libraries(rtkaddgaussiannoise RTK)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  foreach(EXE_NAME rtkaddgaussiannoise)
    install(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  endforeach()
endif()
