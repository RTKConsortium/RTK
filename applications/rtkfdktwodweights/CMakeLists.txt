WRAP_GGO(rtkfdktwodweights_GGO_C rtkfdktwodweights.ggo ../rtkinputprojections_section.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
add_executable(rtkfdktwodweights rtkfdktwodweights.cxx ${rtkfdktwodweights_GGO_C})
target_link_libraries(rtkfdktwodweights RTK)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  foreach(EXE_NAME rtkfdktwodweights) 
    install(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  endforeach() 
endif()

