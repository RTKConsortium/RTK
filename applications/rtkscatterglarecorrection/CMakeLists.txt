WRAP_GGO(rtkscatterglarecorrection_GGO_C rtkscatterglarecorrection.ggo ../rtkinputprojections_section.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
add_executable(rtkscatterglarecorrection rtkscatterglarecorrection.cxx ${rtkscatterglarecorrection_GGO_C})
target_link_libraries(rtkscatterglarecorrection RTK)

if (RTK_USE_CUDA)
  target_link_libraries(rtkscatterglarecorrection rtkcuda)
endif ()

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  foreach(EXE_NAME rtkscatterglarecorrection) 
    install(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  endforeach() 
endif()

