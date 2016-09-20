WRAP_GGO(rtkadmmwavelets_GGO_C rtkadmmwavelets.ggo ../rtkinputprojections_section.ggo ../rtk3Doutputimage_section.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
add_executable(rtkadmmwavelets rtkadmmwavelets.cxx ${rtkadmmwavelets_GGO_C})
target_link_libraries(rtkadmmwavelets RTK)

if (RTK_USE_CUDA)
  target_link_libraries(rtkadmmwavelets rtkcuda)
endif ()

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  foreach(EXE_NAME rtkadmmwavelets) 
    install(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  endforeach() 
endif()

