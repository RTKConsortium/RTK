WRAP_GGO(rtktotalnuclearvariationdenoising_GGO_C rtktotalnuclearvariationdenoising.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
add_executable(rtktotalnuclearvariationdenoising rtktotalnuclearvariationdenoising.cxx ${rtktotalnuclearvariationdenoising_GGO_C})
target_link_libraries(rtktotalnuclearvariationdenoising RTK)

if(RTK_USE_CUDA)
  target_link_libraries(rtktotalnuclearvariationdenoising rtkcuda)
endif()

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  foreach(EXE_NAME rtktotalnuclearvariationdenoising) 
    install(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  endforeach() 
endif()

