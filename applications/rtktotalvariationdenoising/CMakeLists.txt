WRAP_GGO(rtktotalvariationdenoising_GGO_C rtktotalvariationdenoising.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
add_executable(rtktotalvariationdenoising rtktotalvariationdenoising.cxx ${rtktotalvariationdenoising_GGO_C})
target_link_libraries(rtktotalvariationdenoising RTK)

if(RTK_USE_CUDA)
  target_link_libraries(rtktotalvariationdenoising rtkcuda)
endif()

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  foreach(EXE_NAME rtktotalvariationdenoising) 
    install(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  endforeach() 
endif()

