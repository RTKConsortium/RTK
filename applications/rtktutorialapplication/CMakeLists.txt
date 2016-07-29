WRAP_GGO(rtktutorialapplication_GGO_C rtktutorialapplication.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
add_executable(rtktutorialapplication rtktutorialapplication.cxx ${rtktutorialapplication_GGO_C})
target_link_libraries(rtktutorialapplication RTK)

if (RTK_USE_CUDA)
  target_link_libraries(rtktutorialapplication rtkcuda)
endif ()

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  foreach(EXE_NAME rtktutorialapplication) 
    install(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  endforeach() 
endif()

