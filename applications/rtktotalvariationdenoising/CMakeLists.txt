WRAP_GGO(rtktotalvariationdenoising_GGO_C rtktotalvariationdenoising.ggo)
ADD_EXECUTABLE(rtktotalvariationdenoising rtktotalvariationdenoising.cxx ${rtktotalvariationdenoising_GGO_C})
TARGET_LINK_LIBRARIES(rtktotalvariationdenoising RTK)

IF(RTK_USE_CUDA)
  TARGET_LINK_LIBRARIES(rtktotalvariationdenoising rtkcuda)
ENDIF()

IF(RTK_USE_OPENCL)
  TARGET_LINK_LIBRARIES(rtktotalvariationdenoising rtkopencl)
ENDIF()

# Installation code
IF(NOT RTK_INSTALL_NO_EXECUTABLES)
  FOREACH(EXE_NAME rtktotalvariationdenoising) 
    INSTALL(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  ENDFOREACH(EXE_NAME) 
ENDIF(NOT RTK_INSTALL_NO_EXECUTABLES)

