WRAP_GGO(rtkinlinefdk_GGO_C rtkinlinefdk.ggo ../rtkinputprojections_section.ggo ../rtk3Doutputimage_section.ggo)
ADD_EXECUTABLE(rtkinlinefdk rtkinlinefdk.cxx ${rtkinlinefdk_GGO_C})
TARGET_LINK_LIBRARIES(rtkinlinefdk RTK)

IF (RTK_USE_CUDA)
  TARGET_LINK_LIBRARIES(rtkinlinefdk rtkcuda)
ENDIF()

IF (RTK_USE_OPENCL)
  TARGET_LINK_LIBRARIES(rtkinlinefdk rtkopencl)
ENDIF()

# Installation code
IF(NOT RTK_INSTALL_NO_EXECUTABLES)
  FOREACH(EXE_NAME rtkinlinefdk) 
    INSTALL(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  ENDFOREACH(EXE_NAME) 
ENDIF(NOT RTK_INSTALL_NO_EXECUTABLES)

