WRAP_GGO(rtklastdimensionl0gradientdenoising_GGO_C rtklastdimensionl0gradientdenoising.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
ADD_EXECUTABLE(rtklastdimensionl0gradientdenoising rtklastdimensionl0gradientdenoising.cxx ${rtklastdimensionl0gradientdenoising_GGO_C})
TARGET_LINK_LIBRARIES(rtklastdimensionl0gradientdenoising RTK)

IF(RTK_USE_CUDA)
  TARGET_LINK_LIBRARIES(rtklastdimensionl0gradientdenoising rtkcuda)
ENDIF()

IF(RTK_USE_OPENCL)
  TARGET_LINK_LIBRARIES(rtklastdimensionl0gradientdenoising rtkopencl)
ENDIF()

# Installation code
IF(NOT RTK_INSTALL_NO_EXECUTABLES)
  FOREACH(EXE_NAME rtklastdimensionl0gradientdenoising) 
    INSTALL(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  ENDFOREACH(EXE_NAME) 
ENDIF(NOT RTK_INSTALL_NO_EXECUTABLES)

