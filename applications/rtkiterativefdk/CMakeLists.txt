WRAP_GGO(rtkiterativefdk_GGO_C rtkiterativefdk.ggo ../rtkinputprojections_section.ggo ../rtk3Doutputimage_section.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
ADD_EXECUTABLE(rtkiterativefdk rtkiterativefdk.cxx ${rtkiterativefdk_GGO_C})
TARGET_LINK_LIBRARIES(rtkiterativefdk RTK)

IF(RTK_USE_CUDA)
  TARGET_LINK_LIBRARIES(rtkiterativefdk rtkcuda)
ENDIF()

IF(RTK_USE_OPENCL)
  TARGET_LINK_LIBRARIES(rtkiterativefdk rtkopencl)
ENDIF()

# Installation code
IF(NOT RTK_INSTALL_NO_EXECUTABLES)
  FOREACH(EXE_NAME rtkiterativefdk) 
    INSTALL(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  ENDFOREACH(EXE_NAME) 
ENDIF(NOT RTK_INSTALL_NO_EXECUTABLES)

