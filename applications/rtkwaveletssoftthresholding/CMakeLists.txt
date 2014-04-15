WRAP_GGO(rtkwaveletssoftthresholding_GGO_C rtkwaveletssoftthresholding.ggo)
ADD_EXECUTABLE(rtkwaveletssoftthresholding rtkwaveletssoftthresholding.cxx ${rtkwaveletssoftthresholding_GGO_C})
TARGET_LINK_LIBRARIES(rtkwaveletssoftthresholding RTK)

IF (RTK_USE_CUDA)
  TARGET_LINK_LIBRARIES(rtkwaveletssoftthresholding rtkcuda)
ENDIF (RTK_USE_CUDA)

IF (OPENCL_FOUND)
  TARGET_LINK_LIBRARIES(rtkwaveletssoftthresholding rtkopencl)
ENDIF (OPENCL_FOUND)

# Installation code
IF(NOT RTK_INSTALL_NO_EXECUTABLES)
  FOREACH(EXE_NAME rtkwaveletssoftthresholding) 
    INSTALL(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  ENDFOREACH(EXE_NAME) 
ENDIF(NOT RTK_INSTALL_NO_EXECUTABLES)

