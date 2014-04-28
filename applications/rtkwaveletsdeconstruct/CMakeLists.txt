WRAP_GGO(rtkwaveletsdeconstruct_GGO_C rtkwaveletsdeconstruct.ggo)
ADD_EXECUTABLE(rtkwaveletsdeconstruct rtkwaveletsdeconstruct.cxx ${rtkwaveletsdeconstruct_GGO_C})
TARGET_LINK_LIBRARIES(rtkwaveletsdeconstruct RTK)

IF (RTK_USE_CUDA)
  TARGET_LINK_LIBRARIES(rtkwaveletsdeconstruct rtkcuda)
ENDIF (RTK_USE_CUDA)

# Installation code
IF(NOT RTK_INSTALL_NO_EXECUTABLES)
  FOREACH(EXE_NAME rtkwaveletsdeconstruct) 
    INSTALL(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  ENDFOREACH(EXE_NAME) 
ENDIF(NOT RTK_INSTALL_NO_EXECUTABLES)

