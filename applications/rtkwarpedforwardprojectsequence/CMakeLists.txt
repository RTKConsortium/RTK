WRAP_GGO(rtkwarpedforwardprojectsequence_GGO_C rtkwarpedforwardprojectsequence.ggo ../rtk3Doutputimage_section.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
ADD_EXECUTABLE(rtkwarpedforwardprojectsequence rtkwarpedforwardprojectsequence.cxx ${rtkwarpedforwardprojectsequence_GGO_C})
TARGET_LINK_LIBRARIES(rtkwarpedforwardprojectsequence RTK)

IF (RTK_USE_CUDA)
  TARGET_LINK_LIBRARIES(rtkwarpedforwardprojectsequence rtkcuda)
ENDIF (RTK_USE_CUDA)

# Installation code
IF(NOT RTK_INSTALL_NO_EXECUTABLES)
  FOREACH(EXE_NAME rtkwarpedforwardprojectsequence) 
    INSTALL(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  ENDFOREACH(EXE_NAME) 
ENDIF(NOT RTK_INSTALL_NO_EXECUTABLES)

