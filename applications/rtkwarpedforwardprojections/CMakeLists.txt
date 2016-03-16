WRAP_GGO(rtkwarpedforwardprojections_GGO_C rtkwarpedforwardprojections.ggo ../rtk3Doutputimage_section.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
ADD_EXECUTABLE(rtkwarpedforwardprojections rtkwarpedforwardprojections.cxx ${rtkwarpedforwardprojections_GGO_C})
TARGET_LINK_LIBRARIES(rtkwarpedforwardprojections RTK)

IF (RTK_USE_CUDA)
  TARGET_LINK_LIBRARIES(rtkwarpedforwardprojections rtkcuda)
ENDIF (RTK_USE_CUDA)

# Installation code
IF(NOT RTK_INSTALL_NO_EXECUTABLES)
  FOREACH(EXE_NAME rtkwarpedforwardprojections) 
    INSTALL(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  ENDFOREACH(EXE_NAME) 
ENDIF(NOT RTK_INSTALL_NO_EXECUTABLES)

