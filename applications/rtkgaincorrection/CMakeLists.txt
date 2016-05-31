WRAP_GGO(rtkgaincorrection_GGO_C rtkgaincorrection.ggo ../rtkinputprojections_section.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
ADD_EXECUTABLE(rtkgaincorrection rtkgaincorrection.cxx ${rtkgaincorrection_GGO_C})
TARGET_LINK_LIBRARIES(rtkgaincorrection RTK)

IF(RTK_USE_CUDA)
  TARGET_LINK_LIBRARIES(rtkgaincorrection rtkcuda)
ENDIF()

# Installation code
IF(NOT RTK_INSTALL_NO_EXECUTABLES)
  FOREACH(EXE_NAME rtkgaincorrection) 
    INSTALL(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  ENDFOREACH(EXE_NAME) 
ENDIF(NOT RTK_INSTALL_NO_EXECUTABLES)
