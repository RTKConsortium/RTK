WRAP_GGO(rtkthreedshepploganreference_GGO_C rtkthreedshepploganreference.ggo)
ADD_EXECUTABLE(rtkthreedshepploganreference rtkthreedshepploganreference.cxx ${rtkthreedshepploganreference_GGO_C})
TARGET_LINK_LIBRARIES(rtkthreedshepploganreference RTK)

# Installation code
IF(NOT RTK_INSTALL_NO_EXECUTABLES)
  FOREACH(EXE_NAME rtkthreedshepploganreference) 
    INSTALL(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  ENDFOREACH(EXE_NAME) 
ENDIF(NOT RTK_INSTALL_NO_EXECUTABLES)

