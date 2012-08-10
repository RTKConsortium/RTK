WRAP_GGO(rtkthreedphantomreference_GGO_C rtkthreedphantomreference.ggo)
ADD_EXECUTABLE(rtkthreedphantomreference rtkthreedphantomreference.cxx ${rtkthreedphantomreference_GGO_C})
TARGET_LINK_LIBRARIES(rtkthreedphantomreference RTK)

# Installation code
IF(NOT RTK_INSTALL_NO_EXECUTABLES)
  FOREACH(EXE_NAME rtkthreedphantomreference) 
    INSTALL(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  ENDFOREACH(EXE_NAME) 
ENDIF(NOT RTK_INSTALL_NO_EXECUTABLES)

