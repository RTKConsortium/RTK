WRAP_GGO(rtkphantomprojectionscreator_GGO_C rtkphantomprojectionscreator.ggo)
ADD_EXECUTABLE(rtkphantomprojectionscreator rtkphantomprojectionscreator.cxx ${rtkphantomprojectionscreator_GGO_C})
TARGET_LINK_LIBRARIES(rtkphantomprojectionscreator RTK)

# Installation code
IF(NOT RTK_INSTALL_NO_EXECUTABLES)
  FOREACH(EXE_NAME rtkphantomprojectionscreator) 
    INSTALL(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  ENDFOREACH(EXE_NAME) 
ENDIF(NOT RTK_INSTALL_NO_EXECUTABLES)

