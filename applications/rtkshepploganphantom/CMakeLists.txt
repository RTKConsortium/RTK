WRAP_GGO(rtkshepploganphantom_GGO_C rtkshepploganphantom.ggo)
ADD_EXECUTABLE(rtkshepploganphantom rtkshepploganphantom.cxx ${rtkshepploganphantom_GGO_C})
TARGET_LINK_LIBRARIES(rtkshepploganphantom RTK)

# Installation code
IF(NOT RTK_INSTALL_NO_EXECUTABLES)
  FOREACH(EXE_NAME rtkshepploganphantom) 
    INSTALL(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  ENDFOREACH(EXE_NAME) 
ENDIF(NOT RTK_INSTALL_NO_EXECUTABLES)

