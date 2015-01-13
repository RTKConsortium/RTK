WRAP_GGO(rtkprojectshepploganphantom_GGO_C rtkprojectshepploganphantom.ggo ../rtk3Doutputimage_section.ggo)
ADD_EXECUTABLE(rtkprojectshepploganphantom rtkprojectshepploganphantom.cxx ${rtkprojectshepploganphantom_GGO_C})
TARGET_LINK_LIBRARIES(rtkprojectshepploganphantom RTK)

# Installation code
IF(NOT RTK_INSTALL_NO_EXECUTABLES)
  FOREACH(EXE_NAME rtkprojectshepploganphantom) 
    INSTALL(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  ENDFOREACH(EXE_NAME) 
ENDIF(NOT RTK_INSTALL_NO_EXECUTABLES)

