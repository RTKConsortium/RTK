WRAP_GGO(rtkdrawshepploganphantom_GGO_C rtkdrawshepploganphantom.ggo ../rtk3Doutputimage_section.ggo)
ADD_EXECUTABLE(rtkdrawshepploganphantom rtkdrawshepploganphantom.cxx ${rtkdrawshepploganphantom_GGO_C})
TARGET_LINK_LIBRARIES(rtkdrawshepploganphantom RTK)

# Installation code
IF(NOT RTK_INSTALL_NO_EXECUTABLES)
  FOREACH(EXE_NAME rtkdrawshepploganphantom) 
    INSTALL(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  ENDFOREACH(EXE_NAME) 
ENDIF(NOT RTK_INSTALL_NO_EXECUTABLES)

