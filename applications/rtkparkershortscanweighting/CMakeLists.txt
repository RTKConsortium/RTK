WRAP_GGO(rtkparkershortscanweighting_GGO_C rtkparkershortscanweighting.ggo ../rtkinputprojections_section.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
ADD_EXECUTABLE(rtkparkershortscanweighting rtkparkershortscanweighting.cxx ${rtkparkershortscanweighting_GGO_C})
TARGET_LINK_LIBRARIES(rtkparkershortscanweighting RTK)

IF(RTK_USE_CUDA)
  TARGET_LINK_LIBRARIES(rtkparkershortscanweighting rtkcuda)
ENDIF()

# Installation code
IF(NOT RTK_INSTALL_NO_EXECUTABLES)
  FOREACH(EXE_NAME rtkparkershortscanweighting) 
    INSTALL(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  ENDFOREACH(EXE_NAME) 
ENDIF(NOT RTK_INSTALL_NO_EXECUTABLES)

