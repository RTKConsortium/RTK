WRAP_GGO(rtkwangdisplaceddetectorweighting_GGO_C rtkwangdisplaceddetectorweighting.ggo ../rtkinputprojections_section.ggo)
ADD_EXECUTABLE(rtkwangdisplaceddetectorweighting rtkwangdisplaceddetectorweighting.cxx ${rtkwangdisplaceddetectorweighting_GGO_C})
TARGET_LINK_LIBRARIES(rtkwangdisplaceddetectorweighting RTK)

IF(RTK_USE_CUDA)
  TARGET_LINK_LIBRARIES(rtkwangdisplaceddetectorweighting rtkcuda)
ENDIF()

# Installation code
IF(NOT RTK_INSTALL_NO_EXECUTABLES)
  FOREACH(EXE_NAME rtkwangdisplaceddetectorweighting) 
    INSTALL(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  ENDFOREACH(EXE_NAME) 
ENDIF(NOT RTK_INSTALL_NO_EXECUTABLES)

