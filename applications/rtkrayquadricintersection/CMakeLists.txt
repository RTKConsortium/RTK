WRAP_GGO(rtkrayquadricintersection_GGO_C rtkrayquadricintersection.ggo ../rtk3Doutputimage_section.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
ADD_EXECUTABLE(rtkrayquadricintersection rtkrayquadricintersection.cxx ${rtkrayquadricintersection_GGO_C})
TARGET_LINK_LIBRARIES(rtkrayquadricintersection RTK)

# Installation code
IF(NOT RTK_INSTALL_NO_EXECUTABLES)
  FOREACH(EXE_NAME rtkrayquadricintersection) 
    INSTALL(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  ENDFOREACH(EXE_NAME) 
ENDIF(NOT RTK_INSTALL_NO_EXECUTABLES)

