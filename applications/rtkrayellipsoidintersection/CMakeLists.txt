WRAP_GGO(rtkrayellipsoidintersection_GGO_C rtkrayellipsoidintersection.ggo ../rtk3Doutputimage_section.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
ADD_EXECUTABLE(rtkrayellipsoidintersection rtkrayellipsoidintersection.cxx ${rtkrayellipsoidintersection_GGO_C})
TARGET_LINK_LIBRARIES(rtkrayellipsoidintersection RTK)

# Installation code
IF(NOT RTK_INSTALL_NO_EXECUTABLES)
  FOREACH(EXE_NAME rtkrayellipsoidintersection) 
    INSTALL(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  ENDFOREACH(EXE_NAME) 
ENDIF(NOT RTK_INSTALL_NO_EXECUTABLES)

