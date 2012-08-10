WRAP_GGO(rtkdigisensgeometry_GGO_C rtkdigisensgeometry.ggo)
ADD_EXECUTABLE(rtkdigisensgeometry rtkdigisensgeometry.cxx ${rtkdigisensgeometry_GGO_C} rtkDigisensGeometryXMLFileReader.cxx)
TARGET_LINK_LIBRARIES(rtkdigisensgeometry RTK)

# Installation code
IF(NOT RTK_INSTALL_NO_EXECUTABLES)
  FOREACH(EXE_NAME rtkdigisensgeometry) 
    INSTALL(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  ENDFOREACH(EXE_NAME) 
ENDIF(NOT RTK_INSTALL_NO_EXECUTABLES)

