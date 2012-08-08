FIND_PACKAGE(Gengetopt)

#=========================================================
# Executables
ADD_SUBDIRECTORY(rtkprojections)
ADD_SUBDIRECTORY(rtkparkershortscanweighting)
ADD_SUBDIRECTORY(rtkwangdisplaceddetectorweighting)
ADD_SUBDIRECTORY(rtkfdktwodweights)
ADD_SUBDIRECTORY(rtkramp)
ADD_SUBDIRECTORY(rtkfdk)
ADD_SUBDIRECTORY(rtkinlinefdk)
ADD_SUBDIRECTORY(rtkamsterdamshroud)
ADD_SUBDIRECTORY(rtkforwardprojections)
ADD_SUBDIRECTORY(rtkbackprojections)
ADD_SUBDIRECTORY(rtkrayboxintersection)
ADD_SUBDIRECTORY(rtksart)
ADD_SUBDIRECTORY(rtkrayquadricintersection)
ADD_SUBDIRECTORY(rtkphantomprojectionscreator)
ADD_SUBDIRECTORY(rtkthreedphantomreference)
ADD_SUBDIRECTORY(rtkshepploganphantom)
ADD_SUBDIRECTORY(rtkfov)
ADD_SUBDIRECTORY(rtksubselect)

#All the executables below are meant to create RTK ThreeDCircularProjectionGeometry files
ADD_SUBDIRECTORY(rtkvarianobigeometry)
ADD_SUBDIRECTORY(rtksimulatedgeometry)
ADD_SUBDIRECTORY(rtkelektasynergygeometry)
ADD_SUBDIRECTORY(rtkdigisensgeometry)
#=========================================================

#=========================================================
# Installation code
IF(NOT RTK_INSTALL_NO_EXECUTABLES)
  FOREACH(EXE_NAME ${RTK_EXECUTABLE_TARGETS}) 
    INSTALL(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  ENDFOREACH(EXE_NAME) 
ENDIF(NOT RTK_INSTALL_NO_EXECUTABLES)
#=========================================================

