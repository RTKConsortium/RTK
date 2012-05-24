FIND_PACKAGE(Gengetopt)

#=========================================================
# Executables
MACRO (ADD_RTK_EXECUTABLE EXE_NAME)
  WRAP_GGO(${EXE_NAME}_GGO_C ${EXE_NAME}.ggo)
  ADD_EXECUTABLE(${EXE_NAME} ${EXE_NAME}.cxx ${${EXE_NAME}_GGO_C} ${ARGN})
  TARGET_LINK_LIBRARIES(${EXE_NAME} rtkIO)
  SET(RTK_EXECUTABLE_TARGETS ${RTK_EXECUTABLE_TARGETS} ${EXE_NAME})
ENDMACRO (ADD_RTK_EXECUTABLE)

ADD_RTK_EXECUTABLE(rtkprojections)
ADD_RTK_EXECUTABLE(rtkparkershortscanweighting)
ADD_RTK_EXECUTABLE(rtkwangdisplaceddetectorweighting)
ADD_RTK_EXECUTABLE(rtkfdktwodweights)
ADD_RTK_EXECUTABLE(rtkramp)
ADD_RTK_EXECUTABLE(rtkfdk)
ADD_RTK_EXECUTABLE(rtkinlinefdk)
ADD_RTK_EXECUTABLE(rtkamsterdamshroud)
ADD_RTK_EXECUTABLE(rtkforwardprojections)
ADD_RTK_EXECUTABLE(rtkbackprojections)
ADD_RTK_EXECUTABLE(rtkrayboxintersection)
ADD_RTK_EXECUTABLE(rtksart)
ADD_RTK_EXECUTABLE(rtkrayquadricintersection)
ADD_RTK_EXECUTABLE(rtkshepploganphantom)
ADD_RTK_EXECUTABLE(rtkfovfilter)

IF (OPENCL_FOUND)
  TARGET_LINK_LIBRARIES(rtkfdk rtkopencl)
  TARGET_LINK_LIBRARIES(rtkinlinefdk rtkopencl)
ENDIF (OPENCL_FOUND)
IF (CUDA_FOUND)
  TARGET_LINK_LIBRARIES(rtkfdk rtkcuda)
  TARGET_LINK_LIBRARIES(rtkinlinefdk rtkcuda)
ENDIF (CUDA_FOUND)

#All the executables below are meant to create RTK ThreeDCircularProjectionGeometry files
ADD_RTK_EXECUTABLE(rtkvarianobigeometry rtkVarianObiXMLFileReader.cxx)
ADD_RTK_EXECUTABLE(rtksimulatedgeometry)
ADD_RTK_EXECUTABLE(rtkelektasynergygeometry rtkDbf.cxx)
ADD_RTK_EXECUTABLE(rtkdigisensgeometry rtkDigisensGeometryXMLFileReader.cxx)
#=========================================================

#=========================================================
# RabbitCT
IF(CUDA_FOUND)
  OPTION(RTK_RABBITCT "Build library for RabbitCT: http://www5.informatik.uni-erlangen.de/research/projects/rabbitct/" OFF)
  IF(RTK_RABBITCT)
    ADD_LIBRARY(rtkrabbitct SHARED rtkrabbitct.cpp)
    TARGET_LINK_LIBRARIES(rtkrabbitct ${CUDA_LIBRARIES} ITKCommon rtkcuda)
  ENDIF(RTK_RABBITCT)
ENDIF(CUDA_FOUND)
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

