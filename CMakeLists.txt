#=========================================================
# RTK = Reconstruction toolkit
PROJECT(RTK)
#=========================================================

#=========================================================
CMAKE_MINIMUM_REQUIRED (VERSION 2.6.0)

## Tell CMake to be quiet
CMAKE_POLICY (SET CMP0003 NEW)

## Default to release
IF (NOT CMAKE_BUILD_TYPE)
  SET (CMAKE_BUILD_TYPE Release)
ENDIF (NOT CMAKE_BUILD_TYPE)
#=========================================================

#=========================================================
# Find ITK (required)
FIND_PACKAGE(ITK)
IF(ITK_FOUND)
  INCLUDE(${ITK_USE_FILE})
ELSE(ITK_FOUND)
  MESSAGE(FATAL_ERROR "Cannot build without ITK. Set ITK_DIR.")
ENDIF(ITK_FOUND)
#=========================================================

#=========================================================
SET (RTK_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})
SET (RTK_LIBRARY_DIRS ${CMAKE_CURRENT_BINARY_DIR})
SET (RTK_LIBRARIES rtkIO)
CONFIGURE_FILE (${CMAKE_CURRENT_SOURCE_DIR}/RTKconfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/RTKconfig.cmake)
#=========================================================

#=========================================================
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
FIND_PACKAGE(Gengetopt)
#=========================================================

#=========================================================
ADD_LIBRARY(rtkIO STATIC
            itkHisImageIO.cxx
            itkHisImageIOFactory.cxx
            itkHndImageIO.cxx
            itkHndImageIOFactory.cxx)
TARGET_LINK_LIBRARIES(rtkIO ITKIO)
#=========================================================

#=========================================================
WRAP_GGO(rtkprojections_GGO_C rtkprojections.ggo)
ADD_EXECUTABLE(rtkprojections rtkprojections.cxx ${rtkprojections_GGO_C})
TARGET_LINK_LIBRARIES(rtkprojections rtkIO)
#=========================================================

#=========================================================
WRAP_GGO(rtkramp_GGO_C rtkramp.ggo)
ADD_EXECUTABLE(rtkramp rtkramp.cxx ${rtkramp_GGO_C})
TARGET_LINK_LIBRARIES(rtkramp rtkIO ITKAlgorithms)
#=========================================================


