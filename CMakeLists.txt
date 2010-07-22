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
ADD_LIBRARY(rtkIO STATIC
            itkHisImageIO.cxx
            itkHisImageIOFactory.cxx
            itkHndImageIO.cxx
            itkHndImageIOFactory.cxx)
TARGET_LINK_LIBRARIES(rtkIO ITKIO)
#=========================================================

#=========================================================
ADD_EXECUTABLE(rtkprojections rtkprojections.cxx)
TARGET_LINK_LIBRARIES(rtkprojections rtkIO)
#=========================================================

#=========================================================
ADD_EXECUTABLE(rtkramp rtkramp.cxx)
TARGET_LINK_LIBRARIES(rtkramp rtkIO ITKAlgorithms)
#=========================================================


