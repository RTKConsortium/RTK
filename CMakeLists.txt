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
SET(RTK_VERSION_MAJOR "0")
SET(RTK_VERSION_MINOR "1")
SET(RTK_VERSION_PATCH "0")
#=========================================================

#=========================================================
# Help function to debug CMake
MACRO (DD in)
  MESSAGE(${in}=${${in}})
ENDMACRO(DD)
#=========================================================

#=========================================================
# Remove some MS Visual c++ flags
IF(MSVC)
  ADD_DEFINITIONS(-D_CRT_SECURE_NO_WARNINGS -D_CRT_NONSTDC_NO_DEPRECATE -D_SCL_SECURE_NO_WARNINGS)
ENDIF(MSVC)
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
# Propagate cmake options in a header file
INCLUDE_DIRECTORIES(${PROJECT_BINARY_DIR})
CONFIGURE_FILE (${CMAKE_CURRENT_SOURCE_DIR}/rtkConfiguration.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/rtkConfiguration.h)
#=========================================================

#=========================================================
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR} ${CMAKE_MODULE_PATH})
SET(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake"  ${CMAKE_MODULE_PATH})
FIND_PACKAGE(Gengetopt)
FIND_PACKAGE(OpenMP)
FIND_PACKAGE(CUDA_wrap)
FIND_PACKAGE(OpenCL)
#=========================================================

#=========================================================
ADD_LIBRARY(rtkIO STATIC
            itkHisImageIO.cxx
            itkHisImageIOFactory.cxx
            itkHndImageIO.cxx
            itkHndImageIOFactory.cxx
            rtkHomogeneousMatrix.cxx
            itkThreeDCircularProjectionGeometry.cxx
            itkThreeDCircularProjectionGeometryXMLFile.cxx)
TARGET_LINK_LIBRARIES(rtkIO ITKIO)
#=========================================================

#=========================================================
IF (CUDA_FOUND)
  INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
  CUDA_COMPILE (rtkcuda_CUDA_WRAPPERS
                cuda_mem.cu
                cuda_util.cu
                cuda_kernel_util.cu
                itkCudaFDKBackProjectionImageFilter.cu )
  ADD_LIBRARY(rtkcuda STATIC
              itkCudaFDKBackProjectionImageFilter.cxx
	            ${rtkcuda_CUDA_WRAPPERS})
  SET_TARGET_PROPERTIES (rtkcuda PROPERTIES LINKER_LANGUAGE CXX)
  TARGET_LINK_LIBRARIES(rtkcuda ${CUDA_LIBRARIES})
ENDIF (CUDA_FOUND)
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

#=========================================================
#All the executables below are meant to create RTK ThreeDCircularProjectionGeometry files

WRAP_GGO(rtkvarianobigeometry_GGO_C rtkvarianobigeometry.ggo)
ADD_EXECUTABLE(rtkvarianobigeometry rtkvarianobigeometry.cxx itkVarianObiXMLFileReader.cxx ${rtkvarianobigeometry_GGO_C})
TARGET_LINK_LIBRARIES(rtkvarianobigeometry rtkIO)

WRAP_GGO(rtksimulatedgeometry_GGO_C rtksimulatedgeometry.ggo)
ADD_EXECUTABLE(rtksimulatedgeometry rtksimulatedgeometry.cxx ${rtksimulatedgeometry_GGO_C})
TARGET_LINK_LIBRARIES(rtksimulatedgeometry rtkIO)

WRAP_GGO(rtkelektasynergygeometry_GGO_C rtkelektasynergygeometry.ggo)
ADD_EXECUTABLE(rtkelektasynergygeometry rtkelektasynergygeometry.cxx rtkDbf.cxx ${rtkelektasynergygeometry_GGO_C})
TARGET_LINK_LIBRARIES(rtkelektasynergygeometry rtkIO)
#=========================================================

#=========================================================
WRAP_GGO(rtkfdk_GGO_C rtkfdk.ggo)
ADD_EXECUTABLE(rtkfdk rtkfdk.cxx ${rtkfdk_GGO_C})
TARGET_LINK_LIBRARIES(rtkfdk rtkIO ITKAlgorithms)
IF (CUDA_FOUND)
  TARGET_LINK_LIBRARIES(rtkfdk rtkcuda)
ENDIF (CUDA_FOUND)
#=========================================================

#=========================================================
WRAP_GGO(rtkamsterdamshroud_GGO_C rtkamsterdamshroud.ggo)
ADD_EXECUTABLE(rtkamsterdamshroud rtkamsterdamshroud.cxx ${rtkamsterdamshroud_GGO_C})
TARGET_LINK_LIBRARIES(rtkamsterdamshroud rtkIO)
#=========================================================

#=========================================================
# If choose to build documentation, then search for Doxygen executables.
OPTION(BUILD_DOXYGEN "Build Doxygen Documentation" OFF)
IF(BUILD_DOXYGEN)
  FIND_PACKAGE(Doxygen)
  ADD_SUBDIRECTORY(Doxygen)
ENDIF(BUILD_DOXYGEN)
#=========================================================

