#=========================================================
# RTK = Reconstruction toolkit
PROJECT(RTK)
#=========================================================

#=========================================================
#Support for the CTest dashboard testing system
INCLUDE(CTest)
#=========================================================

#=========================================================
CMAKE_MINIMUM_REQUIRED (VERSION 2.8)

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

# Flag required for shared libraries
IF ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
  SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
ENDIF ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
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
# Remove some Intel compiler warnings
IF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  STRING(REPLACE "-Wno-unused-parameter" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
  IF(WIN32)
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Qwd1268")
  ELSE(WIN32)
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -wd1268")
  ENDIF(WIN32)
ENDIF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
#=========================================================

#=========================================================
# Shared libraries option 
OPTION(BUILD_SHARED_LIBS "Build RTK with shared libraries." ${ITK_BUILD_SHARED})
#=========================================================

#=========================================================
SET(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake"  ${CMAKE_MODULE_PATH})
FIND_PACKAGE(Gengetopt)
FIND_PACKAGE(OpenCL)
FIND_PACKAGE(CUDA_wrap)
FIND_PACKAGE(OpenMP)
#=========================================================

#=========================================================
# Propagate cmake options in a header file
INCLUDE_DIRECTORIES(${PROJECT_BINARY_DIR})
CONFIGURE_FILE (${CMAKE_CURRENT_SOURCE_DIR}/rtkConfiguration.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/rtkConfiguration.h)
#=========================================================

#=========================================================
SET (RTK_INCLUDE_DIRS ${RTK_SOURCE_DIR})
SET (RTK_LIBRARY_DIRS ${RTK_BINARY_DIR})
SET (RTK_LIBRARIES rtkIO)

# Generate RTKConfig.cmake for the build tree.
SET (RTK_USE_FILE "${RTK_SOURCE_DIR}/cmake/UseRTK.cmake")
CONFIGURE_FILE (cmake/RTKConfig.cmake.in CMake/RTKConfig.cmake @ONLY)
#=========================================================

#=========================================================
ADD_LIBRARY(rtkIO
            itkHisImageIO.cxx
            itkHisImageIOFactory.cxx
            itkHndImageIO.cxx
            itkHndImageIOFactory.cxx
            rtkHomogeneousMatrix.cxx
            itkThreeDCircularProjectionGeometry.cxx
            itkThreeDCircularProjectionGeometryXMLFile.cxx
            itkFFTWCommon.cxx)
TARGET_LINK_LIBRARIES(rtkIO ${ITK_LIBRARIES})
#=========================================================

#=========================================================
# OpenCL library stuff
IF(OPENCL_LIBRARIES AND OPENCL_INCLUDE_DIRS)
  SET(OPENCL_FOUND TRUE)
ELSE(OPENCL_LIBRARIES AND OPENCL_INCLUDE_DIRS)
  SET(OPENCL_FOUND FALSE)
ENDIF(OPENCL_LIBRARIES AND OPENCL_INCLUDE_DIRS)

IF(OPENCL_FOUND)
  SET (RTK_LIBRARIES ${RTK_LIBRARIES} rtkopencl)
  INCLUDE_DIRECTORIES(${OPENCL_INCLUDE_DIRS})
  ADD_LIBRARY(rtkopencl
              itkOpenCLUtilities.cxx
              itkOpenCLFDKBackProjectionImageFilter.cxx)
  TARGET_LINK_LIBRARIES(rtkopencl ${OPENCL_LIBRARIES} ${ITK_LIBRARIES})

  FILE(GLOB openclSRCs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cl)
  ADD_CUSTOM_TARGET(openclSRCCopy ALL COMMENT "Copying OpenCL source files")
  FOREACH(openclSRC ${openclSRCs})
    ADD_CUSTOM_COMMAND(TARGET openclSRCCopy
                       COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${openclSRC}
                                                        ${RTK_BINARY_DIR}/${openclSRC})
  ENDFOREACH(openclSRC)
  ADD_DEPENDENCIES(rtkopencl openclSRCCopy)
ENDIF(OPENCL_FOUND)
#=========================================================

#=========================================================
# Cuda library stuff
IF (CUDA_FOUND)
  SET (RTK_LIBRARIES ${RTK_LIBRARIES} rtkcuda)
  INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
  CUDA_COMPILE (rtkcuda_CUDA_WRAPPERS
                itkCudaUtilities.cu
                itkCudaFDKBackProjectionImageFilter.cu
                itkCudaFFTRampImageFilter.cu )
  ADD_LIBRARY(rtkcuda
              itkCudaFDKBackProjectionImageFilter.cxx
              itkCudaFFTRampImageFilter.cxx
              ${rtkcuda_CUDA_WRAPPERS})
  SET_TARGET_PROPERTIES (rtkcuda PROPERTIES LINKER_LANGUAGE CXX)
  TARGET_LINK_LIBRARIES(rtkcuda ${CUDA_LIBRARIES} ${CUDA_cufft_LIBRARY} rtkIO)
ENDIF (CUDA_FOUND)
#=========================================================

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
IF (OPENCL_FOUND)
  TARGET_LINK_LIBRARIES(rtkfdk rtkopencl)
ENDIF (OPENCL_FOUND)
IF (CUDA_FOUND)
  TARGET_LINK_LIBRARIES(rtkfdk rtkcuda)
ENDIF (CUDA_FOUND)
ADD_RTK_EXECUTABLE(rtkamsterdamshroud)
ADD_RTK_EXECUTABLE(rtkforwardprojections)
ADD_RTK_EXECUTABLE(rtkbackprojections)
ADD_RTK_EXECUTABLE(rtkrayboxintersection)
ADD_RTK_EXECUTABLE(rtksart)

#All the executables below are meant to create RTK ThreeDCircularProjectionGeometry files
ADD_RTK_EXECUTABLE(rtkvarianobigeometry itkVarianObiXMLFileReader.cxx)
ADD_RTK_EXECUTABLE(rtksimulatedgeometry)
ADD_RTK_EXECUTABLE(rtkelektasynergygeometry rtkDbf.cxx)
ADD_RTK_EXECUTABLE(rtkdigisensgeometry itkDigisensGeometryXMLFileReader.cxx)
#=========================================================

#=========================================================
# If choose to build documentation, then search for Doxygen executables.
OPTION(BUILD_DOXYGEN "Build Doxygen Documentation" OFF)
IF(BUILD_DOXYGEN)
  FIND_PACKAGE(Doxygen)
  ADD_SUBDIRECTORY(Doxygen)
ENDIF(BUILD_DOXYGEN)
#=========================================================

#=========================================================
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
if(NOT RTK_INSTALL_RUNTIME_DIR)
  set(RTK_INSTALL_RUNTIME_DIR bin)
endif()
if(NOT RTK_INSTALL_LIB_DIR)
  set(RTK_INSTALL_LIB_DIR lib)
endif()
if(NOT RTK_INSTALL_ARCHIVE_DIR)
  set(RTK_INSTALL_ARCHIVE_DIR lib)
endif()
if(NOT RTK_INSTALL_INCLUDE_DIR)
  set(RTK_INSTALL_INCLUDE_DIR include/RTK-${RTK_VERSION_MAJOR}.${RTK_VERSION_MINOR})
endif()
if(NOT RTK_INSTALL_PACKAGE_DIR)
  set(RTK_INSTALL_PACKAGE_DIR "lib/cmake/RTK-${RTK_VERSION_MAJOR}.${RTK_VERSION_MINOR}")
endif()


IF(NOT RTK_INSTALL_NO_DEVELOPMENT)
  # Generate RTKConfig.cmake for the install tree.
  SET (RTK_USE_FILE "${CMAKE_INSTALL_PREFIX}/${RTK_INSTALL_PACKAGE_DIR}/UseRTK.cmake")
  SET (RTK_INCLUDE_DIRS "${CMAKE_INSTALL_PREFIX}/${RTK_INSTALL_INCLUDE_DIR}")
  SET (RTK_LIBRARY_DIRS "${CMAKE_INSTALL_PREFIX}/${RTK_INSTALL_LIB_DIR}")
  CONFIGURE_FILE (cmake/RTKConfig.cmake.in CMakeFiles/RTKConfig.cmake @ONLY)
  INSTALL(FILES ${RTK_BINARY_DIR}/CMakeFiles/RTKConfig.cmake
                cmake/UseRTK.cmake
          DESTINATION ${RTK_INSTALL_PACKAGE_DIR})

  # Include .h and .txx files
  FILE(GLOB __files1 "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
  FILE(GLOB __files2 "${CMAKE_CURRENT_SOURCE_DIR}/*.txx")
  INSTALL(FILES ${__files1} ${__files2} ${CMAKE_CURRENT_BINARY_DIR}/rtkConfiguration.h
    DESTINATION ${RTK_INSTALL_INCLUDE_DIR}
    COMPONENT Development)
ENDIF(NOT RTK_INSTALL_NO_DEVELOPMENT)

IF(NOT RTK_INSTALL_NO_LIBRARIES)
  INSTALL(TARGETS rtkIO
    RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
    LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
    ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)

  IF(CUDA_FOUND)
    INSTALL(TARGETS rtkcuda
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  ENDIF(CUDA_FOUND)
ENDIF(NOT RTK_INSTALL_NO_LIBRARIES)

IF(NOT RTK_INSTALL_NO_EXECUTABLES)
  FOREACH(EXE_NAME ${RTK_EXECUTABLE_TARGETS}) 
    INSTALL(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  ENDFOREACH(EXE_NAME) 
ENDIF(NOT RTK_INSTALL_NO_EXECUTABLES)
#=========================================================

