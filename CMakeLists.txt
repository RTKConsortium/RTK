#=========================================================
# ReconTK = Reconstruction toolkit
cmake_minimum_required(VERSION 2.4)
cmake_policy(VERSION 2.4)
if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)
PROJECT(ReconTK)
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

ADD_EXECUTABLE(synergylut synergylut.cxx itkHisImageIO.cxx itkHisImageIOFactory.cxx)
TARGET_LINK_LIBRARIES(synergylut ITKIO)

ADD_EXECUTABLE(synergyramp synergyramp.cxx itkHisImageIO.cxx itkHisImageIOFactory.cxx)
TARGET_LINK_LIBRARIES(synergyramp ITKIO ITKAlgorithms)
