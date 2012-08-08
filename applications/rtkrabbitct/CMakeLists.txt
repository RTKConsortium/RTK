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

