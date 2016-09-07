#=========================================================
# RabbitCT
if(CUDA_FOUND)
  option(RTK_RABBITCT "Build library for RabbitCT: http://www5.informatik.uni-erlangen.de/research/projects/rabbitct/" OFF)
  if(RTK_RABBITCT)
    add_library(rtkrabbitct SHARED rtkrabbitct.cpp)
    target_link_libraries(rtkrabbitct ${CUDA_LIBRARIES} ITKCommon rtkcuda)
  endif()
endif()
#=========================================================

