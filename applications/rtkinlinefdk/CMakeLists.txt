WRAP_GGO(rtkinlinefdk_GGO_C rtkinlinefdk.ggo)
ADD_EXECUTABLE(rtkinlinefdk rtkinlinefdk.cxx ${rtkinlinefdk_GGO_C})
TARGET_LINK_LIBRARIES(rtkinlinefdk RTK)
SET(RTK_EXECUTABLE_TARGETS ${RTK_EXECUTABLE_TARGETS} rtkinlinefdk)

IF (CUDA_FOUND)
  TARGET_LINK_LIBRARIES(rtkinlinefdk rtkcuda)
ENDIF (CUDA_FOUND)

IF (OPENCL_FOUND)
  TARGET_LINK_LIBRARIES(rtkinlinefdk rtkopencl)
ENDIF (OPENCL_FOUND)

