ADD_EXECUTABLE(rtkfdktest rtkfdktest.cxx)
TARGET_LINK_LIBRARIES(rtkfdktest RTK)
ADD_TEST(rtkfdktest ${EXECUTABLE_OUTPUT_PATH}/rtkfdktest)

ADD_EXECUTABLE(rtkdisplaceddetectortest rtkdisplaceddetectortest.cxx)
TARGET_LINK_LIBRARIES(rtkdisplaceddetectortest RTK)
ADD_TEST(rtkdisplaceddetectortest ${EXECUTABLE_OUTPUT_PATH}/rtkdisplaceddetectortest)

ADD_EXECUTABLE(rtkshortscantest rtkshortscantest.cxx)
TARGET_LINK_LIBRARIES(rtkshortscantest RTK)
ADD_TEST(rtkshortscantest ${EXECUTABLE_OUTPUT_PATH}/rtkshortscantest)

ADD_EXECUTABLE(rtkrampfiltertest rtkrampfiltertest.cxx)
TARGET_LINK_LIBRARIES(rtkrampfiltertest RTK)
ADD_TEST(rtkrampfiltertest ${EXECUTABLE_OUTPUT_PATH}/rtkrampfiltertest)

ADD_EXECUTABLE(rtkrampfilternofftwtest rtkrampfiltertest.cxx)
SET_TARGET_PROPERTIES(rtkrampfilternofftwtest PROPERTIES COMPILE_FLAGS -DRAMP_FILTER_TEST_WITHOUT_FFTW)
TARGET_LINK_LIBRARIES(rtkrampfilternofftwtest RTK)
ADD_TEST(rtkrampfiltertest ${EXECUTABLE_OUTPUT_PATH}/rtkrampfiltertest)
ADD_EXECUTABLE(rtkforwardprojectiontest rtkforwardprojectiontest.cxx)
TARGET_LINK_LIBRARIES(rtkforwardprojectiontest RTK)
ADD_TEST(rtkforwardprojectiontest ${EXECUTABLE_OUTPUT_PATH}/rtkforwardprojectiontest)

# Test headers
ADD_EXECUTABLE(rtkheadertest rtkheadertest.cxx)
IF(CUDA_FOUND)
  SET_TARGET_PROPERTIES(rtkheadertest PROPERTIES COMPILE_FLAGS -DUSE_CUDA)
ENDIF(CUDA_FOUND)
IF(OPENCL_FOUND)
  SET_TARGET_PROPERTIES(rtkheadertest PROPERTIES COMPILE_FLAGS -DUSE_OPENCL)
ENDIF(OPENCL_FOUND)
TARGET_LINK_LIBRARIES(rtkheadertest RTK)
ADD_TEST(rtkheadertest ${EXECUTABLE_OUTPUT_PATH}/rtkheadertest)