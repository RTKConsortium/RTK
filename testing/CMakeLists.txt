# Find the data directory.
FIND_PATH(RTK_DATA_ROOT RTKData.readme $ENV{RTK_DATA_ROOT})
MARK_AS_ADVANCED(RTK_DATA_ROOT)
CONFIGURE_FILE (${CMAKE_CURRENT_SOURCE_DIR}/rtkTestConfiguration.h.in
  ${CMAKE_BINARY_DIR}/rtkTestConfiguration.h)


ADD_EXECUTABLE(rtkfdktest rtkfdktest.cxx)
TARGET_LINK_LIBRARIES(rtkfdktest RTK)
ADD_TEST(rtkfdktest ${EXECUTABLE_OUTPUT_PATH}/rtkfdktest)
IF(CUDA_FOUND)
  ADD_EXECUTABLE(rtkfdkcudatest rtkfdktest.cxx)
  TARGET_LINK_LIBRARIES(rtkfdkcudatest RTK rtkcuda)
  SET_TARGET_PROPERTIES(rtkfdkcudatest PROPERTIES COMPILE_FLAGS -DUSE_CUDA)
  ADD_TEST(rtkfdkcudatest ${EXECUTABLE_OUTPUT_PATH}/rtkfdkcudatest)
ENDIF(CUDA_FOUND)
IF(OPENCL_FOUND)
  ADD_EXECUTABLE(rtkfdkopencltest rtkfdktest.cxx)
  TARGET_LINK_LIBRARIES(rtkfdkopencltest RTK rtkopencl)
  SET_TARGET_PROPERTIES(rtkfdkopencltest PROPERTIES COMPILE_FLAGS -DUSE_OPENCL)
  ADD_TEST(rtkfdkopencltest ${EXECUTABLE_OUTPUT_PATH}/rtkfdkopencltest)
ENDIF(OPENCL_FOUND)

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

ADD_EXECUTABLE(rtkRaycastInterpolatorForwardProjectionTest rtkRaycastInterpolatorForwardProjectionTest.cxx)
TARGET_LINK_LIBRARIES(rtkRaycastInterpolatorForwardProjectionTest RTK)
ADD_TEST(rtkRaycastInterpolatorForwardProjectionTest ${EXECUTABLE_OUTPUT_PATH}/rtkRaycastInterpolatorForwardProjectionTest)

ADD_EXECUTABLE(rtkforwardprojectiontest rtkforwardprojectiontest.cxx)
TARGET_LINK_LIBRARIES(rtkforwardprojectiontest RTK)
ADD_TEST(rtkforwardprojectiontest ${EXECUTABLE_OUTPUT_PATH}/rtkforwardprojectiontest)
IF(CUDA_FOUND)
  ADD_EXECUTABLE(rtkforwardprojectioncudatest rtkforwardprojectiontest.cxx)
  TARGET_LINK_LIBRARIES(rtkforwardprojectioncudatest RTK rtkcuda)
  SET_TARGET_PROPERTIES(rtkforwardprojectioncudatest PROPERTIES COMPILE_FLAGS -DUSE_CUDA)
  ADD_TEST(rtkforwardprojectioncudatest ${EXECUTABLE_OUTPUT_PATH}/rtkforwardprojectioncudatest)
ENDIF(CUDA_FOUND)

ADD_EXECUTABLE(rtkgeometryfiletest rtkgeometryfiletest.cxx)
TARGET_LINK_LIBRARIES(rtkgeometryfiletest RTK)
ADD_TEST(rtkgeometryfiletest ${EXECUTABLE_OUTPUT_PATH}/rtkgeometryfiletest)

ADD_EXECUTABLE(rtkfovtest rtkfovtest.cxx)
TARGET_LINK_LIBRARIES(rtkfovtest RTK)
ADD_TEST(rtkfovtest ${EXECUTABLE_OUTPUT_PATH}/rtkfovtest)

ADD_EXECUTABLE(rtkvariantest rtkvariantest.cxx)
TARGET_LINK_LIBRARIES(rtkvariantest RTK)
ADD_TEST(rtkvariantest ${EXECUTABLE_OUTPUT_PATH}/rtkvariantest)

ADD_EXECUTABLE(rtkelektatest rtkelektatest.cxx)
TARGET_LINK_LIBRARIES(rtkelektatest RTK)
ADD_TEST(rtkelektatest ${EXECUTABLE_OUTPUT_PATH}/rtkelektatest)

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
