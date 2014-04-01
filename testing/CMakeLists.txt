# Set the data directory to the directory created by the ExternalData.cmake
# script

set(RTK_DATA_ROOT ${CMAKE_BINARY_DIR}/ExternalData/testing/Data CACHE PATH "Path of the data root")
MARK_AS_ADVANCED(RTK_DATA_ROOT)

CONFIGURE_FILE (${CMAKE_CURRENT_SOURCE_DIR}/rtkTestConfiguration.h.in
  ${CMAKE_BINARY_DIR}/rtkTestConfiguration.h)

# Macro to add CUDA test from regular test
macro(add_cuda_test testname testfile)
 if(RTK_USE_CUDA AND CUDA_HAVE_GPU)
   set(testfullname ${testname}cudatest)
   add_executable(${testfullname} ${testfile})
   target_link_libraries(${testfullname} RTK rtkcuda)
   set_target_properties(${testfullname} PROPERTIES COMPILE_FLAGS -DUSE_CUDA)
   add_test(${testfullname} ${EXECUTABLE_OUTPUT_PATH}/${testfullname})
 endif()
endmacro()

# Executable to check the image quality
ADD_EXECUTABLE(rtkcheckimagequality rtkcheckimagequality.cxx)
TARGET_LINK_LIBRARIES(rtkcheckimagequality RTK)

ADD_EXECUTABLE(rtkfdktest rtkfdktest.cxx)
TARGET_LINK_LIBRARIES(rtkfdktest RTK)
ADD_TEST(rtkfdktest ${EXECUTABLE_OUTPUT_PATH}/rtkfdktest)

IF(RTK_USE_CUDA AND CUDA_HAVE_GPU)
  ADD_EXECUTABLE(rtkfdkcudatest rtkfdktest.cxx)
  TARGET_LINK_LIBRARIES(rtkfdkcudatest RTK rtkcuda)
  SET_TARGET_PROPERTIES(rtkfdkcudatest PROPERTIES COMPILE_FLAGS -DUSE_CUDA)
  ADD_TEST(rtkfdkcudatest ${EXECUTABLE_OUTPUT_PATH}/rtkfdkcudatest)
ENDIF(RTK_USE_CUDA AND CUDA_HAVE_GPU)

IF(OPENCL_FOUND AND OPENCL_HAVE_GPU)
  ADD_EXECUTABLE(rtkfdkopencltest rtkfdktest.cxx)
  TARGET_LINK_LIBRARIES(rtkfdkopencltest RTK rtkopencl)
  SET_TARGET_PROPERTIES(rtkfdkopencltest PROPERTIES COMPILE_FLAGS -DUSE_OPENCL)
  ADD_TEST(rtkfdkopencltest ${EXECUTABLE_OUTPUT_PATH}/rtkfdkopencltest)
ENDIF(OPENCL_FOUND AND OPENCL_HAVE_GPU)

ADD_CUDA_TEST(rtkcropfilter rtkcroptest.cxx)

ADD_EXECUTABLE(rtkmotioncompensatedfdktest rtkmotioncompensatedfdktest.cxx)
TARGET_LINK_LIBRARIES(rtkmotioncompensatedfdktest RTK)
ADD_TEST(rtkmotioncompensatedfdktest ${EXECUTABLE_OUTPUT_PATH}/rtkmotioncompensatedfdktest)

ADD_EXECUTABLE(rtkdisplaceddetectortest rtkdisplaceddetectortest.cxx)
TARGET_LINK_LIBRARIES(rtkdisplaceddetectortest RTK)
ADD_TEST(rtkdisplaceddetectortest ${EXECUTABLE_OUTPUT_PATH}/rtkdisplaceddetectortest)

ADD_EXECUTABLE(rtkshortscantest rtkshortscantest.cxx)
TARGET_LINK_LIBRARIES(rtkshortscantest RTK)
ADD_TEST(rtkshortscantest ${EXECUTABLE_OUTPUT_PATH}/rtkshortscantest)

ADD_EXECUTABLE(rtkrampfiltertest rtkrampfiltertest.cxx)
TARGET_LINK_LIBRARIES(rtkrampfiltertest RTK)
ADD_TEST(rtkrampfiltertest ${EXECUTABLE_OUTPUT_PATH}/rtkrampfiltertest)
ADD_CUDA_TEST(rtkrampfilter rtkrampfiltertest.cxx)

ADD_EXECUTABLE(rtkrampfiltertest2 rtkrampfiltertest2.cxx)
TARGET_LINK_LIBRARIES(rtkrampfiltertest2 RTK)
ADD_TEST(rtkrampfiltertest2 ${EXECUTABLE_OUTPUT_PATH}/rtkrampfiltertest2)
ADD_CUDA_TEST(rtkrampfilter2 rtkrampfiltertest2.cxx)


ADD_EXECUTABLE(rtkrampfilternofftwtest rtkrampfiltertest.cxx)
SET_TARGET_PROPERTIES(rtkrampfilternofftwtest PROPERTIES COMPILE_FLAGS -DRAMP_FILTER_TEST_WITHOUT_FFTW)
TARGET_LINK_LIBRARIES(rtkrampfilternofftwtest RTK)
ADD_TEST(rtkrampfilternofftwtest ${EXECUTABLE_OUTPUT_PATH}/rtkrampfilternofftwtest)
IF(RTK_USE_CUDA AND CUDA_HAVE_GPU)
  ADD_EXECUTABLE(rtkrampfilternofftwcudatest rtkrampfiltertest.cxx)
  TARGET_LINK_LIBRARIES(rtkrampfilternofftwcudatest RTK rtkcuda)
  SET_TARGET_PROPERTIES(rtkrampfilternofftwcudatest PROPERTIES COMPILE_FLAGS "-DUSE_CUDA -DRAMP_FILTER_TEST_WITHOUT_FFTW")
  ADD_TEST(rtkrampfilternofftwcudatest ${EXECUTABLE_OUTPUT_PATH}/rtkrampfilternofftwcudatest)
ENDIF(RTK_USE_CUDA AND CUDA_HAVE_GPU)

ADD_EXECUTABLE(rtkRaycastInterpolatorForwardProjectionTest rtkRaycastInterpolatorForwardProjectionTest.cxx)
TARGET_LINK_LIBRARIES(rtkRaycastInterpolatorForwardProjectionTest RTK)
ADD_TEST(rtkRaycastInterpolatorForwardProjectionTest ${EXECUTABLE_OUTPUT_PATH}/rtkRaycastInterpolatorForwardProjectionTest)

ADD_EXECUTABLE(rtkforwardprojectiontest rtkforwardprojectiontest.cxx)
TARGET_LINK_LIBRARIES(rtkforwardprojectiontest RTK)
ADD_TEST(rtkforwardprojectiontest ${EXECUTABLE_OUTPUT_PATH}/rtkforwardprojectiontest)
ADD_CUDA_TEST(rtkforwardprojection rtkforwardprojectiontest.cxx)

ADD_EXECUTABLE(rtkgeometryfiletest rtkgeometryfiletest.cxx)
TARGET_LINK_LIBRARIES(rtkgeometryfiletest RTK)
ADD_TEST(rtkgeometryfiletest ${EXECUTABLE_OUTPUT_PATH}/rtkgeometryfiletest)

ADD_EXECUTABLE(rtkTestReg23ProjectionGeometry rtkTestReg23ProjectionGeometry.cxx)
TARGET_LINK_LIBRARIES(rtkTestReg23ProjectionGeometry RTK)
ADD_TEST(rtkTestReg23ProjectionGeometry ${EXECUTABLE_OUTPUT_PATH}/rtkTestReg23ProjectionGeometry)

ADD_EXECUTABLE(rtkfovtest rtkfovtest.cxx)
TARGET_LINK_LIBRARIES(rtkfovtest RTK)
ADD_TEST(rtkfovtest ${EXECUTABLE_OUTPUT_PATH}/rtkfovtest)

ADD_EXECUTABLE(rtkbinningtest rtkbinningtest.cxx)
TARGET_LINK_LIBRARIES(rtkbinningtest RTK)
ADD_TEST(rtkbinningtest ${EXECUTABLE_OUTPUT_PATH}/rtkbinningtest)

ADD_EXECUTABLE(rtkmediantest rtkmediantest.cxx)
TARGET_LINK_LIBRARIES(rtkmediantest RTK)
ADD_TEST(rtkmediantest ${EXECUTABLE_OUTPUT_PATH}/rtkmediantest)

ADD_EXECUTABLE(rtkamsterdamshroudtest rtkamsterdamshroudtest.cxx)
TARGET_LINK_LIBRARIES(rtkamsterdamshroudtest RTK)
RTK_ADD_TEST(NAME rtkamsterdamshroudtest 
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkamsterdamshroudtest 
             DATA{Data/Baseline/AmsterdamShroud/Amsterdam.mha})

ADD_EXECUTABLE(rtkvariantest rtkvariantest.cxx)
TARGET_LINK_LIBRARIES(rtkvariantest RTK)
RTK_ADD_TEST(NAME rtkvariantest 
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkvariantest 
             DATA{Data/Input/Varian/raw.hnd,acqui.xml}
             DATA{Data/Baseline/Varian/attenuation.mha,geometry.xml})

ADD_EXECUTABLE(rtkelektatest rtkelektatest.cxx)
TARGET_LINK_LIBRARIES(rtkelektatest RTK)
RTK_ADD_TEST(NAME rtkelektatest
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkelektatest
             DATA{Data/Input/Elekta/FRAME.DBF,IMAGE.DBF,raw.his}
             DATA{Data/Baseline/Elekta/geometry.xml,attenuation.mha})

ADD_EXECUTABLE(rtkluttest rtkluttest.cxx)
TARGET_LINK_LIBRARIES(rtkluttest RTK)
RTK_ADD_TEST(NAME rtkluttest
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkluttest
             DATA{Data/Input/Elekta/raw.his}
             DATA{Data/Baseline/Elekta/attenuation.mha})

ADD_EXECUTABLE(rtkimagxtest rtkimagxtest.cxx)
TARGET_LINK_LIBRARIES(rtkimagxtest RTK)
RTK_ADD_TEST(NAME rtkimagxtest 
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkimagxtest 
             DATA{Data/Input/ImagX/raw.xml,raw.raw}
             DATA{Data/Baseline/ImagX/attenuation.mha})

ADD_EXECUTABLE(rtkedftest rtkedftest.cxx)
TARGET_LINK_LIBRARIES(rtkedftest RTK)
RTK_ADD_TEST(NAME rtkedftest
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkedftest
             DATA{Data/Input/ESRF/raw.edf,dark.edf,refHST0000.edf}
             DATA{Data/Baseline/ESRF/attenuation.mha})

ADD_EXECUTABLE(rtkdigisenstest rtkdigisenstest.cxx)
TARGET_LINK_LIBRARIES(rtkdigisenstest RTK)
RTK_ADD_TEST(NAME rtkdigisenstest
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkdigisenstest
             DATA{Data/Input/Digisens/ima0010.tif,calibration.cal},
             DATA{Data/Baseline/Digisens/attenuation.mha,geometry.xml})

ADD_EXECUTABLE(rtkxradtest rtkxradtest.cxx)
TARGET_LINK_LIBRARIES(rtkxradtest RTK)
RTK_ADD_TEST(NAME rtkxradtest 
             WORKING_DIRECTORY ${RTK_DATA_ROOT}/Input/XRad
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkxradtest
             DATA{Data/Input/XRad/flat.header,flat.img,dark.header,dark.img,SolidWater_HiGain1x1.header,SolidWater_HiGain1x1_firstProj.header,SolidWater_HiGain1x1_firstProj.img}
             DATA{Data/Baseline/XRad/geometry.xml,attenuation.mha})

ADD_EXECUTABLE(rtkprojectgeometricphantomtest rtkprojectgeometricphantomtest.cxx)
TARGET_LINK_LIBRARIES(rtkprojectgeometricphantomtest RTK)
RTK_ADD_TEST(NAME rtkprojectgeometricphantomtest 
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkprojectgeometricphantomtest 
             DATA{Data/Input/GeometricPhantom/SheppLogan.txt})

ADD_EXECUTABLE(rtkdrawgeometricphantomtest rtkdrawgeometricphantomtest.cxx)
TARGET_LINK_LIBRARIES(rtkdrawgeometricphantomtest RTK)
RTK_ADD_TEST(NAME rtkdrawgeometricphantomtest 
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkdrawgeometricphantomtest 
             DATA{Data/Input/GeometricPhantom/SheppLogan.txt}
             DATA{Data/Input/GeometricPhantom/Geometries.txt})

ADD_EXECUTABLE(rtksarttest rtksarttest.cxx)
TARGET_LINK_LIBRARIES(rtksarttest RTK)
ADD_TEST(rtksarttest ${EXECUTABLE_OUTPUT_PATH}/rtksarttest)
ADD_CUDA_TEST(rtksart rtksarttest.cxx)

ADD_EXECUTABLE(rtkjosephadjointoperatorstest rtkjosephadjointoperatorstest.cxx)
TARGET_LINK_LIBRARIES(rtkjosephadjointoperatorstest RTK)
ADD_TEST(rtkjosephadjointoperatorstest ${EXECUTABLE_OUTPUT_PATH}/rtkjosephadjointoperatorstest)

ADD_EXECUTABLE(rtktotalvariationtest rtktotalvariationtest.cxx)
TARGET_LINK_LIBRARIES(rtktotalvariationtest RTK)
ADD_TEST(rtktotalvariationtest ${EXECUTABLE_OUTPUT_PATH}/rtktotalvariationtest)

ADD_EXECUTABLE(rtkgradienttest rtkgradienttest.cxx)
TARGET_LINK_LIBRARIES(rtkgradienttest RTK)
ADD_TEST(rtkgradienttest ${EXECUTABLE_OUTPUT_PATH}/rtkgradienttest)

ADD_EXECUTABLE(rtkdivergencetest rtkdivergencetest.cxx)
TARGET_LINK_LIBRARIES(rtkdivergencetest RTK)
ADD_TEST(rtkdivergencetest ${EXECUTABLE_OUTPUT_PATH}/rtkdivergencetest)

ADD_EXECUTABLE(rtkconjugategradienttest rtkconjugategradienttest.cxx)
TARGET_LINK_LIBRARIES(rtkconjugategradienttest RTK)
ADD_TEST(rtkconjugategradienttest ${EXECUTABLE_OUTPUT_PATH}/rtkconjugategradienttest)

ADD_EXECUTABLE(rtksirttest rtksirttest.cxx)
TARGET_LINK_LIBRARIES(rtksirttest RTK)
ADD_TEST(rtksirttest ${EXECUTABLE_OUTPUT_PATH}/rtksirttest)
ADD_CUDA_TEST(rtksirt rtksirttest.cxx)

ADD_EXECUTABLE(rtkadmmtotalvariationtest rtkadmmtotalvariationtest.cxx)
TARGET_LINK_LIBRARIES(rtkadmmtotalvariationtest RTK)
ADD_TEST(rtkadmmtotalvariationtest ${EXECUTABLE_OUTPUT_PATH}/rtkadmmtotalvariationtest)
ADD_CUDA_TEST(rtkadmmtotalvariation rtkadmmtotalvariationtest.cxx)

# Test headers
ADD_EXECUTABLE(rtkheadertest rtkheadertest.cxx)
IF(RTK_USE_CUDA)
  SET_TARGET_PROPERTIES(rtkheadertest PROPERTIES COMPILE_FLAGS -DUSE_CUDA)
ENDIF(RTK_USE_CUDA)
IF(OPENCL_FOUND)
  SET_TARGET_PROPERTIES(rtkheadertest PROPERTIES COMPILE_FLAGS -DUSE_OPENCL)
ENDIF(OPENCL_FOUND)
TARGET_LINK_LIBRARIES(rtkheadertest RTK)
ADD_TEST(rtkheadertest ${EXECUTABLE_OUTPUT_PATH}/rtkheadertest)

