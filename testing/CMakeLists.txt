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
   target_link_libraries(${testfullname} ${RTK_LIBRARIES})
   set_target_properties(${testfullname} PROPERTIES COMPILE_FLAGS -DUSE_CUDA)
   add_test(${testfullname} ${EXECUTABLE_OUTPUT_PATH}/${testfullname})
 endif()
endmacro()

# Executable to check the image quality
ADD_EXECUTABLE(rtkcheckimagequality rtkcheckimagequality.cxx)
TARGET_LINK_LIBRARIES(rtkcheckimagequality ${RTK_LIBRARIES})

ADD_EXECUTABLE(rtkfdktest rtkfdktest.cxx)
TARGET_LINK_LIBRARIES(rtkfdktest ${RTK_LIBRARIES})
ADD_TEST(rtkfdktest ${EXECUTABLE_OUTPUT_PATH}/rtkfdktest)
ADD_CUDA_TEST(rtkfdk rtkfdktest.cxx)
ADD_CUDA_TEST(rtkfdkprojweightcomp rtkfdkprojweightcompcudatest.cxx)

IF(OPENCL_FOUND AND OPENCL_HAVE_GPU)
  ADD_EXECUTABLE(rtkfdkopencltest rtkfdktest.cxx)
  TARGET_LINK_LIBRARIES(rtkfdkopencltest ${RTK_LIBRARIES})
  SET_TARGET_PROPERTIES(rtkfdkopencltest PROPERTIES COMPILE_FLAGS -DUSE_OPENCL)
  ADD_TEST(rtkfdkopencltest ${EXECUTABLE_OUTPUT_PATH}/rtkfdkopencltest)
ENDIF(OPENCL_FOUND AND OPENCL_HAVE_GPU)

ADD_CUDA_TEST(rtkcropfilter rtkcroptest.cxx)

ADD_EXECUTABLE(rtkmotioncompensatedfdktest rtkmotioncompensatedfdktest.cxx)
TARGET_LINK_LIBRARIES(rtkmotioncompensatedfdktest ${RTK_LIBRARIES})
ADD_TEST(rtkmotioncompensatedfdktest ${EXECUTABLE_OUTPUT_PATH}/rtkmotioncompensatedfdktest)

ADD_EXECUTABLE(rtkdisplaceddetectortest rtkdisplaceddetectortest.cxx)
TARGET_LINK_LIBRARIES(rtkdisplaceddetectortest ${RTK_LIBRARIES})
ADD_TEST(rtkdisplaceddetectortest ${EXECUTABLE_OUTPUT_PATH}/rtkdisplaceddetectortest)
ADD_CUDA_TEST(rtkdisplaceddetectortest rtkdisplaceddetectortest.cxx)
ADD_CUDA_TEST(rtkdisplaceddetectorcomp rtkdisplaceddetectorcompcudatest.cxx)

ADD_EXECUTABLE(rtkshortscantest rtkshortscantest.cxx)
TARGET_LINK_LIBRARIES(rtkshortscantest ${RTK_LIBRARIES})
ADD_TEST(rtkshortscantest ${EXECUTABLE_OUTPUT_PATH}/rtkshortscantest)
ADD_CUDA_TEST(rtkshortscantest rtkshortscantest.cxx)
ADD_CUDA_TEST(rtkshortscancomp rtkshortscancompcudatest.cxx)

ADD_EXECUTABLE(rtkrampfiltertest rtkrampfiltertest.cxx)
TARGET_LINK_LIBRARIES(rtkrampfiltertest ${RTK_LIBRARIES})
ADD_TEST(rtkrampfiltertest ${EXECUTABLE_OUTPUT_PATH}/rtkrampfiltertest)
ADD_CUDA_TEST(rtkrampfilter rtkrampfiltertest.cxx)

ADD_EXECUTABLE(rtkrampfiltertest2 rtkrampfiltertest2.cxx)
TARGET_LINK_LIBRARIES(rtkrampfiltertest2 ${RTK_LIBRARIES})
ADD_TEST(rtkrampfiltertest2 ${EXECUTABLE_OUTPUT_PATH}/rtkrampfiltertest2)
ADD_CUDA_TEST(rtkrampfilter2 rtkrampfiltertest2.cxx)

ADD_EXECUTABLE(rtkrampfilternofftwtest rtkrampfiltertest.cxx)
SET_TARGET_PROPERTIES(rtkrampfilternofftwtest PROPERTIES COMPILE_FLAGS -DRAMP_FILTER_TEST_WITHOUT_FFTW)
TARGET_LINK_LIBRARIES(rtkrampfilternofftwtest ${RTK_LIBRARIES})
ADD_TEST(rtkrampfilternofftwtest ${EXECUTABLE_OUTPUT_PATH}/rtkrampfilternofftwtest)
IF(RTK_USE_CUDA AND CUDA_HAVE_GPU)
  ADD_EXECUTABLE(rtkrampfilternofftwcudatest rtkrampfiltertest.cxx)
  TARGET_LINK_LIBRARIES(rtkrampfilternofftwcudatest ${RTK_LIBRARIES})
  SET_TARGET_PROPERTIES(rtkrampfilternofftwcudatest PROPERTIES COMPILE_FLAGS "-DUSE_CUDA -DRAMP_FILTER_TEST_WITHOUT_FFTW")
  ADD_TEST(rtkrampfilternofftwcudatest ${EXECUTABLE_OUTPUT_PATH}/rtkrampfilternofftwcudatest)
ENDIF(RTK_USE_CUDA AND CUDA_HAVE_GPU)

ADD_EXECUTABLE(rtkRaycastInterpolatorForwardProjectionTest rtkRaycastInterpolatorForwardProjectionTest.cxx)
TARGET_LINK_LIBRARIES(rtkRaycastInterpolatorForwardProjectionTest ${RTK_LIBRARIES})
ADD_TEST(rtkRaycastInterpolatorForwardProjectionTest ${EXECUTABLE_OUTPUT_PATH}/rtkRaycastInterpolatorForwardProjectionTest)

ADD_EXECUTABLE(rtkforwardprojectiontest rtkforwardprojectiontest.cxx)
TARGET_LINK_LIBRARIES(rtkforwardprojectiontest ${RTK_LIBRARIES})
ADD_TEST(rtkforwardprojectiontest ${EXECUTABLE_OUTPUT_PATH}/rtkforwardprojectiontest)
ADD_CUDA_TEST(rtkforwardprojection rtkforwardprojectiontest.cxx)

ADD_EXECUTABLE(rtkgeometryfiletest rtkgeometryfiletest.cxx)
TARGET_LINK_LIBRARIES(rtkgeometryfiletest ${RTK_LIBRARIES})
ADD_TEST(rtkgeometryfiletest ${EXECUTABLE_OUTPUT_PATH}/rtkgeometryfiletest)

ADD_EXECUTABLE(rtkTestReg23ProjectionGeometry rtkTestReg23ProjectionGeometry.cxx)
TARGET_LINK_LIBRARIES(rtkTestReg23ProjectionGeometry ${RTK_LIBRARIES})
ADD_TEST(rtkTestReg23ProjectionGeometry ${EXECUTABLE_OUTPUT_PATH}/rtkTestReg23ProjectionGeometry)

ADD_EXECUTABLE(rtkfovtest rtkfovtest.cxx)
TARGET_LINK_LIBRARIES(rtkfovtest ${RTK_LIBRARIES})
ADD_TEST(rtkfovtest ${EXECUTABLE_OUTPUT_PATH}/rtkfovtest)

ADD_EXECUTABLE(rtkbinningtest rtkbinningtest.cxx)
TARGET_LINK_LIBRARIES(rtkbinningtest ${RTK_LIBRARIES})
ADD_TEST(rtkbinningtest ${EXECUTABLE_OUTPUT_PATH}/rtkbinningtest)

ADD_EXECUTABLE(rtkmediantest rtkmediantest.cxx)
TARGET_LINK_LIBRARIES(rtkmediantest ${RTK_LIBRARIES})
ADD_TEST(rtkmediantest ${EXECUTABLE_OUTPUT_PATH}/rtkmediantest)

ADD_EXECUTABLE(rtkamsterdamshroudtest rtkamsterdamshroudtest.cxx)
TARGET_LINK_LIBRARIES(rtkamsterdamshroudtest ${RTK_LIBRARIES})
RTK_ADD_TEST(NAME rtkamsterdamshroudtest 
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkamsterdamshroudtest 
             DATA{Data/Baseline/AmsterdamShroud/Amsterdam.mha})

ADD_EXECUTABLE(rtkvariantest rtkvariantest.cxx)
TARGET_LINK_LIBRARIES(rtkvariantest ${RTK_LIBRARIES})
RTK_ADD_TEST(NAME rtkvariantest 
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkvariantest 
             DATA{Data/Input/Varian/raw.hnd,acqui.xml}
             DATA{Data/Baseline/Varian/attenuation.mha,geometry.xml})

ADD_EXECUTABLE(rtkelektatest rtkelektatest.cxx)
TARGET_LINK_LIBRARIES(rtkelektatest ${RTK_LIBRARIES})
RTK_ADD_TEST(NAME rtkelektatest
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkelektatest
             DATA{Data/Input/Elekta/FRAME.DBF,IMAGE.DBF,raw.his}
             DATA{Data/Baseline/Elekta/geometry.xml,attenuation.mha})

ADD_EXECUTABLE(rtkluttest rtkluttest.cxx)
TARGET_LINK_LIBRARIES(rtkluttest ${RTK_LIBRARIES})
RTK_ADD_TEST(NAME rtkluttest
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkluttest
             DATA{Data/Input/Elekta/raw.his}
             DATA{Data/Baseline/Elekta/attenuation.mha})

ADD_EXECUTABLE(rtkimagxtest rtkimagxtest.cxx)
TARGET_LINK_LIBRARIES(rtkimagxtest ${RTK_LIBRARIES})
RTK_ADD_TEST(NAME rtkimagxtest 
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkimagxtest 
             DATA{Data/Input/ImagX/raw.xml,raw.raw}
             DATA{Data/Baseline/ImagX/attenuation.mha})

ADD_EXECUTABLE(rtkedftest rtkedftest.cxx)
TARGET_LINK_LIBRARIES(rtkedftest ${RTK_LIBRARIES})
RTK_ADD_TEST(NAME rtkedftest
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkedftest
             DATA{Data/Input/ESRF/raw.edf,dark.edf,refHST0000.edf}
             DATA{Data/Baseline/ESRF/attenuation.mha})

ADD_EXECUTABLE(rtkdigisenstest rtkdigisenstest.cxx)
TARGET_LINK_LIBRARIES(rtkdigisenstest ${RTK_LIBRARIES})
RTK_ADD_TEST(NAME rtkdigisenstest
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkdigisenstest
             DATA{Data/Input/Digisens/ima0010.tif,calibration.cal},
             DATA{Data/Baseline/Digisens/attenuation.mha,geometry.xml})

ADD_EXECUTABLE(rtkxradtest rtkxradtest.cxx)
TARGET_LINK_LIBRARIES(rtkxradtest ${RTK_LIBRARIES})
RTK_ADD_TEST(NAME rtkxradtest 
             WORKING_DIRECTORY ${RTK_DATA_ROOT}/Input/XRad
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkxradtest
             DATA{Data/Input/XRad/flat.header,flat.img,dark.header,dark.img,SolidWater_HiGain1x1.header,SolidWater_HiGain1x1_firstProj.header,SolidWater_HiGain1x1_firstProj.img}
             DATA{Data/Baseline/XRad/geometry.xml,attenuation.mha})

ADD_EXECUTABLE(rtkprojectgeometricphantomtest rtkprojectgeometricphantomtest.cxx)
TARGET_LINK_LIBRARIES(rtkprojectgeometricphantomtest ${RTK_LIBRARIES})
RTK_ADD_TEST(NAME rtkprojectgeometricphantomtest 
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkprojectgeometricphantomtest 
             DATA{Data/Input/GeometricPhantom/SheppLogan.txt})

ADD_EXECUTABLE(rtkdrawgeometricphantomtest rtkdrawgeometricphantomtest.cxx)
TARGET_LINK_LIBRARIES(rtkdrawgeometricphantomtest ${RTK_LIBRARIES})
RTK_ADD_TEST(NAME rtkdrawgeometricphantomtest 
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkdrawgeometricphantomtest 
             DATA{Data/Input/GeometricPhantom/SheppLogan.txt}
             DATA{Data/Input/GeometricPhantom/Geometries.txt})

ADD_EXECUTABLE(rtksarttest rtksarttest.cxx)
TARGET_LINK_LIBRARIES(rtksarttest ${RTK_LIBRARIES})
ADD_TEST(rtksarttest ${EXECUTABLE_OUTPUT_PATH}/rtksarttest)
ADD_CUDA_TEST(rtksart rtksarttest.cxx)

ADD_EXECUTABLE(rtkfourdconjugategradienttest rtkfourdconjugategradienttest.cxx)
TARGET_LINK_LIBRARIES(rtkfourdconjugategradienttest ${RTK_LIBRARIES})
ADD_TEST(rtkfourdconjugategradienttest ${EXECUTABLE_OUTPUT_PATH}/rtkfourdconjugategradienttest)
ADD_CUDA_TEST(rtkfourdconjugategradient rtkfourdconjugategradienttest.cxx)

ADD_EXECUTABLE(rtkjosephadjointoperatorstest rtkjosephadjointoperatorstest.cxx)
TARGET_LINK_LIBRARIES(rtkjosephadjointoperatorstest ${RTK_LIBRARIES})
ADD_TEST(rtkjosephadjointoperatorstest ${EXECUTABLE_OUTPUT_PATH}/rtkjosephadjointoperatorstest)

ADD_EXECUTABLE(rtkfourdadjointoperatorstest rtkfourdadjointoperatorstest.cxx)
TARGET_LINK_LIBRARIES(rtkfourdadjointoperatorstest ${RTK_LIBRARIES})
RTK_ADD_TEST(NAME rtkfourdadjointoperatorstest
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkfourdadjointoperatorstest
             DATA{Data/Input/Phases/phases_slow.txt})

ADD_EXECUTABLE(rtkinterpolatesplatadjointtest rtkinterpolatesplatadjointtest.cxx)
TARGET_LINK_LIBRARIES(rtkinterpolatesplatadjointtest ${RTK_LIBRARIES})
RTK_ADD_TEST(NAME rtkinterpolatesplatadjointtest
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkinterpolatesplatadjointtest
             DATA{Data/Input/Phases/phases_slow.txt})

ADD_EXECUTABLE(rtktotalvariationtest rtktotalvariationtest.cxx)
TARGET_LINK_LIBRARIES(rtktotalvariationtest ${RTK_LIBRARIES})
ADD_TEST(rtktotalvariationtest ${EXECUTABLE_OUTPUT_PATH}/rtktotalvariationtest)

ADD_EXECUTABLE(rtkgradienttest rtkgradienttest.cxx)
TARGET_LINK_LIBRARIES(rtkgradienttest ${RTK_LIBRARIES})
ADD_TEST(rtkgradienttest ${EXECUTABLE_OUTPUT_PATH}/rtkgradienttest)

ADD_EXECUTABLE(rtkdivergencetest rtkdivergencetest.cxx)
TARGET_LINK_LIBRARIES(rtkdivergencetest ${RTK_LIBRARIES})
ADD_TEST(rtkdivergencetest ${EXECUTABLE_OUTPUT_PATH}/rtkdivergencetest)

ADD_EXECUTABLE(rtkconjugategradienttest rtkconjugategradienttest.cxx)
TARGET_LINK_LIBRARIES(rtkconjugategradienttest ${RTK_LIBRARIES})
ADD_TEST(rtkconjugategradienttest ${EXECUTABLE_OUTPUT_PATH}/rtkconjugategradienttest)

ADD_EXECUTABLE(rtkI0estimationtest rtkI0estimationtest.cxx)
TARGET_LINK_LIBRARIES(rtkI0estimationtest ${RTK_LIBRARIES})
ADD_TEST(rtkI0estimationtest ${EXECUTABLE_OUTPUT_PATH}/rtkI0estimationtest)

# We cannot compile this test using CPU if GPU is present
# This is because of rtkIterativeConeBeamReconstructionFilter
if(NOT RTK_USE_CUDA)
  ADD_EXECUTABLE(rtkconjugategradientreconstructiontest rtkconjugategradientreconstructiontest.cxx)
  TARGET_LINK_LIBRARIES(rtkconjugategradientreconstructiontest ${RTK_LIBRARIES})
  ADD_TEST(rtkconjugategradientreconstructiontest ${EXECUTABLE_OUTPUT_PATH}/rtkconjugategradientreconstructiontest)
endif()
ADD_CUDA_TEST(rtkconjugategradientreconstruction rtkconjugategradientreconstructiontest.cxx)

# We cannot compile this test using CPU if GPU is present
# This is because of rtkIterativeConeBeamReconstructionFilter
if(NOT RTK_USE_CUDA)
  ADD_EXECUTABLE(rtkadmmtotalvariationtest rtkadmmtotalvariationtest.cxx)
  TARGET_LINK_LIBRARIES(rtkadmmtotalvariationtest ${RTK_LIBRARIES})
  RTK_ADD_TEST(NAME rtkadmmtotalvariationtest
               COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkadmmtotalvariationtest
               DATA{Data/Input/Phases/phases.txt}
               DATA{Data/Input/Phases/phases_3projs.txt})
endif()
if(RTK_USE_CUDA AND CUDA_HAVE_GPU)
 SET(testfullname rtkadmmtotalvariationcudatest)
 ADD_EXECUTABLE(${testfullname} rtkadmmtotalvariationtest.cxx)
 TARGET_LINK_LIBRARIES(${testfullname} ${RTK_LIBRARIES})
 SET_TARGET_PROPERTIES(${testfullname} PROPERTIES COMPILE_FLAGS -DUSE_CUDA)
 RTK_ADD_TEST(NAME ${testfullname}
              COMMAND ${EXECUTABLE_OUTPUT_PATH}/${testfullname}
              DATA{Data/Input/Phases/phases.txt}
              DATA{Data/Input/Phases/phases_3projs.txt})
endif()

# We cannot compile this test using CPU if GPU is present
# This is because of rtkIterativeConeBeamReconstructionFilter
if(NOT RTK_USE_CUDA)
  ADD_EXECUTABLE(rtkfourdroostertest rtkfourdroostertest.cxx)
  TARGET_LINK_LIBRARIES(rtkfourdroostertest ${RTK_LIBRARIES})
  ADD_TEST(rtkfourdroostertest ${EXECUTABLE_OUTPUT_PATH}/rtkfourdroostertest)
endif()
ADD_CUDA_TEST(rtkfourdrooster rtkfourdroostertest.cxx)


# We cannot compile this test using CPU if GPU is present
# This is because of rtkIterativeConeBeamReconstructionFilter
if(NOT RTK_USE_CUDA)
  ADD_EXECUTABLE(rtkadmmwaveletstest rtkadmmwaveletstest.cxx)
  TARGET_LINK_LIBRARIES(rtkadmmwaveletstest RTK)
  ADD_TEST(rtkadmmwaveletstest ${EXECUTABLE_OUTPUT_PATH}/rtkadmmwaveletstest)
endif()
ADD_CUDA_TEST(rtkadmmwavelets rtkadmmwaveletstest.cxx)

ADD_EXECUTABLE(rtkwaveletstest rtkwaveletstest.cxx)
TARGET_LINK_LIBRARIES(rtkwaveletstest RTK)
ADD_TEST(rtkwaveletstest ${EXECUTABLE_OUTPUT_PATH}/rtkwaveletstest)

# Test headers
ADD_EXECUTABLE(rtkheadertest rtkheadertest.cxx)
IF(RTK_USE_CUDA)
  SET_TARGET_PROPERTIES(rtkheadertest PROPERTIES COMPILE_FLAGS -DUSE_CUDA)
ENDIF(RTK_USE_CUDA)
IF(OPENCL_FOUND)
  SET_TARGET_PROPERTIES(rtkheadertest PROPERTIES COMPILE_FLAGS -DUSE_OPENCL)
ENDIF(OPENCL_FOUND)
TARGET_LINK_LIBRARIES(rtkheadertest ${RTK_LIBRARIES})
ADD_TEST(rtkheadertest ${EXECUTABLE_OUTPUT_PATH}/rtkheadertest)

