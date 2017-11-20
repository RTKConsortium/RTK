# Set the data directory to the directory created by the ExternalData.cmake
# script

set(RTK_DATA_ROOT ${CMAKE_BINARY_DIR}/ExternalData/testing/Data CACHE PATH "Path of the data root")
mark_as_advanced(RTK_DATA_ROOT)

configure_file (${CMAKE_CURRENT_SOURCE_DIR}/rtkTestConfiguration.h.in
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

# Macro to add CUDA test when external data is required
macro(rtk_add_cuda_test testname testfile)
 if(RTK_USE_CUDA AND CUDA_HAVE_GPU)
   set(testfullname ${testname}cudatest)
   add_executable(${testfullname} ${testfile})
   target_link_libraries(${testfullname} ${RTK_LIBRARIES})
   set_target_properties(${testfullname} PROPERTIES COMPILE_FLAGS -DUSE_CUDA)
   set(data_list "")
   foreach(loop_data ${ARGN})
      set(data_list "${data_list} DATA{${loop_data}}")
   endforeach()
   RTK_ADD_TEST(NAME ${testfullname}
		COMMAND ${EXECUTABLE_OUTPUT_PATH}/${testfullname}
		${data_list}
		)
 endif()
endmacro()

# Executable to check the image quality
add_executable(rtkcheckimagequality rtkcheckimagequality.cxx)
target_link_libraries(rtkcheckimagequality ${RTK_LIBRARIES})

add_executable(rtkfdktest rtkfdktest.cxx)
target_link_libraries(rtkfdktest ${RTK_LIBRARIES})
add_test(rtkfdktest ${EXECUTABLE_OUTPUT_PATH}/rtkfdktest)
ADD_CUDA_TEST(rtkfdk rtkfdktest.cxx)
ADD_CUDA_TEST(rtkfdkprojweightcomp rtkfdkprojweightcompcudatest.cxx)

add_executable(rtkimporttest rtkimporttest.cxx)
target_link_libraries(rtkimporttest ${RTK_LIBRARIES})
add_test(rtkimporttest ${EXECUTABLE_OUTPUT_PATH}/rtkimporttest)
ADD_CUDA_TEST(rtkimport rtkimporttest.cxx)

ADD_CUDA_TEST(rtkcropfilter rtkcroptest.cxx)

add_executable(rtkmotioncompensatedfdktest rtkmotioncompensatedfdktest.cxx)
target_link_libraries(rtkmotioncompensatedfdktest ${RTK_LIBRARIES})
add_test(rtkmotioncompensatedfdktest ${EXECUTABLE_OUTPUT_PATH}/rtkmotioncompensatedfdktest)

add_executable(rtkdisplaceddetectortest rtkdisplaceddetectortest.cxx)
target_link_libraries(rtkdisplaceddetectortest ${RTK_LIBRARIES})
add_test(rtkdisplaceddetectortest ${EXECUTABLE_OUTPUT_PATH}/rtkdisplaceddetectortest)
ADD_CUDA_TEST(rtkdisplaceddetectortest rtkdisplaceddetectortest.cxx)
ADD_CUDA_TEST(rtkdisplaceddetectorcomp rtkdisplaceddetectorcompcudatest.cxx)

add_executable(rtkdisplaceddetectorcompoffsettest rtkdisplaceddetectorcompoffsettest.cxx)
target_link_libraries(rtkdisplaceddetectorcompoffsettest ${RTK_LIBRARIES})
add_test(rtkdisplaceddetectorcompoffsettest ${EXECUTABLE_OUTPUT_PATH}/rtkdisplaceddetectorcompoffsettest)

add_executable(rtkshortscantest rtkshortscantest.cxx)
target_link_libraries(rtkshortscantest ${RTK_LIBRARIES})
add_test(rtkshortscantest ${EXECUTABLE_OUTPUT_PATH}/rtkshortscantest)
ADD_CUDA_TEST(rtkshortscantest rtkshortscantest.cxx)
ADD_CUDA_TEST(rtkshortscancomp rtkshortscancompcudatest.cxx)

add_executable(rtkrampfiltertest rtkrampfiltertest.cxx)
target_link_libraries(rtkrampfiltertest ${RTK_LIBRARIES})
add_test(rtkrampfiltertest ${EXECUTABLE_OUTPUT_PATH}/rtkrampfiltertest)
ADD_CUDA_TEST(rtkrampfilter rtkrampfiltertest.cxx)

add_executable(rtkrampfiltertest2 rtkrampfiltertest2.cxx)
target_link_libraries(rtkrampfiltertest2 ${RTK_LIBRARIES})
add_test(rtkrampfiltertest2 ${EXECUTABLE_OUTPUT_PATH}/rtkrampfiltertest2)
ADD_CUDA_TEST(rtkrampfilter2 rtkrampfiltertest2.cxx)

add_executable(rtkrampfilternofftwtest rtkrampfiltertest.cxx)
set_target_properties(rtkrampfilternofftwtest PROPERTIES COMPILE_FLAGS -DRAMP_FILTER_TEST_WITHOUT_FFTW)
target_link_libraries(rtkrampfilternofftwtest ${RTK_LIBRARIES})
add_test(rtkrampfilternofftwtest ${EXECUTABLE_OUTPUT_PATH}/rtkrampfilternofftwtest)
if(RTK_USE_CUDA AND CUDA_HAVE_GPU)
  add_executable(rtkrampfilternofftwcudatest rtkrampfiltertest.cxx)
  target_link_libraries(rtkrampfilternofftwcudatest ${RTK_LIBRARIES})
  set_target_properties(rtkrampfilternofftwcudatest PROPERTIES COMPILE_FLAGS "-DUSE_CUDA -DRAMP_FILTER_TEST_WITHOUT_FFTW")
  add_test(rtkrampfilternofftwcudatest ${EXECUTABLE_OUTPUT_PATH}/rtkrampfilternofftwcudatest)
endif()

add_executable(rtkscatterglarefiltertest rtkscatterglarefiltertest.cxx)
target_link_libraries(rtkscatterglarefiltertest ${RTK_LIBRARIES})
add_test(rtkscatterglarefiltertest ${EXECUTABLE_OUTPUT_PATH}/rtkscatterglarefiltertest)
ADD_CUDA_TEST(rtkscatterglarefilter rtkscatterglarefiltertest.cxx)

add_executable(rtkgaincorrectiontest rtkgaincorrectiontest.cxx)
target_link_libraries(rtkgaincorrectiontest ${RTK_LIBRARIES})
add_test(rtkgaincorrectiontest ${EXECUTABLE_OUTPUT_PATH}/rtkgaincorrectiontest)
ADD_CUDA_TEST(rtkgaincorrectiontest rtkgaincorrectiontest.cxx)

add_executable(rtkscatterglarefilternofftwtest rtkscatterglarefiltertest.cxx)
set_target_properties(rtkscatterglarefilternofftwtest PROPERTIES COMPILE_FLAGS -DRAMP_FILTER_TEST_WITHOUT_FFTW)
target_link_libraries(rtkscatterglarefilternofftwtest ${RTK_LIBRARIES})

add_executable(rtkRaycastInterpolatorForwardProjectionTest rtkRaycastInterpolatorForwardProjectionTest.cxx)
target_link_libraries(rtkRaycastInterpolatorForwardProjectionTest ${RTK_LIBRARIES})
add_test(rtkRaycastInterpolatorForwardProjectionTest ${EXECUTABLE_OUTPUT_PATH}/rtkRaycastInterpolatorForwardProjectionTest)

add_executable(rtkforwardprojectiontest rtkforwardprojectiontest.cxx)
target_link_libraries(rtkforwardprojectiontest ${RTK_LIBRARIES})
add_test(rtkforwardprojectiontest ${EXECUTABLE_OUTPUT_PATH}/rtkforwardprojectiontest)
ADD_CUDA_TEST(rtkforwardprojection rtkforwardprojectiontest.cxx)

add_executable(rtkgeometryfiletest rtkgeometryfiletest.cxx)
target_link_libraries(rtkgeometryfiletest ${RTK_LIBRARIES})
add_test(rtkgeometryfiletest ${EXECUTABLE_OUTPUT_PATH}/rtkgeometryfiletest)

add_executable(rtkTestReg23ProjectionGeometry rtkTestReg23ProjectionGeometry.cxx)
target_link_libraries(rtkTestReg23ProjectionGeometry ${RTK_LIBRARIES})
add_test(rtkTestReg23ProjectionGeometry ${EXECUTABLE_OUTPUT_PATH}/rtkTestReg23ProjectionGeometry)

add_executable(rtkfovtest rtkfovtest.cxx)
target_link_libraries(rtkfovtest ${RTK_LIBRARIES})
add_test(rtkfovtest ${EXECUTABLE_OUTPUT_PATH}/rtkfovtest)

add_executable(rtkbinningtest rtkbinningtest.cxx)
target_link_libraries(rtkbinningtest ${RTK_LIBRARIES})
add_test(rtkbinningtest ${EXECUTABLE_OUTPUT_PATH}/rtkbinningtest)

add_executable(rtkl0gradientnormtest rtkl0gradientnormtest.cxx)
target_link_libraries(rtkl0gradientnormtest ${RTK_LIBRARIES})
add_test(rtkl0gradientnormtest ${EXECUTABLE_OUTPUT_PATH}/rtkl0gradientnormtest)

add_executable(rtkmediantest rtkmediantest.cxx)
target_link_libraries(rtkmediantest ${RTK_LIBRARIES})
add_test(rtkmediantest ${EXECUTABLE_OUTPUT_PATH}/rtkmediantest)

add_executable(rtkwaterprecorrectiontest rtkwaterprecorrectiontest.cxx)
target_link_libraries(rtkwaterprecorrectiontest ${RTK_LIBRARIES})
add_test(rtkwaterprecorrectiontest ${EXECUTABLE_OUTPUT_PATH}/rtkwaterprecorrectiontest)

add_executable(rtklutbasedvarI0rawtoatttest rtklutbasedvarI0rawtoatttest.cxx)
target_link_libraries(rtklutbasedvarI0rawtoatttest ${RTK_LIBRARIES})
add_test(rtklutbasedvarI0rawtoatttest ${EXECUTABLE_OUTPUT_PATH}/rtklutbasedvarI0rawtoatttest)

add_executable(rtkdecomposespectralprojectionstest rtkdecomposespectralprojectionstest.cxx)
target_link_libraries(rtkdecomposespectralprojectionstest ${RTK_LIBRARIES})
RTK_ADD_TEST(NAME rtkdecomposespectralprojectionstest
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkdecomposespectralprojectionstest
             DATA{Data/Input/Spectral/incident_spectrum.mha}
             DATA{Data/Input/Spectral/material_attenuations.mha}
             DATA{Data/Input/Spectral/detector_response.mha})
             
add_executable(rtkvectorimageconverterstest rtkvectorimageconverterstest.cxx)
target_link_libraries(rtkvectorimageconverterstest ${RTK_LIBRARIES})
add_test(rtkvectorimageconverterstest ${EXECUTABLE_OUTPUT_PATH}/rtkvectorimageconverterstest)

add_executable(rtkamsterdamshroudtest rtkamsterdamshroudtest.cxx)
target_link_libraries(rtkamsterdamshroudtest ${RTK_LIBRARIES})
RTK_ADD_TEST(NAME rtkamsterdamshroudtest
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkamsterdamshroudtest
             DATA{Data/Baseline/AmsterdamShroud/Amsterdam_crop.mha}
             DATA{Data/Baseline/AmsterdamShroud/Amsterdam.mha})

add_executable(rtkvariantest rtkvariantest.cxx)
target_link_libraries(rtkvariantest ${RTK_LIBRARIES})
RTK_ADD_TEST(NAME rtkvariantest
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkvariantest
             DATA{Data/Input/Varian/raw.hnd,acqui.xml,Proj_00000.xim,acqui_probeam.xml}
             DATA{Data/Baseline/Varian/attenuation.mha,geometry.xml,attenuationProBeam.mha,geometryProBeam.xml})

add_executable(rtkelektatest rtkelektatest.cxx)
target_link_libraries(rtkelektatest ${RTK_LIBRARIES})
RTK_ADD_TEST(NAME rtkelektatest
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkelektatest
             DATA{Data/Input/Elekta/FRAME.DBF,IMAGE.DBF,raw.his,_Frames.xml}
             DATA{Data/Baseline/Elekta/geometry.xml,attenuation.mha,geometry5.xml})

add_executable(rtkluttest rtkluttest.cxx)
target_link_libraries(rtkluttest ${RTK_LIBRARIES})
RTK_ADD_TEST(NAME rtkluttest
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkluttest
             DATA{Data/Input/Elekta/raw.his}
             DATA{Data/Baseline/Elekta/attenuation.mha})

add_executable(rtkimagxtest rtkimagxtest.cxx)
target_link_libraries(rtkimagxtest ${RTK_LIBRARIES})
RTK_ADD_TEST(NAME rtkimagxtest
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkimagxtest
             DATA{Data/Input/ImagX/calibration.xml,room.xml,raw.xml,raw.raw,1.dcm}
             DATA{Data/Baseline/ImagX/geo.xml,attenuation.mha,attenuationDCM.mha})

add_executable(rtkedftest rtkedftest.cxx)
target_link_libraries(rtkedftest ${RTK_LIBRARIES})
RTK_ADD_TEST(NAME rtkedftest
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkedftest
             DATA{Data/Input/ESRF/raw.edf,dark.edf,refHST0000.edf}
             DATA{Data/Baseline/ESRF/attenuation.mha})

add_executable(rtkdigisenstest rtkdigisenstest.cxx)
target_link_libraries(rtkdigisenstest ${RTK_LIBRARIES})
RTK_ADD_TEST(NAME rtkdigisenstest
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkdigisenstest
             DATA{Data/Input/Digisens/ima0010.tif,calibration.cal},
             DATA{Data/Baseline/Digisens/attenuation.mha,geometry.xml})

add_executable(rtkxradtest rtkxradtest.cxx)
target_link_libraries(rtkxradtest ${RTK_LIBRARIES})
RTK_ADD_TEST(NAME rtkxradtest
             WORKING_DIRECTORY ${RTK_DATA_ROOT}/Input/XRad
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkxradtest
             DATA{Data/Input/XRad/flat.header,flat.img,dark.header,dark.img,SolidWater_HiGain1x1.header,SolidWater_HiGain1x1_firstProj.header,SolidWater_HiGain1x1_firstProj.img}
             DATA{Data/Baseline/XRad/geometry.xml,attenuation.mha})

add_executable(rtkprojectgeometricphantomtest rtkprojectgeometricphantomtest.cxx)
target_link_libraries(rtkprojectgeometricphantomtest ${RTK_LIBRARIES})
RTK_ADD_TEST(NAME rtkprojectgeometricphantomtest
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkprojectgeometricphantomtest
             DATA{Data/Input/GeometricPhantom/SheppLogan.txt})

add_executable(rtkdrawgeometricphantomtest rtkdrawgeometricphantomtest.cxx)
target_link_libraries(rtkdrawgeometricphantomtest ${RTK_LIBRARIES})
RTK_ADD_TEST(NAME rtkdrawgeometricphantomtest
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkdrawgeometricphantomtest
             DATA{Data/Input/GeometricPhantom/SheppLogan.txt}
             DATA{Data/Input/GeometricPhantom/Geometries.txt})

add_executable(rtksarttest rtksarttest.cxx)
target_link_libraries(rtksarttest ${RTK_LIBRARIES})
add_test(rtksarttest ${EXECUTABLE_OUTPUT_PATH}/rtksarttest)
ADD_CUDA_TEST(rtksart rtksarttest.cxx)

add_executable(rtkfourdsarttest rtkfourdsarttest.cxx)
target_link_libraries(rtkfourdsarttest ${RTK_LIBRARIES})
add_test(rtkfourdsarttest ${EXECUTABLE_OUTPUT_PATH}/rtkfourdsarttest)
ADD_CUDA_TEST(rtkfourdsart rtkfourdsarttest.cxx)

add_executable(rtkfourdconjugategradienttest rtkfourdconjugategradienttest.cxx)
target_link_libraries(rtkfourdconjugategradienttest ${RTK_LIBRARIES})
add_test(rtkfourdconjugategradienttest ${EXECUTABLE_OUTPUT_PATH}/rtkfourdconjugategradienttest)
ADD_CUDA_TEST(rtkfourdconjugategradient rtkfourdconjugategradienttest.cxx)

add_executable(rtkwarpfourdtoprojectionstacktest rtkwarpfourdtoprojectionstacktest.cxx)
target_link_libraries(rtkwarpfourdtoprojectionstacktest ${RTK_LIBRARIES})
add_test(rtkwarpfourdtoprojectionstacktest ${EXECUTABLE_OUTPUT_PATH}/rtkwarpfourdtoprojectionstacktest)
ADD_CUDA_TEST(rtkwarpfourdtoprojectionstack rtkwarpfourdtoprojectionstacktest.cxx)

add_executable(rtkwarpprojectionstacktofourdtest rtkwarpprojectionstacktofourdtest.cxx)
target_link_libraries(rtkwarpprojectionstacktofourdtest ${RTK_LIBRARIES})
add_test(rtkwarpprojectionstacktofourdtest ${EXECUTABLE_OUTPUT_PATH}/rtkwarpprojectionstacktofourdtest)
ADD_CUDA_TEST(rtkwarpprojectionstacktofourd rtkwarpprojectionstacktofourdtest.cxx)

add_executable(rtkcylindricaldetectorreconstructiontest rtkcylindricaldetectorreconstructiontest.cxx)
target_link_libraries(rtkcylindricaldetectorreconstructiontest ${RTK_LIBRARIES})
add_test(rtkcylindricaldetectorreconstructiontest ${EXECUTABLE_OUTPUT_PATH}/rtkcylindricaldetectorreconstructiontest)
ADD_CUDA_TEST(rtkcylindricaldetectorreconstruction rtkcylindricaldetectorreconstructiontest.cxx)

add_executable(rtkadjointoperatorstest rtkadjointoperatorstest.cxx)
target_link_libraries(rtkadjointoperatorstest ${RTK_LIBRARIES})
add_test(rtkadjointoperatorstest ${EXECUTABLE_OUTPUT_PATH}/rtkadjointoperatorstest)
ADD_CUDA_TEST(rtkadjointoperators rtkadjointoperatorstest.cxx)

add_executable(rtkfourdadjointoperatorstest rtkfourdadjointoperatorstest.cxx)
target_link_libraries(rtkfourdadjointoperatorstest ${RTK_LIBRARIES})
RTK_ADD_TEST(NAME rtkfourdadjointoperatorstest
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkfourdadjointoperatorstest
             DATA{Data/Input/Phases/phases_slow.txt})

add_executable(rtkinterpolatesplatadjointtest rtkinterpolatesplatadjointtest.cxx)
target_link_libraries(rtkinterpolatesplatadjointtest ${RTK_LIBRARIES})
RTK_ADD_TEST(NAME rtkinterpolatesplatadjointtest
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkinterpolatesplatadjointtest
             DATA{Data/Input/Phases/phases_slow.txt})

add_executable(rtklaplaciantest rtklaplaciantest.cxx)
target_link_libraries(rtklaplaciantest ${RTK_LIBRARIES})
RTK_ADD_TEST(NAME rtklaplaciantest 
	     COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtklaplaciantest
	     DATA{Data/Baseline/Laplacian/Laplacian.mha})
RTK_ADD_CUDA_TEST(rtklaplacian rtklaplaciantest.cxx "Data/Baseline/Laplacian/Laplacian.mha")

add_executable(rtktotalvariationtest rtktotalvariationtest.cxx)
target_link_libraries(rtktotalvariationtest ${RTK_LIBRARIES})
add_test(rtktotalvariationtest ${EXECUTABLE_OUTPUT_PATH}/rtktotalvariationtest)

add_executable(rtkgradienttest rtkgradienttest.cxx)
target_link_libraries(rtkgradienttest ${RTK_LIBRARIES})
add_test(rtkgradienttest ${EXECUTABLE_OUTPUT_PATH}/rtkgradienttest)

add_executable(rtkdivergencetest rtkdivergencetest.cxx)
target_link_libraries(rtkdivergencetest ${RTK_LIBRARIES})
add_test(rtkdivergencetest ${EXECUTABLE_OUTPUT_PATH}/rtkdivergencetest)

add_executable(rtklagcorrectiontest rtklagcorrectiontest.cxx)
target_link_libraries(rtklagcorrectiontest ${RTK_LIBRARIES})
add_test(rtklagcorrectiontest ${EXECUTABLE_OUTPUT_PATH}/rtklagcorrectiontest)
ADD_CUDA_TEST(rtklagcorrection rtklagcorrectiontest.cxx)

add_executable(rtkconjugategradienttest rtkconjugategradienttest.cxx)
target_link_libraries(rtkconjugategradienttest ${RTK_LIBRARIES})
add_test(rtkconjugategradienttest ${EXECUTABLE_OUTPUT_PATH}/rtkconjugategradienttest)

add_executable(rtkwarptest rtkwarptest.cxx)
target_link_libraries(rtkwarptest ${RTK_LIBRARIES})
add_test(rtkwarptest ${EXECUTABLE_OUTPUT_PATH}/rtkwarptest)

add_executable(rtkI0estimationtest rtkI0estimationtest.cxx)
target_link_libraries(rtkI0estimationtest ${RTK_LIBRARIES})
add_test(rtkI0estimationtest ${EXECUTABLE_OUTPUT_PATH}/rtkI0estimationtest)

if(ITK_VERSION VERSION_GREATER 4.3.2)
  add_executable(rtkselectoneprojpercycletest rtkselectoneprojpercycletest.cxx)
  target_link_libraries(rtkselectoneprojpercycletest ${RTK_LIBRARIES})
  add_test(rtkselectoneprojpercycletest ${EXECUTABLE_OUTPUT_PATH}/rtkselectoneprojpercycletest)
endif()

# We cannot compile these tests using CPU if GPU is present
# This is because of rtkIterativeConeBeamReconstructionFilter
if(NOT RTK_USE_CUDA)
  add_executable(rtkiterativefdktest rtkiterativefdktest.cxx)
  target_link_libraries(rtkiterativefdktest ${RTK_LIBRARIES})
  add_test(rtkiterativefdktest ${EXECUTABLE_OUTPUT_PATH}/rtkiterativefdktest)

  add_executable(rtkconjugategradientreconstructiontest rtkconjugategradientreconstructiontest.cxx)
  target_link_libraries(rtkconjugategradientreconstructiontest ${RTK_LIBRARIES})
  add_test(rtkconjugategradientreconstructiontest ${EXECUTABLE_OUTPUT_PATH}/rtkconjugategradientreconstructiontest)

  add_executable(rtkfourdroostertest rtkfourdroostertest.cxx)
  target_link_libraries(rtkfourdroostertest ${RTK_LIBRARIES})
  add_test(rtkfourdroostertest ${EXECUTABLE_OUTPUT_PATH}/rtkfourdroostertest)

  add_executable(rtkadmmwaveletstest rtkadmmwaveletstest.cxx)
  target_link_libraries(rtkadmmwaveletstest RTK)
  add_test(rtkadmmwaveletstest ${EXECUTABLE_OUTPUT_PATH}/rtkadmmwaveletstest)

  add_executable(rtkadmmtotalvariationtest rtkadmmtotalvariationtest.cxx)
  target_link_libraries(rtkadmmtotalvariationtest ${RTK_LIBRARIES})
  RTK_ADD_TEST(NAME rtkadmmtotalvariationtest
               COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkadmmtotalvariationtest
               DATA{Data/Input/Phases/phases.txt}
               DATA{Data/Input/Phases/phases_3projs.txt})

  add_executable(rtkregularizedconjugategradienttest rtkregularizedconjugategradienttest.cxx)
  target_link_libraries(rtkregularizedconjugategradienttest RTK)
  add_test(rtkregularizedconjugategradienttest ${EXECUTABLE_OUTPUT_PATH}/rtkregularizedconjugategradienttest)
  
  add_executable(rtkcyclicdeformationtest rtkcyclicdeformationtest.cxx)
  target_link_libraries(rtkcyclicdeformationtest RTK)
  add_test(rtkcyclicdeformationtest ${EXECUTABLE_OUTPUT_PATH}/rtkcyclicdeformationtest)
endif()
ADD_CUDA_TEST(rtkiterativefdk rtkiterativefdktest.cxx)
ADD_CUDA_TEST(rtkconjugategradientreconstruction rtkconjugategradientreconstructiontest.cxx)
ADD_CUDA_TEST(rtkfourdrooster rtkfourdroostertest.cxx)
ADD_CUDA_TEST(rtkadmmwavelets rtkadmmwaveletstest.cxx)
RTK_ADD_CUDA_TEST(rtkadmmtotalvariation rtkadmmtotalvariationtest.cxx "Data/Input/Phases/phases.txt" "Data/Input/Phases/phases_3projs.txt")
ADD_CUDA_TEST(rtkregularizedconjugategradient rtkregularizedconjugategradienttest.cxx)
ADD_CUDA_TEST(rtkcudaraycastadjointoperatorstest rtkcudaraycastadjointoperatorstest.cxx)
ADD_CUDA_TEST(rtkcyclicdeformation rtkcyclicdeformationtest.cxx)

add_executable(rtkwaveletstest rtkwaveletstest.cxx)
target_link_libraries(rtkwaveletstest RTK)
add_test(rtkwaveletstest ${EXECUTABLE_OUTPUT_PATH}/rtkwaveletstest)

# Test headers
file(GLOB RTK_ALL_HEADERS RELATIVE ${PROJECT_SOURCE_DIR}/code "${PROJECT_SOURCE_DIR}/code/rtk*.h")
list(SORT RTK_ALL_HEADERS)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/rtkheadertest.cxx "#ifdef USE_CUDA\n")
file(GLOB RTK_CUDA_HEADERS RELATIVE ${PROJECT_SOURCE_DIR}/code "${PROJECT_SOURCE_DIR}/code/rtkCuda*.h")
list(SORT RTK_CUDA_HEADERS)
foreach(RTK_CUDA_HEADER ${RTK_CUDA_HEADERS})
  file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/rtkheadertest.cxx "# include \"${RTK_CUDA_HEADER}\"\n")
  list(REMOVE_ITEM RTK_ALL_HEADERS ${RTK_CUDA_HEADER})
endforeach()
file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/rtkheadertest.cxx "#endif //USE_CUDA\n")

foreach(RTK_ALL_HEADER ${RTK_ALL_HEADERS})
  file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/rtkheadertest.cxx "#include \"${RTK_ALL_HEADER}\"\n")
endforeach()
file(READ rtkheadertestfooter.cxx RTKHEADERTESTFOOTER)
file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/rtkheadertest.cxx "\n\n${RTKHEADERTESTFOOTER}")

add_executable(rtkheadertest ${CMAKE_CURRENT_BINARY_DIR}/rtkheadertest.cxx)
if(RTK_USE_CUDA)
  set_target_properties(rtkheadertest PROPERTIES COMPILE_FLAGS -DUSE_CUDA)
endif()

target_link_libraries(rtkheadertest ${RTK_LIBRARIES})
add_test(rtkheadertest ${EXECUTABLE_OUTPUT_PATH}/rtkheadertest)

# Test the manager used to automatically clean up the gengetopt args_info structures
add_executable(rtkargsinfomanagertest rtkargsinfomanagertest.cxx)
target_link_libraries(rtkargsinfomanagertest RTK)
add_test(rtkargsinfomanagertest ${EXECUTABLE_OUTPUT_PATH}/rtkargsinfomanagertest)

add_executable(rtkgeometryclonetest rtkgeometryclonetest.cxx)
target_link_libraries(rtkgeometryclonetest ${RTK_LIBRARIES})
add_test(rtkgeometryclonetest ${EXECUTABLE_OUTPUT_PATH}/rtkgeometryclonetest)

add_executable(rtkgeometryfrommatrixtest rtkgeometryfrommatrixtest.cxx)
target_link_libraries(rtkgeometryfrommatrixtest ${RTK_LIBRARIES})
add_test(rtkgeometryfrommatrixtest ${EXECUTABLE_OUTPUT_PATH}/rtkgeometryfrommatrixtest)

add_executable(rtkoratest rtkoratest.cxx)
target_link_libraries(rtkoratest ${RTK_LIBRARIES})
RTK_ADD_TEST(NAME rtkoratest
             COMMAND ${EXECUTABLE_OUTPUT_PATH}/rtkoratest
             DATA{Data/Input/Ora/0_afterLog.ora.xml,0_afterLog.mhd,0_afterLog.raw}
             DATA{Data/Baseline/Ora/geometry.xml,attenuation.mha})


