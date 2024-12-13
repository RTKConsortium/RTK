configure_file (${CMAKE_CURRENT_SOURCE_DIR}/rtkTestConfiguration.h.in
  ${RTK_BINARY_DIR}/rtkTestConfiguration.h)

#-----------------------------------------------------------------------------
# Find ITK.
# Required to include ITK_USE_FILE in order to Register IO factories
# Force requested modules to be RTK dependencies only, otherwise all
# available factories will try to register themselves.
if (NOT ITK_DIR)
  set(ITK_DIR ${ITK_BINARY_DIR}/CMakeTmp)
endif()
find_package(ITK REQUIRED COMPONENTS ${ITK_MODULE_RTK_DEPENDS})
include(${ITK_USE_FILE})

#-----------------------------------------------------------------------------
# rtk_add_test(testname testfile [DATA{}])
# Add regular test as opposed to tests depending on CUDA
function(rtk_add_test testname testfile)
  add_executable(${testname} ${testfile})
  target_link_libraries( ${testname} ${RTK_LIBRARIES} ${ITK_LIBRARIES})
  itk_add_test(NAME ${testname}
    COMMAND itkTestDriver
    $<TARGET_FILE:${testname}>
    ${ARGN})
endfunction()

#-----------------------------------------------------------------------------
# rtk_add_cuda_test(testname testfile [DATA{}])
# Add test depending on CUDA
function(rtk_add_cuda_test testname testfile)
  if(RTK_USE_CUDA)
    add_executable(${testname} ${testfile})
    target_link_libraries( ${testname} ${RTK_LIBRARIES} ${ITK_LIBRARIES})
    itk_add_test(NAME ${testname}
      COMMAND itkTestDriver
      $<TARGET_FILE:${testname}>
      ${ARGN})
    set_target_properties(${testname} PROPERTIES COMPILE_FLAGS -DUSE_CUDA)
  endif()
endfunction()


itk_module_test()

# Use sha512 algorithm to generate content link
if(NOT ITK_SOURCE_DIR)
  set(ExternalData_LINK_CONTENT SHA512)
endif()

rtk_add_test(rtkFDKTest rtkfdktest.cxx)
rtk_add_cuda_test(rtkFDKCudaTest rtkfdktest.cxx)

rtk_add_cuda_test(rtkFDKProjWeightCompCudaTest rtkfdkprojweightcompcudatest.cxx)

rtk_add_test(rtkFBPParallelTest rtkfbpparalleltest.cxx)

rtk_add_test(rtkImportTest rtkimporttest.cxx)
rtk_add_cuda_test(rtkImportCudaTest rtkimporttest.cxx)

rtk_add_cuda_test(rtkCropFilterCudaTest rtkcroptest.cxx)

rtk_add_test(rtkMotionCompensatedFDKTest rtkmotioncompensatedfdktest.cxx)

rtk_add_test(rtkDisplacedDetectorTest rtkdisplaceddetectortest.cxx)
rtk_add_cuda_test(rtkDisplacedDetectorCudaTest rtkdisplaceddetectortest.cxx)
rtk_add_cuda_test(rtkDisplacedDetectorCompCudaTest rtkdisplaceddetectorcompcudatest.cxx)

rtk_add_test(rtkDisplacedDetectorCompOffsetTest rtkdisplaceddetectorcompoffsettest.cxx)

rtk_add_test(rtkShortScanTest rtkshortscantest.cxx)
rtk_add_cuda_test(rtkShortScanCudaTest rtkshortscantest.cxx)
rtk_add_cuda_test(rtkShortScanCompCudaTest rtkshortscancompcudatest.cxx)

rtk_add_test(rtkRampFilterTest rtkrampfiltertest.cxx)
rtk_add_cuda_test(rtkRampFilterCudaTest rtkrampfiltertest.cxx)

rtk_add_test(rtkRampFilterTest2 rtkrampfiltertest2.cxx)
rtk_add_cuda_test(rtkRampFilterCudaTest2 rtkrampfiltertest2.cxx)

rtk_add_test(rtkHilbertFilterTest rtkhilbertfiltertest.cxx)

rtk_add_test(rtkScatterGlareFilterTest rtkscatterglarefiltertest.cxx)
rtk_add_cuda_test(rtkScatterGlareFilterCudaTest rtkscatterglarefiltertest.cxx)

rtk_add_test(rtkScatterGlareFilterNoFFTWTest rtkscatterglarefiltertest.cxx)

rtk_add_test(rtkGainCorrectionTest rtkgaincorrectiontest.cxx)
rtk_add_cuda_test(rtkGainCorrectionCudaTest rtkgaincorrectiontest.cxx)

rtk_add_test(rtkForwardProjectionTest rtkforwardprojectiontest.cxx)
rtk_add_cuda_test(rtkForwardProjectionCudaTest rtkforwardprojectiontest.cxx)
rtk_add_test(rtkForwardAttenuatedProjectionTest rtkforwardattenuatedprojectiontest.cxx)
rtk_add_test(rtkZengProjectionTest rtkzengforwardprojectiontest.cxx)
rtk_add_test(rtkMaximumIntensityProjectionTest rtkmaximumintensityprojectiontest.cxx)

rtk_add_test(rtkGeometryFileTest rtkgeometryfiletest.cxx)

rtk_add_test(rtkReg23ProjectionGeometryTest rtkTestReg23ProjectionGeometry.cxx)

rtk_add_test(rtkFOVTest rtkfovtest.cxx)

rtk_add_test(rtkBinningTest rtkbinningtest.cxx)

rtk_add_test(rtkl0GradientNormTest rtkl0gradientnormtest.cxx)

rtk_add_test(rtkWaterPreCorrectionTest rtkwaterprecorrectiontest.cxx)

rtk_add_test(rtkLUTBasedVarI0RawToAttTest rtklutbasedvarI0rawtoatttest.cxx)

rtk_add_test(rtkDecomposeSpectralProjectionsTest rtkdecomposespectralprojectionstest.cxx
  DATA{Input/Spectral/incident_spectrum.mha}
  DATA{Input/Spectral/detector_response.mha}
  DATA{Input/Spectral/material_attenuations.mha})

rtk_add_test(rtkSpectralOneStepTest rtkspectralonesteptest.cxx
  DATA{Input/Spectral/OneStep/no_vector_incident_spectrum_64_rows.mha}
  DATA{Input/Spectral/OneStep/incident_spectrum_64_rows.mha}
  DATA{Input/Spectral/detector_response.mha}
  DATA{Input/Spectral/material_attenuations.mha})

rtk_add_test(rtkVectorImageConvertersTest rtkvectorimageconverterstest.cxx)

rtk_add_test(rtkAmsterdamShroudTest rtkamsterdamshroudtest.cxx
  DATA{Baseline/AmsterdamShroud/Amsterdam_crop.mha}
  DATA{Baseline/AmsterdamShroud/Amsterdam.mha})

rtk_add_test(rtkVarianTest rtkvariantest.cxx
  DATA{Input/Varian/raw.hnd}
  DATA{Input/Varian/acqui.xml}
  DATA{Input/Varian/Proj_00000.xim}
  DATA{Input/Varian/acqui_probeam.xml}
  DATA{Input/Varian/image_00052.hnc}
  DATA{Baseline/Varian/geometry.xml}
  DATA{Baseline/Varian/attenuation.mha}
  DATA{Baseline/Varian/geometryProBeam.xml}
  DATA{Baseline/Varian/attenuationProBeam.mha}
  DATA{Baseline/Varian/geometryHnc.xml}
  DATA{Baseline/Varian/attenuationHnc.mha})

rtk_add_test(rtkElektaTest rtkelektatest.cxx
  DATA{Input/Elekta/IMAGE.DBF}
  DATA{Input/Elekta/FRAME.DBF}
  DATA{Input/Elekta/raw.his}
  DATA{Input/Elekta/_Frames.xml}
  DATA{Baseline/Elekta/geometry.xml}
  DATA{Baseline/Elekta/attenuation.mha}
  DATA{Baseline/Elekta/geometry5.xml})

rtk_add_test(rtkLUTTest rtkluttest.cxx
  DATA{Input/Elekta/raw.his}
  DATA{Baseline/Elekta/attenuation.mha})

rtk_add_test(rtkImagXTest rtkimagxtest.cxx
  DATA{Input/ImagX/1.dcm}
  DATA{Input/ImagX/calibration.xml}
  DATA{Input/ImagX/room.xml}
  DATA{Input/ImagX/raw.xml,raw.raw}
  DATA{Baseline/ImagX/geo.xml}
  DATA{Baseline/ImagX/attenuation.mha}
  DATA{Baseline/ImagX/attenuationDCM.mha})

rtk_add_test(rtkEdfTest rtkedftest.cxx
  DATA{Input/ESRF/raw.edf,dark.edf,refHST0000.edf}
  DATA{Baseline/ESRF/attenuation.mha})

rtk_add_test(rtkDigisensTest rtkdigisenstest.cxx
  DATA{Input/Digisens/calibration.cal}
  DATA{Input/Digisens/ima0010.tif}
  DATA{Baseline/Digisens/geometry.xml}
  DATA{Baseline/Digisens/attenuation.mha})

rtk_add_test(rtkXRadTest rtkxradtest.cxx
  DATA{Input/XRad/SolidWater_HiGain1x1.header}
  DATA{Input/XRad/SolidWater_HiGain1x1_firstProj.header,flat.header,flat.img,dark.header,dark.img,SolidWater_HiGain1x1_firstProj.img}
  DATA{Baseline/XRad/geometry.xml}
  DATA{Baseline/XRad/attenuation.mha})

rtk_add_test(rtkProjectGeometricPhantomTest rtkprojectgeometricphantomtest.cxx
  DATA{Input/GeometricPhantom/SheppLogan.txt})

rtk_add_test(rtkDrawGeometricPhantomTest rtkdrawgeometricphantomtest.cxx
  DATA{Input/GeometricPhantom/SheppLogan.txt}
  DATA{Input/GeometricPhantom/Geometries.txt})

rtk_add_test(rtkWeidingerForwardModelTest rtkweidingerforwardmodeltest.cxx
  DATA{Input/Spectral/OneStep/materialProjections.mha}
  DATA{Input/Spectral/OneStep/photonCounts.mha}
  DATA{Input/Spectral/OneStep/spectrum.mha}
  DATA{Input/Spectral/OneStep/projOfOnes.mha}
  DATA{Input/Spectral/OneStep/binnedDetectorResponse.csv}
  DATA{Input/Spectral/OneStep/materialAttenuations.csv}
  DATA{Baseline/Spectral/OneStep/out1.mha}
  DATA{Baseline/Spectral/OneStep/out2.mha})

rtk_add_test(rtkNewtonUpdateTest rtknewtonupdatetest.cxx
  DATA{Input/Spectral/OneStep/gradient.mha}
  DATA{Input/Spectral/OneStep/hessian.mha}
  DATA{Baseline/Spectral/OneStep/newtonUpdate.mha})

rtk_add_test(rtkSartTest rtksarttest.cxx)
rtk_add_cuda_test(rtkSartCudaTest rtksarttest.cxx)

rtk_add_test(rtkOsemTest rtkosemtest.cxx)
rtk_add_cuda_test(rtkOsemCudaTest rtkosemtest.cxx)

rtk_add_test(rtkFourDSartTest rtkfourdsarttest.cxx)
rtk_add_cuda_test(rtkFourDSartCudaTest rtkfourdsarttest.cxx)

rtk_add_test(rtkFourDConjugateGradientTest rtkfourdconjugategradienttest.cxx)
rtk_add_cuda_test(rtkFourDConjugateGradientCudaTest rtkfourdconjugategradienttest.cxx)

rtk_add_test(rtkWarpFourDToProjectionStackTest rtkwarpfourdtoprojectionstacktest.cxx)
rtk_add_cuda_test(rtkWarpFourDToProjectionStackCudaTest rtkwarpfourdtoprojectionstacktest.cxx)

rtk_add_test(rtkWarpProjectionStackToFourDTest rtkwarpprojectionstacktofourdtest.cxx)
rtk_add_cuda_test(rtkWarpProjectionStackToFourDCudaTest rtkwarpprojectionstacktofourdtest.cxx)

rtk_add_test(rtkCylindricalDetectorReconstructionTest rtkcylindricaldetectorreconstructiontest.cxx)
rtk_add_cuda_test(rtkCylindricalDetectorReconstructionCudaTest rtkcylindricaldetectorreconstructiontest.cxx)


rtk_add_test(rtkAdjointOperatorsTest rtkadjointoperatorstest.cxx)
rtk_add_cuda_test(rtkAdjointOperatorsCudaTest rtkadjointoperatorstest.cxx)

rtk_add_test(rtkFourDAdjointOperatorsTest rtkfourdadjointoperatorstest.cxx
  DATA{Input/Phases/phases_slow.txt})

rtk_add_test(rtkInterpolateSplatAdjointTest rtkinterpolatesplatadjointtest.cxx
  DATA{Input/Phases/phases_slow.txt})

rtk_add_test(rtkLaplacianTest rtklaplaciantest.cxx
  DATA{Baseline/Laplacian/Laplacian.mha})
rtk_add_cuda_test(rtkLaplacianCudaTest rtklaplaciantest.cxx
  DATA{Baseline/Laplacian/Laplacian.mha})

rtk_add_test(rtkTotalVariationTest rtktotalvariationtest.cxx)

rtk_add_test(rtkGradientTest rtkgradienttest.cxx)

rtk_add_test(rtkDivergenceTest rtkdivergencetest.cxx)

rtk_add_test(rtkLagCorrectionTest rtklagcorrectiontest.cxx)
rtk_add_cuda_test(rtkLagCorrectionCudaTest rtklagcorrectiontest.cxx)

rtk_add_test(rtkConjugateGradientTest rtkconjugategradienttest.cxx)

rtk_add_test(rtkWarpTest rtkwarptest.cxx)

rtk_add_test(rtkI0EstimationTest rtkI0estimationtest.cxx)

rtk_add_test(rtkSelectOneProjPerCycleTest rtkselectoneprojpercycletest.cxx)

# We cannot compile these tests using CPU if GPU is present
# This is because of rtkIterativeConeBeamReconstructionFilter
if(NOT RTK_USE_CUDA)
  rtk_add_test(rtkIterativeFDKTest rtkiterativefdktest.cxx)

  rtk_add_test(rtkConjugateGradientReconstructionTest rtkconjugategradientreconstructiontest.cxx)

  rtk_add_test(rtkFourDRoosterTest rtkfourdroostertest.cxx)

  rtk_add_test(rtkADMMWaveletsTest rtkadmmwaveletstest.cxx)

  rtk_add_test(rtkADMMTotalVariationTest rtkadmmtotalvariationtest.cxx
    DATA{Input/Phases/phases.txt}
    DATA{Input/Phases/phases_3projs.txt})

  rtk_add_test(rtkRegularizedConjugateGradientTest rtkregularizedconjugategradienttest.cxx)

  rtk_add_test(rtkCyclicDeformationTest rtkcyclicdeformationtest.cxx)
endif()

rtk_add_cuda_test(rtkIterativeFDKCudaTest rtkiterativefdktest.cxx)
rtk_add_cuda_test(rtkConjugateGradientReconstructionCudaTest rtkconjugategradientreconstructiontest.cxx)
rtk_add_cuda_test(rtkFourDRoosterCudaTest rtkfourdroostertest.cxx)
rtk_add_cuda_test(rtkADMMWaveletsCudaTest rtkadmmwaveletstest.cxx)
rtk_add_cuda_test(rtkADMMTotalVariationCudaTest rtkadmmtotalvariationtest.cxx
  DATA{Input/Phases/phases.txt}
  DATA{Input/Phases/phases_3projs.txt})
rtk_add_cuda_test(rtkRegularizedConjugateGradientCudaTest rtkregularizedconjugategradienttest.cxx)
rtk_add_cuda_test(rtkCudaRaycastAdjointOperatorsCudaTest rtkcudaraycastadjointoperatorstest.cxx)
rtk_add_cuda_test(rtkCyclicDeformationCudaTest rtkcyclicdeformationtest.cxx)

rtk_add_test(rtkWaveletsTest rtkwaveletstest.cxx)

# Test the manager used to automatically clean up the gengetopt args_info structures
rtk_add_test(rtkArgsInfoManagerTest rtkargsinfomanagertest.cxx)

rtk_add_test(rtkGeometryCloneTest rtkgeometryclonetest.cxx)
rtk_add_test(rtkGeometryFromMatrixTest rtkgeometryfrommatrixtest.cxx)
rtk_add_test(rtkParallelGeometryFromMatrixTest rtkparallelgeometryfrommatrixtest.cxx)

rtk_add_test(rtkOraTest rtkoratest.cxx
  DATA{Input/Ora/0_afterLog.ora.xml,0_afterLog.mhd,0_afterLog.raw}
  DATA{Baseline/Ora/geometry.xml}
  DATA{Input/Ora/2006137_20220918183246810.ora.xml,2006137_20220918183246810.mhd}
  DATA{Baseline/Ora/geometry_yawtilt.xml}
  DATA{Baseline/Ora/geometry_optitrack.xml}
  DATA{Baseline/Ora/attenuation.mha})

rtk_add_test(rtkBioscanTest rtkbioscantest.cxx
  DATA{Input/Bioscan/bioscan.dcm}
  DATA{Baseline/Bioscan/geometry.xml})

rtk_add_test(rtkVarianceReconstructionTest rtkvariancereconstructiontest.cxx)

if(ITK_WRAP_PYTHON)
  itk_python_add_test(NAME rtkFirstReconstructionPythonTest COMMAND rtkFirstReconstruction.py ${CMAKE_CURRENT_BINARY_DIR}/rtkFirstReconstruction.mha)
  itk_python_add_test(NAME rtkMaximumIntensityPythonTest COMMAND rtkMaximumIntensity.py  ${CMAKE_CURRENT_BINARY_DIR}/rtkFirstReconstruction.mha)
endif()
