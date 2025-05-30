set(RTK_SRCS
  rtkBioscanGeometryReader.cxx
  rtkBoxShape.cxx
  rtkConditionalMedianImageFilter.cxx
  rtkConvexShape.cxx
  rtkDbf.cxx
  rtkDCMImagXImageIO.cxx
  rtkDCMImagXImageIOFactory.cxx
  rtkDigisensGeometryReader.cxx
  rtkDigisensGeometryXMLFileReader.cxx
  rtkEdfImageIO.cxx
  rtkEdfImageIOFactory.cxx
  rtkElektaSynergyGeometryReader.cxx
  rtkElektaXVI5GeometryXMLFileReader.cxx
  rtkForbildPhantomFileReader.cxx
  rtkGeometricPhantom.cxx
  rtkGlobalResourceProbe.cxx
  rtkHisImageIO.cxx
  rtkHisImageIOFactory.cxx
  rtkHncImageIO.cxx
  rtkHncImageIOFactory.cxx
  rtkHndImageIO.cxx
  rtkHndImageIOFactory.cxx
  rtkImagXImageIO.cxx
  rtkImagXImageIOFactory.cxx
  rtkImagXXMLFileReader.cxx
  rtkIntersectionOfConvexShapes.cxx
  rtkIOFactories.cxx
  rtkOraGeometryReader.cxx
  rtkOraImageIO.cxx
  rtkOraImageIOFactory.cxx
  rtkOraXMLFileReader.cxx
  rtkPhaseReader.cxx
  rtkPhasesToInterpolationWeights.cxx
  rtkQuadricShape.cxx
  rtkReg23ProjectionGeometry.cxx
  rtkSheppLoganPhantom.cxx
  rtkSignalToInterpolationWeights.cxx
  rtkThreeDCircularProjectionGeometry.cxx
  rtkThreeDCircularProjectionGeometryXMLFileReader.cxx
  rtkThreeDCircularProjectionGeometryXMLFileWriter.cxx
  rtkResourceProbesCollector.cxx
  rtkVarianObiGeometryReader.cxx
  rtkVarianObiXMLFileReader.cxx
  rtkVarianProBeamGeometryReader.cxx
  rtkVarianProBeamXMLFileReader.cxx
  rtkWatcherForResourceProbe.cxx
  rtkXimImageIO.cxx
  rtkXimImageIOFactory.cxx
  rtkXRadGeometryReader.cxx
  rtkXRadImageIO.cxx
  rtkXRadImageIOFactory.cxx
  )

#=========================================================
# This is a fix for when using Module_ITKVtkGlue
# (https://gitlab.kitware.com/vtk/vtk/issues/15895)
get_directory_property(defCom COMPILE_DEFINITIONS)
set(defComSansVTK)
foreach(def ${defCom})
  if(NOT def MATCHES "^vtk.*")
    list(APPEND defComSansVTK ${def})
  endif()
endforeach()
set_directory_properties(PROPERTIES COMPILE_DEFINITIONS "${defComSansVTK}")

#=========================================================
# CUDA library
#=========================================================
if (RTK_USE_CUDA)
  set(RTK_SRCS ${RTK_SRCS}
    rtkCudaAverageOutOfROIImageFilter.cxx
    rtkCudaConstantVolumeSeriesSource.cxx
    rtkCudaConstantVolumeSource.cxx
    rtkCudaCropImageFilter.cxx
    rtkCudaCyclicDeformationImageFilter.cxx
    rtkCudaDisplacedDetectorImageFilter.cxx
    rtkCudaFDKBackProjectionImageFilter.cxx
    rtkCudaFDKConeBeamReconstructionFilter.cxx
    rtkCudaFDKWeightProjectionFilter.cxx
    rtkCudaForwardWarpImageFilter.cxx
    rtkCudaInterpolateImageFilter.cxx
    rtkCudaIterativeFDKConeBeamReconstructionFilter.cxx
    rtkCudaLagCorrectionImageFilter.cxx
    rtkCudaLaplacianImageFilter.cxx
    rtkCudaLastDimensionTVDenoisingImageFilter.cxx
    rtkCudaParkerShortScanImageFilter.cxx
    rtkCudaPolynomialGainCorrectionImageFilter.cxx
    rtkCudaRayCastBackProjectionImageFilter.cxx
    rtkCudaSplatImageFilter.cxx
    rtkCudaTotalVariationDenoisingBPDQImageFilter.cxx
    rtkCudaWarpBackProjectionImageFilter.cxx
    rtkCudaWarpForwardProjectionImageFilter.cxx
    rtkCudaWarpImageFilter.cxx)

  set(RTK_SRCS ${RTK_SRCS}
    rtkCudaAverageOutOfROIImageFilter.cu
    rtkCudaBackProjectionImageFilter.cu
    rtkCudaConjugateGradientImageFilter.cu
    rtkCudaConstantVolumeSeriesSource.cu
    rtkCudaConstantVolumeSource.cu
    rtkCudaCropImageFilter.cu
    rtkCudaCyclicDeformationImageFilter.cu
    rtkCudaDisplacedDetectorImageFilter.cu
    rtkCudaFDKBackProjectionImageFilter.cu
    rtkCudaFDKWeightProjectionFilter.cu
    rtkCudaFFTProjectionsConvolutionImageFilter.cu
    rtkCudaFirstOrderKernels.cu
    rtkCudaForwardProjectionImageFilter.cu
    rtkCudaForwardWarpImageFilter.cu
    rtkCudaInterpolateImageFilter.cu
    rtkCudaLagCorrectionImageFilter.cu
    rtkCudaLaplacianImageFilter.cu
    rtkCudaLastDimensionTVDenoisingImageFilter.cu
    rtkCudaParkerShortScanImageFilter.cu
    rtkCudaPolynomialGainCorrectionImageFilter.cu
    rtkCudaRayCastBackProjectionImageFilter.cu
    rtkCudaSplatImageFilter.cu
    rtkCudaTotalVariationDenoisingBPDQImageFilter.cu
    rtkCudaUtilities.cu
    rtkCudaWarpBackProjectionImageFilter.cu
    rtkCudaWarpForwardProjectionImageFilter.cu
    rtkCudaWarpImageFilter.cu
    rtkCudaWeidingerForwardModelImageFilter.cu
    )
endif()
#=========================================================

itk_module_add_library(RTK ${RTK_SRCS})
target_link_libraries(RTK PUBLIC lpsolve55)

if (RTK_USE_CUDA)
  target_link_libraries(RTK PUBLIC CUDA::cufft CUDA::cublas)
  set_property(TARGET RTK PROPERTY CUDA_STANDARD ${CMAKE_CXX_STANDARD})
endif ()
