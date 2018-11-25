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
  rtkElektaXVI5GeometryXMLFile.cxx
  rtkForbildPhantomFileReader.cxx
  rtkGeometricPhantom.cxx
  rtkGeometricPhantomFileReader.cxx
  rtkGlobalTimer.cxx
  rtkGlobalTimerProbesCollector.cxx
  rtkHisImageIO.cxx
  rtkHisImageIOFactory.cxx
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
  rtkQuadricShape.cxx
  rtkReg23ProjectionGeometry.cxx
  rtkSheppLoganPhantom.cxx
  rtkThreeDCircularProjectionGeometry.cxx
  rtkThreeDCircularProjectionGeometryXMLFileReader.cxx
  rtkThreeDCircularProjectionGeometryXMLFileWriter.cxx
  rtkTimeProbesCollectorBase.cxx
  rtkVarianObiGeometryReader.cxx
  rtkVarianObiXMLFileReader.cxx
  rtkVarianProBeamGeometryReader.cxx
  rtkVarianProBeamXMLFileReader.cxx
  rtkWatcherForTimer.cxx
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
  set(RTK_CUDA_FILES
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

  set(CUDA_PROPAGATE_HOST_FLAGS ON)
  set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)#WARNING: mandatory for defining runtime type \MT vs \MD
  CUDA_COMPILE(RTK_CUDA_WRAPPERS ${RTK_CUDA_FILES})

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
    rtkCudaWarpImageFilter.cxx
    ${RTK_CUDA_WRAPPERS}
    ${RTK_CUDA_FILES})
endif ()
#=========================================================

itk_module_add_library(RTK ${RTK_SRCS})
target_link_libraries(RTK LINK_PUBLIC lpsolve55)

if (RTK_USE_CUDA)
  target_link_libraries( RTK LINK_PUBLIC ITKCudaCommon ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_cufft_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} )
endif ()
