#=========================================================
# Propagate cmake options in a header file
CONFIGURE_FILE (${CMAKE_CURRENT_SOURCE_DIR}/rtkConfiguration.h.in
  ${CMAKE_BINARY_DIR}/rtkConfiguration.h)
SET (RTK_LIBRARIES RTK lpsolve55)
#=========================================================

#=========================================================
SET(RTK_LIBRARY_FILES
            rtkHisImageIO.cxx
            rtkHisImageIOFactory.cxx
            rtkElektaSynergyGeometryReader.cxx
            rtkElektaXVI5GeometryXMLFile.cxx
            rtkDbf.cxx
            rtkHndImageIO.cxx
            rtkHndImageIOFactory.cxx
            rtkHncImageIO.cxx
            rtkHncImageIOFactory.cxx
            rtkVarianObiXMLFileReader.cxx
            rtkVarianObiGeometryReader.cxx
            rtkEdfImageIO.cxx
            rtkEdfImageIOFactory.cxx
            rtkImagXImageIO.cxx
            rtkImagXImageIOFactory.cxx
            rtkDCMImagXImageIO.cxx
            rtkDCMImagXImageIOFactory.cxx
            rtkImagXXMLFileReader.cxx
            rtkThreeDCircularProjectionGeometry.cxx
            rtkReg23ProjectionGeometry.cxx
            rtkThreeDCircularProjectionGeometryXMLFile.cxx
            rtkGeometricPhantomFileReader.cxx
            rtkDigisensGeometryXMLFileReader.cxx
            rtkDigisensGeometryReader.cxx
            rtkXRadGeometryReader.cxx
            rtkXRadImageIO.cxx
            rtkXRadImageIOFactory.cxx
            rtkIOFactories.cxx
            rtkConvertEllipsoidToQuadricParametersFunction.cxx
            rtkDrawQuadricSpatialObject.cxx)
IF(RTK_TIME_EACH_FILTER)
    SET(RTK_LIBRARY_FILES
            ${RTK_LIBRARY_FILES}
            rtkGlobalTimer.cxx
            rtkGlobalTimerProbesCollector.cxx
            rtkWatcherForTimer.cxx
            rtkTimeProbesCollectorBase.cxx)
ENDIF()
ADD_LIBRARY(RTK ${RTK_LIBRARY_FILES})

TARGET_LINK_LIBRARIES(RTK ${ITK_LIBRARIES})
TARGET_LINK_LIBRARIES(RTK lpsolve55)
TARGET_LINK_LIBRARIES(RTK ${BZIP2_LIBRARIES})
#=========================================================

#=========================================================
# OpenCL library stuff
IF(RTK_USE_OPENCL)
  SET(RTK_LIBRARIES ${RTK_LIBRARIES} rtkopencl)
  ADD_LIBRARY(rtkopencl
              rtkOpenCLUtilities.cxx
              rtkOpenCLFDKBackProjectionImageFilter.cxx)
  TARGET_LINK_LIBRARIES(rtkopencl ${OPENCL_LIBRARIES} ${ITK_LIBRARIES})

  FILE(GLOB openclSRCs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cl)
  ADD_CUSTOM_TARGET(openclSRCCopy ALL COMMENT "Copying OpenCL source files")
  FOREACH(openclSRC ${openclSRCs})
    ADD_CUSTOM_COMMAND(TARGET openclSRCCopy
                       COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${openclSRC}
                                                        ${RTK_BINARY_DIR}/${openclSRC})
  ENDFOREACH(openclSRC)
  ADD_DEPENDENCIES(rtkopencl openclSRCCopy)
ENDIF(RTK_USE_OPENCL)
#=========================================================

#=========================================================
# Cuda library stuff
IF (RTK_USE_CUDA)
  SET (RTK_LIBRARIES ${RTK_LIBRARIES} rtkcuda ITKCudaCommon ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_CUFFT_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
  SET (rtkcuda_CUDA_FILES
       rtkCudaCropImageFilter.cu
       rtkCudaUtilities.cu
       rtkCudaFDKBackProjectionImageFilter.cu
       rtkCudaFFTConvolutionImageFilter.cu
       rtkCudaFDKWeightProjectionFilter.cu
       rtkCudaDisplacedDetectorImageFilter.cu
       rtkCudaParkerShortScanImageFilter.cu
       rtkCudaForwardProjectionImageFilter.cu
       rtkCudaBackProjectionImageFilter.cu
       rtkCudaSplatImageFilter.cu
       rtkCudaInterpolateImageFilter.cu
       rtkCudaWarpImageFilter.cu
       rtkCudaForwardWarpImageFilter.cu
       rtkCudaRayCastBackProjectionImageFilter.cu
       rtkCudaTotalVariationDenoisingBPDQImageFilter.cu
       rtkCudaLastDimensionTVDenoisingImageFilter.cu
       rtkCudaConstantVolumeSource.cu
       rtkCudaConstantVolumeSeriesSource.cu
       rtkCudaConjugateGradientImageFilter_3f.cu
       rtkCudaConjugateGradientImageFilter_4f.cu
       rtkCudaAverageOutOfROIImageFilter.cu
       rtkCudaLaplacianImageFilter.cu
       rtkCudaFirstOrderKernels.cu
       rtkCudaWarpForwardProjectionImageFilter.cu
       rtkCudaWarpBackProjectionImageFilter.cu
       rtkCudaCyclicDeformationImageFilter.cu
       rtkCudaLagCorrectionImageFilter.cu
       rtkCudaPolynomialGainCorrectionImageFilter.cu
       )
  # This is a fix for ITK4 to avoid too long cmd lines error
  # (http://www.itk.org/Bug/view.php?id=12198)
  IF(WIN32)
    GET_DIRECTORY_PROPERTY(dirInc INCLUDE_DIRECTORIES)
    FIND_PATH(itkMacroPath itkMacro.h ${dirInc})
    FIND_PATH(itkConfigurePath itkConfigure.h ${dirInc})
    FIND_PATH(rtkConfigurationPath rtkConfiguration.h ${dirInc})
    SET_DIRECTORY_PROPERTIES(PROPERTIES INCLUDE_DIRECTORIES "${itkMacroPath};${itkConfigurePath};${rtkConfigurationPath}")
  ENDIF()

  CUDA_COMPILE (rtkcuda_CUDA_WRAPPERS ${rtkcuda_CUDA_FILES})

  # Rollback to all includes
  IF(WIN32)
    SET_DIRECTORY_PROPERTIES(PROPERTIES INCLUDE_DIRECTORIES "${dirInc}")
  ENDIF()

  ADD_LIBRARY(rtkcuda
              rtkCudaFDKWeightProjectionFilter.cxx
              rtkCudaDisplacedDetectorImageFilter.cxx
              rtkCudaParkerShortScanImageFilter.cxx
              rtkCudaCropImageFilter.cxx
              rtkCudaFDKBackProjectionImageFilter.cxx
              rtkCudaFDKConeBeamReconstructionFilter.cxx
              rtkCudaIterativeFDKConeBeamReconstructionFilter.cxx
              rtkCudaBackProjectionImageFilter.cxx
              rtkCudaSplatImageFilter.cxx
              rtkCudaInterpolateImageFilter.cxx
              rtkCudaWarpImageFilter.cxx
              rtkCudaForwardWarpImageFilter.cxx
              rtkCudaRayCastBackProjectionImageFilter.cxx
              rtkCudaTotalVariationDenoisingBPDQImageFilter.cxx
              rtkCudaLastDimensionTVDenoisingImageFilter.cxx
              rtkCudaWarpForwardProjectionImageFilter.cxx
              rtkCudaWarpBackProjectionImageFilter.cxx
              rtkCudaConstantVolumeSource.cxx
              rtkCudaConstantVolumeSeriesSource.cxx
              rtkCudaConjugateGradientImageFilter_3f.cxx
              rtkCudaConjugateGradientImageFilter_4f.cxx
              rtkCudaAverageOutOfROIImageFilter.cxx
              rtkCudaLaplacianImageFilter.cxx
              rtkCudaCyclicDeformationImageFilter.cxx
              rtkCudaLagCorrectionImageFilter.cxx
              rtkCudaPolynomialGainCorrectionImageFilter.cxx
              ${rtkcuda_CUDA_WRAPPERS}
              ${rtkcuda_CUDA_FILES})
  SET_TARGET_PROPERTIES (rtkcuda PROPERTIES LINKER_LANGUAGE CXX)
  TARGET_LINK_LIBRARIES(rtkcuda ${CUDA_LIBRARIES} ${CUDA_cufft_LIBRARY} ${CUDA_cublas_LIBRARY} RTK ITKCudaCommon)
ENDIF (RTK_USE_CUDA)
#=========================================================

#=========================================================
# Installation code
IF(NOT RTK_INSTALL_NO_DEVELOPMENT)
  # Generate RTKConfig.cmake for the install tree.
  SET (RTK_USE_FILE "${CMAKE_INSTALL_PREFIX}/${RTK_INSTALL_PACKAGE_DIR}/UseRTK.cmake")
  SET (RTK_INCLUDE_DIRS "${CMAKE_INSTALL_PREFIX}/${RTK_INSTALL_INCLUDE_DIR};${CMAKE_INSTALL_PREFIX}/${RTK_INSTALL_INCLUDE_DIR}/lpsolve")
  SET (RTK_LIBRARY_DIRS "${CMAKE_INSTALL_PREFIX}/${RTK_INSTALL_LIB_DIR}")
  CONFIGURE_FILE (${PROJECT_SOURCE_DIR}/cmake/RTKConfig.cmake.in ${RTK_BINARY_DIR}/CMakeFiles/RTKConfig.cmake @ONLY)
  CONFIGURE_FILE (${PROJECT_SOURCE_DIR}/cmake/RTKConfigVersion.cmake.in ${RTK_BINARY_DIR}/CMakeFiles/RTKConfigVersion.cmake @ONLY)
  INSTALL(FILES ${RTK_BINARY_DIR}/CMakeFiles/RTKConfig.cmake
                ${RTK_BINARY_DIR}/CMakeFiles/RTKConfigVersion.cmake
                ${RTK_SOURCE_DIR}/cmake/UseRTK.cmake
          DESTINATION ${RTK_INSTALL_PACKAGE_DIR})

  # Include .h and .hxx files
  FILE(GLOB __files1 "${PROJECT_SOURCE_DIR}/code/*.h")
  FILE(GLOB __files2 "${PROJECT_SOURCE_DIR}/code/*.hxx")
  INSTALL(FILES ${__files1} ${__files2} ${RTK_BINARY_DIR}/rtkConfiguration.h
    DESTINATION ${RTK_INSTALL_INCLUDE_DIR}
    COMPONENT Development)
  IF(RTK_USE_CUDA)
    FILE(GLOB __files3 "${PROJECT_SOURCE_DIR}/code/*.hcu")
    INSTALL(FILES ${__files3}
      DESTINATION ${RTK_INSTALL_INCLUDE_DIR}
      COMPONENT Development)
  ENDIF(RTK_USE_CUDA)
ENDIF(NOT RTK_INSTALL_NO_DEVELOPMENT)

IF(NOT RTK_INSTALL_NO_LIBRARIES)
  INSTALL(TARGETS RTK
    RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
    LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
    ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)

  IF(RTK_USE_CUDA)
    INSTALL(TARGETS rtkcuda
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  ENDIF(RTK_USE_CUDA)

  IF(OPENCL_FOUND)
    INSTALL(TARGETS rtkopencl
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  ENDIF(OPENCL_FOUND)
ENDIF(NOT RTK_INSTALL_NO_LIBRARIES)
#=========================================================

SET(RTK_LIBRARIES ${RTK_LIBRARIES} PARENT_SCOPE)
