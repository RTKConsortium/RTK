#=========================================================
# Propagate cmake options in a header file
configure_file (${CMAKE_CURRENT_SOURCE_DIR}/rtkConfiguration.h.in
  ${CMAKE_BINARY_DIR}/rtkConfiguration.h)
set (RTK_LIBRARIES RTK lpsolve55)
#=========================================================

#=========================================================
set(RTK_LIBRARY_FILES
            rtkHisImageIO.cxx
            rtkHisImageIOFactory.cxx
            rtkElektaSynergyGeometryReader.cxx
            rtkElektaXVI5GeometryXMLFile.cxx
            rtkDbf.cxx
            rtkHndImageIO.cxx
            rtkHndImageIOFactory.cxx
            rtkXimImageIO.cxx
            rtkXimImageIOFactory.cxx
            rtkVarianObiXMLFileReader.cxx
            rtkVarianObiGeometryReader.cxx
            rtkVarianProBeamXMLFileReader.cxx
            rtkVarianProBeamGeometryReader.cxx
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
            rtkDrawQuadricSpatialObject.cxx
            rtkOraXMLFileReader.cxx
            rtkOraGeometryReader.cxx
            rtkOraImageIO.cxx
            rtkOraImageIOFactory.cxx
	    rtkConditionalMedianImageFilter.cxx)

if(RTK_TIME_EACH_FILTER)
    set(RTK_LIBRARY_FILES
            ${RTK_LIBRARY_FILES}
            rtkGlobalTimer.cxx
            rtkGlobalTimerProbesCollector.cxx
            rtkWatcherForTimer.cxx
            rtkTimeProbesCollectorBase.cxx)
endif()
add_library(RTK ${RTK_LIBRARY_FILES})

target_link_libraries(RTK ${ITK_LIBRARIES})
target_link_libraries(RTK lpsolve55)
#=========================================================

#=========================================================
# Cuda library stuff
if (RTK_USE_CUDA)
  set (RTK_LIBRARIES ${RTK_LIBRARIES} rtkcuda ITKCudaCommon ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_CUFFT_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
  set (rtkcuda_CUDA_FILES
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
  if(WIN32)
    get_directory_property(dirInc INCLUDE_DIRECTORIES)
    find_path(itkMacroPath itkMacro.h ${dirInc})
    find_path(itkConfigurePath itkConfigure.h ${dirInc})
    find_path(rtkConfigurationPath rtkConfiguration.h ${dirInc})
    set_directory_properties(PROPERTIES INCLUDE_DIRECTORIES "${itkMacroPath};${itkConfigurePath};${rtkConfigurationPath}")
  endif()

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

  CUDA_COMPILE (rtkcuda_CUDA_WRAPPERS ${rtkcuda_CUDA_FILES})

  # Rollback to all includes
  if(WIN32)
    set_directory_properties(PROPERTIES INCLUDE_DIRECTORIES "${dirInc}")
  endif()

  # Rollback to all compile definitions
  set_directory_properties(PROPERTIES COMPILE_DEFINITIONS "${defCom}")

  add_library(rtkcuda
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
  set_target_properties (rtkcuda PROPERTIES LINKER_LANGUAGE CXX)
  target_link_libraries(rtkcuda ${CUDA_LIBRARIES} ${CUDA_cufft_LIBRARY} ${CUDA_cublas_LIBRARY} RTK ITKCudaCommon)
endif ()
#=========================================================

#=========================================================
# Installation code
if(NOT RTK_INSTALL_NO_DEVELOPMENT)
  # Generate RTKConfig.cmake for the install tree.
  set (RTK_USE_FILE "${CMAKE_INSTALL_PREFIX}/${RTK_INSTALL_PACKAGE_DIR}/UseRTK.cmake")
  set (RTK_INCLUDE_DIRS "${CMAKE_INSTALL_PREFIX}/${RTK_INSTALL_INCLUDE_DIR};${CMAKE_INSTALL_PREFIX}/${RTK_INSTALL_INCLUDE_DIR}/lpsolve")
  set (RTK_LIBRARY_DIRS "${CMAKE_INSTALL_PREFIX}/${RTK_INSTALL_LIB_DIR}")
  configure_file (${PROJECT_SOURCE_DIR}/cmake/RTKConfig.cmake.in ${RTK_BINARY_DIR}/CMakeFiles/RTKConfig.cmake @ONLY)
  configure_file (${PROJECT_SOURCE_DIR}/cmake/RTKConfigVersion.cmake.in ${RTK_BINARY_DIR}/CMakeFiles/RTKConfigVersion.cmake @ONLY)
  install(FILES ${RTK_BINARY_DIR}/CMakeFiles/RTKConfig.cmake
                ${RTK_BINARY_DIR}/CMakeFiles/RTKConfigVersion.cmake
                ${RTK_SOURCE_DIR}/cmake/UseRTK.cmake
          DESTINATION ${RTK_INSTALL_PACKAGE_DIR})

  # Include .h and .hxx files
  file(GLOB __files1 "${PROJECT_SOURCE_DIR}/code/*.h")
  file(GLOB __files2 "${PROJECT_SOURCE_DIR}/code/*.hxx")
  install(FILES ${__files1} ${__files2} ${RTK_BINARY_DIR}/rtkConfiguration.h
    DESTINATION ${RTK_INSTALL_INCLUDE_DIR}
    COMPONENT Development)
  if(RTK_USE_CUDA)
    file(GLOB __files3 "${PROJECT_SOURCE_DIR}/code/*.hcu")
    install(FILES ${__files3}
      DESTINATION ${RTK_INSTALL_INCLUDE_DIR}
      COMPONENT Development)
  endif()
endif()

if(NOT RTK_INSTALL_NO_LIBRARIES)
  install(TARGETS RTK
    RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
    LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
    ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)

  if(RTK_USE_CUDA)
    install(TARGETS rtkcuda
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  endif()
endif()
#=========================================================

set(RTK_LIBRARIES ${RTK_LIBRARIES} PARENT_SCOPE)
