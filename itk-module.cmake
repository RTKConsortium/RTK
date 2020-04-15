set(DOCUMENTATION "")

# -----------------------------------------
#  Required Modules to build RTK library :
set(RTK_IO_DEPENDS
  ITKIOCSV
  ITKIOGDCM
  ITKGDCM
  ITKIOMeta
  ITKIORAW
  ITKIOTIFF
  ITKIOXML
  )

if(ITK_WRAP_PYTHON)
  set(RTK_BRIDGE_DEPENDS
  ITKBridgeNumPy
  )
endif()

set(RTK_DEPENDS
  ITKCommon
  ITKConvolution
  ITKFFT
  ITKOptimizers
  ITKRegistrationCommon
  ITKSmoothing
  ITKImageNoise
  ${RTK_IO_DEPENDS}
  ${RTK_BRIDGE_DEPENDS}
  )

# -----------------------------------------
#  Required Modules to build RTK tests :
set(RTK_TEST_DEPENDS
  ITKTestKernel)

# # -----------------------------------------
# # CUDA optional dependencies
if(ITK_SOURCE_DIR)
  if(${RTK_USE_CUDA})
    list(APPEND RTK_DEPENDS ITKCudaCommon)
  endif()
endif()

#=========================================================
# Module RTK
#=========================================================
itk_module(RTK
  ENABLE_SHARED
  EXCLUDE_FROM_DEFAULT
  DEPENDS
    ${RTK_DEPENDS}
  TEST_DEPENDS
    ${RTK_TEST_DEPENDS}
  FACTORY_NAMES
    ImageIO::DCMImagX
    ImageIO::His
    ImageIO::Hnc
    ImageIO::Hnd
    ImageIO::ImagX
    ImageIO::Ora
    ImageIO::Xim
    ImageIO::XRad
  DESCRIPTION
    "${DOCUMENTATION}"
  )
