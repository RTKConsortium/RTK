set(DOCUMENTATION "")

# -----------------------------------------
#  Required Modules to build RTK library :
set(RTK_IO_DEPENDS
  ITKIOCSV
  ITKIOGDCM
  ITKIOMeta
  ITKIORAW
  ITKIOTIFF
  ITKIOXML
  )

set(RTK_DEPENDS
  ITKCommon
  ITKConvolution
  ITKFFT
  ITKOptimizers
  ITKRegistrationCommon
  ITKSmoothing
  ITKImageNoise
  ${RTK_IO_DEPENDS}
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
  DESCRIPTION
    "${DOCUMENTATION}"
  )
