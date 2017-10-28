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
  ${RTK_IO_DEPENDS}
  )

# -----------------------------------------
#  Required Modules to build RTK tests :
set(RTK_TEST_DEPENDS
  ITKTestKernel)

if(NOT ${ITK_VERSION} VERSION_LESS "4.6.0")
  list(APPEND RTK_TEST_DEPENDS ITKImageNoise) #required by rtkRegularizedConjugateGradientTest
endif()

# -----------------------------------------
# CUDA optional dependencies
if(${RTK_USE_CUDA})
  list(APPEND RTK_DEPENDS ITKCudaCommon)
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
