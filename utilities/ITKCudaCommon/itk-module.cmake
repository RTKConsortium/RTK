# if (ITK_USE_CUDA)

set(DOCUMENTATION "")

itk_module(ITKCudaCommon
  EXCLUDE_FROM_DEFAULT
  COMPILE_DEPENDS
    ITKCommon
  TEST_DEPENDS
    ITKTestKernel
  DESCRIPTION
    "${DOCUMENTATION}"
  )

# endif()
  