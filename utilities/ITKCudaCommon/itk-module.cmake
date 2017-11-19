if(ITK_SOURCE_DIR)
  get_filename_component(_CURRENT_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
  file(READ "${_CURRENT_DIR}/README" DOCUMENTATION)

  itk_module(ITKCudaCommon
    ENABLE_SHARED
    EXCLUDE_FROM_DEFAULT
    DEPENDS
      ITKCommon
    TEST_DEPENDS
      ITKTestKernel
    DESCRIPTION
      "${DOCUMENTATION}"
    )
endif()
