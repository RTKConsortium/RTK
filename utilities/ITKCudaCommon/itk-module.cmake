if (ITK_USE_CUDA)

  get_filename_component(MY_CURENT_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
  file(READ "${MY_CURENT_DIR}/README" DOCUMENTATION)

  option(Module_ITKCudaCommon "Compile the ITK Cuda external module" FALSE)

  if (${Module_ITKCudaCommon})

    # define the dependencies of the include module and the tests
    itk_module(ITKCudaCommon
      DEPENDS
        ITKCommon
      TEST_DEPENDS
        ITKTestKernel
      DESCRIPTION
        "${DOCUMENTATION}"
    )

  endif()

endif()
  