set(ITK_VERSION "v5.0a01")
foreach(DEBUG_RELEASE Release Debug)
  foreach(FFTW ON OFF)
    foreach(STATIC_SHARED Static Shared)
      if(${STATIC_SHARED} EQUAL "Shared")
        set(BUILD_SHARED_LIBS ON)
      endif()

      set(CTEST_SITE "shiitake.clb")
      set(CTEST_BUILD_NAME "Windows7-64bit-MSVC13-ITK${ITK_VERSION}-${STATIC_SHARED}-${DEBUG_RELEASE}-FFTW${FFTW}")
      set(CTEST_UPDATE_COMMAND "C:\\Program Files\\Git\\bin\\git.exe")
      set(CTEST_SOURCE_DIRECTORY "D:\\src\\rtk\\RTK")
      set(CTEST_BINARY_DIRECTORY "D:\\src\\rtk\\RTK-ITK${ITK_VERSION}-${STATIC_SHARED}-${DEBUG_RELEASE}-FFTW${FFTW}")
      set(CTEST_NOTES_FILES "${CTEST_SCRIPT_DIRECTORY}/${CTEST_SCRIPT_NAME}")
      set(CTEST_CMAKE_GENERATOR "Visual Studio 12 2013 Win64")
      set(CTEST_TEST_TIMEOUT "1000")
      set(CTEST_BUILD_CONFIGURATION ${DEBUG_RELEASE})
      set(CTEST_CONFIGURATION_TYPE ${DEBUG_RELEASE})
      ctest_empty_binary_directory(${CTEST_BINARY_DIRECTORY})

      #file(WRITE ${CTEST_BINARY_DIRECTORY}/CTestCustom.cmake
      #  "set(CTEST_CUSTOM_WARNING_EXCEPTION ${CTEST_CUSTOM_WARNING_EXCEPTION}
      #  \"WARNING non-zero return value in ctest from:\")")

      set(ENV{PATH} "D:/src/itk/ITK-${ITK_VERSION}-${STATIC_SHARED}-${DEBUG_RELEASE}-FFTW${FFTW}/bin/${DEBUG_RELEASE};$ENV{PATH}")
      set(ENV{PATH} "D:/src/itk/RTK-ITK${ITK_VERSION}-${STATIC_SHARED}-${DEBUG_RELEASE}-FFTW${FFTW}/bin/${DEBUG_RELEASE};$ENV{PATH}")
      set(ENV{PATH} "D:/src/fftw-3.3.7/build/${DEBUG_RELEASE};$ENV{PATH}")

      ctest_start(Nightly)
      ctest_update()

      set(cfg_options
         -DITK_DIR:PATH=D:\\src\\itk\\ITK-${ITK_VERSION}-${STATIC_SHARED}-${DEBUG_RELEASE}-FFTW${FFTW}
         -DRTK_USE_CUDA:BOOL=ON
         -DExternalData_OBJECT_STORES:PATH=D:/src/rtk/data
         -DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED_LIBS}
         -DCUDA_TOOLKIT_ROOT_DIR:PATH=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.0
        )

      ctest_configure(OPTIONS "${cfg_options}")
      ctest_read_custom_files("${CTEST_BINARY_DIRECTORY}")
      ctest_build()
      ctest_test()
      ctest_submit()
    endforeach()
  endforeach()
endforeach()
