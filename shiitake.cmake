set(MSVC_VERSION 14)
set(MSVC_YEAR 2015)
foreach(ITK_VERSION "v5.0a02" "v4.13.0")
  foreach(DEBUG_RELEASE "Release" "Debug")
    foreach(FFTW OFF ON)
      foreach(STATIC_SHARED "Shared" "Static")
        set(BUILD_SHARED_LIBS OFF)
        if("${STATIC_SHARED}" STREQUAL "Shared")
          set(BUILD_SHARED_LIBS ON)
        endif()
        set(FASTTEST FALSE)
        if("${DEBUG_RELEASE}" STREQUAL "Debug")
          set(FASTTEST TRUE)
        endif()
        if("${ITK_VERSION}" STREQUAL "v4.13.0" AND "${STATIC_SHARED}" STREQUAL "Shared" AND "${FFTW}" STREQUAL ON)
          continue()
        endif()
        if("${ITK_VERSION}" STREQUAL "v4.13.0")
          set(CTEST_CUSTOM_TESTS_IGNORE RTKInDoxygenGroup)
        endif()

        set(CTEST_SITE "shiitake.clb")
        set(CTEST_BUILD_NAME "Windows7-64bit-MSVC${MSVC-VERSION}-ITK${ITK_VERSION}-${STATIC_SHARED}-${DEBUG_RELEASE}-FFTW${FFTW}")
        set(CTEST_UPDATE_COMMAND "C:\\Program Files\\Git\\bin\\git.exe")
        set(CTEST_SOURCE_DIRECTORY "D:\\src\\rtk\\RTK")
        set(CTEST_BINARY_DIRECTORY "D:\\src\\rtk\\${MSVC_YEAR}-${ITK_VERSION}-${STATIC_SHARED}-${DEBUG_RELEASE}-FFTW${FFTW}")
        set(CTEST_NOTES_FILES "${CTEST_SCRIPT_DIRECTORY}/${CTEST_SCRIPT_NAME}")
        set(CTEST_CMAKE_GENERATOR "Visual Studio ${MSVC_VERSION} ${MSVC_YEAR} Win64")
        set(CTEST_TEST_TIMEOUT "600")
        set(CTEST_BUILD_CONFIGURATION ${DEBUG_RELEASE})
        set(CTEST_CONFIGURATION_TYPE ${DEBUG_RELEASE})

        set(ENV{PATH} "D:/src/kwstyle;$ENV{PATH}")
        set(ENV{PATH} "D:/src/itk/${MSVC_YEAR}-${ITK_VERSION}-${STATIC_SHARED}-${DEBUG_RELEASE}-FFTW${FFTW}/bin/${DEBUG_RELEASE};$ENV{PATH}")
        set(ENV{PATH} "D:/src/itk/${MSVC_YEAR}-${ITK_VERSION}-${STATIC_SHARED}-${DEBUG_RELEASE}-FFTW${FFTW}/lib/${DEBUG_RELEASE};$ENV{PATH}")
        set(ENV{PATH} "D:/src/fftw-3.3.7/build/${DEBUG_RELEASE};$ENV{PATH}")


        set(cfg_options
           -DITK_DIR:PATH=D:\\src\\itk\\${MSVC_YEAR}-${ITK_VERSION}-${STATIC_SHARED}-${DEBUG_RELEASE}-FFTW${FFTW}
           -DRTK_USE_CUDA:BOOL=ON
           -DITK_USE_KWSTYLE:BOOL=OFF
           -DFAST_TESTS_NO_CHECKS=${FASTTEST}
           -DExternalData_OBJECT_STORES:PATH=D:/src/rtk/data
           -DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED_LIBS}
           -DCUDA_TOOLKIT_ROOT_DIR:PATH=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.0
           -DKWSTYLE_EXECUTABLE:FILEPATH=D:\\src\\kwstyle\\KWStyle.exe
          )

        ctest_empty_binary_directory(${CTEST_BINARY_DIRECTORY})
        ctest_start(Nightly)
        ctest_update()
        ctest_configure(OPTIONS "${cfg_options}")
        ctest_read_custom_files("${CTEST_BINARY_DIRECTORY}")
        ctest_build()
        ctest_test()
        ctest_submit()
      endforeach()
    endforeach()
  endforeach()
endforeach()
