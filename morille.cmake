set(CTEST_SITE "morille.clb")
set(CTEST_UPDATE_COMMAND "C:\\Program Files\\Git\\bin\\git.exe")
set(CTEST_SOURCE_DIRECTORY "C:\\src\\rtk\\RTK")
set(MSVC_VERSION 16)
set(MSVC_YEAR 2019)
set(CTEST_CMAKE_GENERATOR "Visual Studio ${MSVC_VERSION} ${MSVC_YEAR}")
foreach(ITK_VERSION "master")
  foreach(DEBUG_RELEASE "Release" "Debug")
    set(ENV{PATH} "C:/src/fftw/build/${DEBUG_RELEASE};$ENV{PATH}")
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

        set(CTEST_BUILD_NAME "Windows7-64bit-MSVC${MSVC-VERSION}-ITK${ITK_VERSION}-${STATIC_SHARED}-${DEBUG_RELEASE}-FFTW${FFTW}")
        set(CTEST_BINARY_DIRECTORY "C:\\src\\rtk\\${MSVC_YEAR}-${ITK_VERSION}-${STATIC_SHARED}-${DEBUG_RELEASE}-FFTW${FFTW}")
        set(CTEST_NOTES_FILES "${CTEST_SCRIPT_DIRECTORY}/${CTEST_SCRIPT_NAME}")
        set(CTEST_TEST_TIMEOUT "600")
        set(CTEST_BUILD_CONFIGURATION ${DEBUG_RELEASE})
        set(CTEST_CONFIGURATION_TYPE ${DEBUG_RELEASE})

        set(ENV{PATH} "C:/src/itk/${MSVC_YEAR}-${ITK_VERSION}-${STATIC_SHARED}-${DEBUG_RELEASE}-FFTW${FFTW}/bin/${DEBUG_RELEASE};$ENV{PATH}")
        set(ENV{PATH} "C:/src/itk/${MSVC_YEAR}-${ITK_VERSION}-${STATIC_SHARED}-${DEBUG_RELEASE}-FFTW${FFTW}/lib/${DEBUG_RELEASE};$ENV{PATH}")


        set(cfg_options
           -DITK_DIR:PATH=C:\\src\\itk\\${MSVC_YEAR}-${ITK_VERSION}-${STATIC_SHARED}-${DEBUG_RELEASE}-FFTW${FFTW}
           -DRTK_USE_CUDA:BOOL=ON
           -DITK_USE_KWSTYLE:BOOL=OFF
           -DFAST_TESTS_NO_CHECKS=${FASTTEST}
           -DExternalData_OBJECT_STORES:PATH=C:/src/rtk/data
           -DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED_LIBS}
           -DCMAKE_CUDA_ARCHITECTURES:STRING=52
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
