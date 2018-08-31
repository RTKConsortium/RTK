set(MSVC_VERSION 14)
set(MSVC_YEAR 2015)
foreach(ITK_VERSION "v4.13.0" "v5.0a02")
  foreach(FFTW ON OFF)
    foreach(DEBUG_RELEASE Release Debug)
      foreach(STATIC_SHARED Static Shared)
        set(BUILD_SHARED_LIBS OFF)
        if(${STATIC_SHARED} STREQUAL Shared)
          set(BUILD_SHARED_LIBS ON)
        endif()

        set(CTEST_SITE "shiitake.clb")
        set(CTEST_GIT_COMMAND "C:\\Program Files\\Git\\bin\\git.exe")
        set(CTEST_GIT_UPDATE_CUSTOM "${CTEST_GIT_COMMAND} checkout ${ITK_VERSION}")
        set(CTEST_SOURCE_DIRECTORY "D:\\src\\itk\\ITK-${ITK_VERSION}")
        set(CTEST_BINARY_DIRECTORY "D:\\src\\itk\\${MSVC_YEAR}-${ITK_VERSION}-${STATIC_SHARED}-${DEBUG_RELEASE}-FFTW${FFTW}")
        set(CTEST_CMAKE_GENERATOR "Visual Studio ${MSVC_VERSION} ${MSVC_YEAR} Win64")
        set(CTEST_BUILD_CONFIGURATION ${DEBUG_RELEASE})
        set(CTEST_CONFIGURATION_TYPE ${DEBUG_RELEASE})

        set(ENV{PATH} "D:/src/fftw-3.3.7/build/${DEBUG_RELEASE};$ENV{PATH}")
        set(ENV{PATH} "D:/src/kwstyle;$ENV{PATH}")

        ctest_start(Nightly)
        ctest_update()

        set(cfg_options
           -DITK_FUTURE_LEGACY=ON
           -DITK_LEGACY_REMOVE=ON
           -DBUILD_EXAMPLES=OFF
           -DBUILD_TESTING=OFF
           -DITK_USE_FFTWD=${FFTW}
           -DITK_USE_FFTWF=${FFTW}
           -DITK_USE_SYSTEM_FFTW=${FFTW}
           -DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED_LIBS}
           -DFFTWD_LIB:PATH=D:/src/fftw-3.3.7/build/${DEBUG_RELEASE}/fftw3.lib
           -DFFTWD_THREADS_LIB:PATH=D:/src/fftw-3.3.7/build/${DEBUG_RELEASE}/fftw3.lib
           -DFFTWF_LIB:PATH=D:/src/fftw-3.3.7/build/${DEBUG_RELEASE}/fftw3f.lib
           -DFFTWF_THREADS_LIB:PATH=D:/src/fftw-3.3.7/build/${DEBUG_RELEASE}/fftw3f.lib
           -DFFTW_INCLUDE_PATH:PATH=D:/src/fftw-3.3.7/api
          )
        ctest_configure(OPTIONS "${cfg_options}")
        ctest_build()
      endforeach()
    endforeach()
  endforeach()
endforeach()
