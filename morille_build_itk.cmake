set(CTEST_SITE "morille.clb")
set(CTEST_GIT_COMMAND "C:\\Program Files\\Git\\bin\\git.exe")
set(CTEST_GIT_UPDATE_CUSTOM "${CTEST_GIT_COMMAND}" pull)
set(MSVC_VERSION 16)
set(MSVC_YEAR 2019)
set(CTEST_CMAKE_GENERATOR "Visual Studio ${MSVC_VERSION} ${MSVC_YEAR}")
foreach(ITK_VERSION master release)
  set(CTEST_SOURCE_DIRECTORY "C:\\src\\itk\\ITK-${ITK_VERSION}")
  foreach(FFTW ON OFF)
    foreach(DEBUG_RELEASE Release Debug)
      foreach(STATIC_SHARED Static Shared)
        set(BUILD_SHARED_LIBS OFF)
        if(${STATIC_SHARED} STREQUAL Shared)
          set(BUILD_SHARED_LIBS ON)
        endif()

        set(CTEST_BINARY_DIRECTORY "C:\\src\\itk\\${MSVC_YEAR}-${ITK_VERSION}-${STATIC_SHARED}-${DEBUG_RELEASE}-FFTW${FFTW}")
        set(CTEST_BUILD_CONFIGURATION ${DEBUG_RELEASE})
        set(CTEST_CONFIGURATION_TYPE ${DEBUG_RELEASE})

        set(ENV{PATH} "C:/src/fftw-3.3.9/build/${DEBUG_RELEASE};$ENV{PATH}")

        ctest_start(Continuous)
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
           -DFFTWD_BASE_LIB:PATH=C:/src/fftw/build/${DEBUG_RELEASE}/fftw3.lib
           -DFFTWD_LIBRARIES:PATH=C:/src/fftw/build/${DEBUG_RELEASE}/fftw3.lib
           -DFFTWD_THREADS_LIB:PATH=C:/src/fftw/build/${DEBUG_RELEASE}/fftw3.lib
           -DFFTWF_BASE_LIB:PATH=C:/src/fftw/build/${DEBUG_RELEASE}/fftw3f.lib
           -DFFTWF_LIBRARIES:PATH=C:/src/fftw/build/${DEBUG_RELEASE}/fftw3f.lib
           -DFFTWF_THREADS_LIB:PATH=C:/src/fftw/build/${DEBUG_RELEASE}/fftw3f.lib
           -DFFTW_INCLUDE_PATH:PATH=C:/src/fftw/fftw-3.3.9/api
          )
        ctest_configure(OPTIONS "${cfg_options}")
        ctest_build()
      endforeach()
    endforeach()
  endforeach()
endforeach()
