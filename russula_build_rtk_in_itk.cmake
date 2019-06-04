set(CTEST_SITE "russula.clb")
set(CTEST_BUILD_FLAGS -j16)
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(CTEST_GIT_COMMAND "git")
foreach(ITK_VERSION "master" "release")
  foreach(FFTW ON)
    foreach(DEBUG_RELEASE RelWithDebInfo)
      foreach(STATIC_SHARED Static Shared)
        set(BUILD_SHARED_LIBS OFF)
        if(${STATIC_SHARED} STREQUAL Shared)
          set(BUILD_SHARED_LIBS ON)
        endif()
        set(CTEST_BUILD_NAME "InITK-Linux-ITK${ITK_VERSION}-${STATIC_SHARED}-${DEBUG_RELEASE}-FFTW${FFTW}")
        set(CTEST_SOURCE_DIRECTORY "/home/srit/src/rtk/dashboard_tests/ITK-${ITK_VERSION}")
        file(REMOVE_RECURSE "${CTEST_SOURCE_DIRECTORY}/Modules/Remote/RTK")
        set(CTEST_BINARY_DIRECTORY "/home/srit/src/rtk/dashboard_tests/${CTEST_BUILD_NAME}")
        set(CTEST_BUILD_CONFIGURATION ${DEBUG_RELEASE})
        set(CTEST_CONFIGURATION_TYPE ${DEBUG_RELEASE})

        ctest_empty_binary_directory(${CTEST_BINARY_DIRECTORY})
        ctest_start(Nightly)
        if("${ITK_VERSION}" STREQUAL "master")
            ctest_update()
            file(READ "${CTEST_SOURCE_DIRECTORY}/Modules/Remote/RTK.remote.cmake" REMOTE_RTK_FILE)
            string(REGEX REPLACE "GIT_TAG.*$" "GIT_TAG master" REMOTE_RTK_FILE "${REMOTE_RTK_FILE}")
            file(WRITE "${CTEST_SOURCE_DIRECTORY}/Modules/Remote/RTK.remote.cmake" "${REMOTE_RTK_FILE}")
        endif()

        set(cfg_options
            -DITK_LEGACY_REMOVE=ON
            -DITK_BUILD_DEFAULT_MODULES=OFF
            -DModule_RTK=ON
            -DBUILD_EXAMPLES=OFF
            -DBUILD_TESTING=ON
            -DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED_LIBS} 
          )
        if("${FFTW}" STREQUAL ON)
          set(cfg_options ${cfg_options}
              -DITK_USE_FFTWD=${FFTW}
              -DITK_USE_FFTWF=${FFTW}
              -DITK_USE_SYSTEM_FFTW=${FFTW}
          )
        endif()
        ctest_configure(OPTIONS "${cfg_options}")
        ctest_build()
        ctest_test()

        # Use RTK parameters for submission
        set(CTEST_PROJECT_NAME "RTK")
        set(CTEST_NIGHTLY_START_TIME "1:00:00 UTC")

        set(CTEST_DROP_METHOD "http")
        set(CTEST_DROP_SITE "my.cdash.org")
        set(CTEST_DROP_LOCATION "/submit.php?project=RTK")
        set(CTEST_DROP_SITE_CDASH TRUE)
        ctest_submit()
      endforeach()
    endforeach()
  endforeach()
endforeach()
