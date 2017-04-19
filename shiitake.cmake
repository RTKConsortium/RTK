set(CTEST_SITE "shiitake.clb")
set(CTEST_BUILD_NAME "Windows7-64bit-MSVC13-ITK4.10.0-Shared")
set(CTEST_UPDATE_COMMAND "C:\\Program Files\\Git\\bin\\git.exe")
set(CTEST_SOURCE_DIRECTORY "D:\\src\\rtk\\RTK")
set(CTEST_BINARY_DIRECTORY "D:\\src\\rtk\\RTK-ITK4.10.0-Shared")
set(CTEST_NOTES_FILES "${CTEST_SCRIPT_DIRECTORY}/${CTEST_SCRIPT_NAME}")
set(CTEST_CMAKE_GENERATOR "Visual Studio 12 2013 Win64")
set(CTEST_TEST_TIMEOUT "900")
set(CTEST_BUILD_CONFIGURATION Release)
set(CTEST_CONFIGURATION_TYPE Release)
ctest_empty_binary_directory(${CTEST_BINARY_DIRECTORY})

file(WRITE ${CTEST_BINARY_DIRECTORY}/CTestCustom.cmake
  "set(CTEST_CUSTOM_WARNING_EXCEPTION ${CTEST_CUSTOM_WARNING_EXCEPTION}
  \"WARNING non-zero return value in ctest from:\")")

set(ENV{PATH} "D:/src/rtk/RTK-4.10.0-Shared/bin/Release;$ENV{PATH}")
   
ctest_start(Nightly)
#ctest_start(Experimental)
ctest_update()

set(cfg_options
   -DITK_DIR:PATH=D:/src/itk/build410_0
   -DRTK_USE_CUDA:BOOL=ON
   -DBUILD_SHARED_LIBS:BOOL=ON
  )

ctest_configure(OPTIONS "${cfg_options}")
CTEST_READ_CUSTOM_FILES("${CTEST_BINARY_DIRECTORY}")
ctest_build()
ctest_test()
ctest_submit()

