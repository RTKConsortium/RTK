set(CTEST_SITE "shiitake.clb")
set(CTEST_BUILD_NAME "Windows7-64bit-MSVC13-ITK4.12.2-Static-Release-NoFFTW")
set(CTEST_UPDATE_COMMAND "C:\\Program Files\\Git\\bin\\git.exe")
set(CTEST_SOURCE_DIRECTORY "D:\\src\\rtk\\RTK")
set(CTEST_BINARY_DIRECTORY "D:\\src\\rtk\\RTK-ITK4.12.2-Static-Release-NoFFTW")
set(CTEST_NOTES_FILES "${CTEST_SCRIPT_DIRECTORY}/${CTEST_SCRIPT_NAME}")
set(CTEST_CMAKE_GENERATOR "Visual Studio 12 2013 Win64")
set(CTEST_TEST_TIMEOUT "1000")
set(CTEST_BUILD_CONFIGURATION Release)
set(CTEST_CONFIGURATION_TYPE Release)
ctest_empty_binary_directory(${CTEST_BINARY_DIRECTORY})

file(WRITE ${CTEST_BINARY_DIRECTORY}/CTestCustom.cmake
  "set(CTEST_CUSTOM_WARNING_EXCEPTION ${CTEST_CUSTOM_WARNING_EXCEPTION}
  \"WARNING non-zero return value in ctest from:\")")

set(ENV{PATH} "D:/src/rtk/RTK-4.12.2-Static-Release-NoFFTW/bin/Release;$ENV{PATH}")

ctest_start(Nightly)
ctest_update()

set(cfg_options
   -DITK_DIR:PATH=D:/src/itk/build-static-nofftw
   -DRTK_USE_CUDA:BOOL=ON
   -DExternalData_OBJECT_STORES:PATH=D:/src/rtk/data
   -DBUILD_SHARED_LIBS:BOOL=OFF
   -DCUDA_TOOLKIT_ROOT_DIR:PATH=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.0
  )

ctest_configure(OPTIONS "${cfg_options}")
CTEST_READ_CUSTOM_FILES("${CTEST_BINARY_DIRECTORY}")
ctest_build()
ctest_test()
ctest_submit()
