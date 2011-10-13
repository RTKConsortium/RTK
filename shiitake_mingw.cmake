# client maintainer: simon.rit@creatis.insa-lyon.fr
set(CTEST_SITE "shiitake.clb")
set(CTEST_BUILD_NAME "Windows7-64bit-MinGW")
set(CTEST_BUILD_CONFIGURATION Release)
set(CTEST_CMAKE_GENERATOR "MinGW Makefiles")
set(CTEST_MAKE_PROGRAM "C:/MinGW/bin/mingw32-make.exe")
set(CTEST_GIT_COMMAND "C:/cygwin/bin/git")
set(dashboard_binary_name "RTK_win64_mingw")
set( ENV{ITK_DIR} "Z:/src/itk/win64_mingw")

include("${CTEST_SCRIPT_DIRECTORY}/rtk_common.cmake")

