# client maintainer: simon.rit@creatis.insa-lyon.fr
set(CTEST_SITE "russula.clb")
set(CTEST_BUILD_NAME "Linux-64bit-gcc-shared")
set(CTEST_BUILD_CONFIGURATION Release)
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(dashboard_binary_name "RTK_lin64_gcc_shared")
set(ENV{ITK_DIR} "/home/srit/src/itk/lin64-SharedLibs-Release")
set(CTEST_BUILD_FLAGS -j16)
include("${CTEST_SCRIPT_DIRECTORY}/rtk_common.cmake")

