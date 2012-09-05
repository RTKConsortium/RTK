# client maintainer: simon.rit@creatis.insa-lyon.fr
set(CTEST_SITE "shiitake.clb")
set(CTEST_BUILD_NAME "Linux-64bit-gcc-debug")
set(CTEST_BUILD_CONFIGURATION Debug)
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(dashboard_binary_name "RTK_lin64_gcc_debug")
set(ENV{ITK_DIR} "/home/srit/src/itk/lin64")

include("${CTEST_SCRIPT_DIRECTORY}/rtk_common.cmake")

