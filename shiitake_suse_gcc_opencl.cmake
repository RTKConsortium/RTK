# client maintainer: simon.rit@creatis.insa-lyon.fr
set(CTEST_SITE "shiitake.clb")
set(CTEST_BUILD_NAME "Linux-64bit-gcc-opencl")
set(CTEST_BUILD_CONFIGURATION Release)
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(dashboard_binary_name "RTK_lin64_gcc_opencl")
set(ENV{ITK_DIR} "/home/srit/src/itk/lin64")
set(ENV{LD_LIBRARY_PATH} "/home/srit/Download/AMD-APP-SDK-v2.5-lnx64/lib/x86_64:$ENV{LD_LIBRARY_PATH}")

include("${CTEST_SCRIPT_DIRECTORY}/rtk_common.cmake")

