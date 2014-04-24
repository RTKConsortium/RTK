# client maintainer: simon.rit@creatis.insa-lyon.fr
set(CTEST_SITE "shiitake.clb")
set(CTEST_BUILD_NAME "Linux-64bit-gcc-opencl")
set(CTEST_BUILD_CONFIGURATION Release)
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(dashboard_binary_name "RTK_lin64_gcc_opencl")
set(ENV{ITK_DIR} "/home/srit/src/itk4/lin64")
set(CONFIGURE_OPTIONS "-DOPENCL_ROOT_DIR=/home/srit/Download/AMD-APP-SDK-v2.5-lnx64")

include("${CTEST_SCRIPT_DIRECTORY}/rtk_common.cmake")

