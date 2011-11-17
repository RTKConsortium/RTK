# client maintainer: simon.rit@creatis.insa-lyon.fr
set(CTEST_SITE "shiitake.clb")
set(CTEST_BUILD_NAME "Linux-64bit-gcc-cuda40")
set(CTEST_BUILD_CONFIGURATION Release)
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(dashboard_binary_name "RTK_lin64_gcc_cuda40")
set(ENV{ITK_DIR} "/home/srit/src/itk/lin64")
set(ENV{CUDA_BIN_PATH} "/home/srit/Download/cuda40/bin")

include("${CTEST_SCRIPT_DIRECTORY}/rtk_common.cmake")

