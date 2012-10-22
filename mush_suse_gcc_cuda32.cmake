# client maintainer: simon.rit@creatis.insa-lyon.fr
set(CTEST_SITE "mush.clb")
set(CTEST_BUILD_NAME "Linux-64bit-gcc-cuda32")
set(CTEST_BUILD_CONFIGURATION Release)
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(dashboard_binary_name "RTK_lin64_gcc_cuda32")
set(ENV{ITK_DIR} "/home/srit/src/itk/lin64")
set(ENV{CUDA_BIN_PATH} "/home/srit/Download/cuda32/bin")

include("${CTEST_SCRIPT_DIRECTORY}/rtk_common.cmake")

