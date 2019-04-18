# client maintainer: simon.rit@creatis.insa-lyon.fr
set(CTEST_SITE "russula.clb")
set(CTEST_BUILD_NAME "Linux-64bit-gcc-Cuda10.0")
set(CTEST_BUILD_CONFIGURATION Release)
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(dashboard_binary_name "RTK_lin64_gcc_cuda100")
set(ENV{ITK_DIR} "/home/srit/src/itk/lin64")
set(CTEST_BUILD_FLAGS -j16)
set(ENV{CUDA_BIN_PATH} "/home/srit/Downloads/cuda100/bin")
include("${CTEST_SCRIPT_DIRECTORY}/rtk_common.cmake")

