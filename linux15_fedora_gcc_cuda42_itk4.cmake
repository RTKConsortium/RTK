# client maintainer: simon.rit@creatis.insa-lyon.fr
set(CTEST_SITE "linux15.dg.creatis.insa-lyon.fr")
set(CTEST_BUILD_NAME "Linux-64bit-gcc-cuda42-itk4")
set(CTEST_BUILD_CONFIGURATION Release)
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(CTEST_DASHBOARD_ROOT "/tmp/RTK_dashboard")
set(dashboard_binary_name "RTK_lin64_gcc_cuda42_itk4")
set(ENV{ITK_DIR} "/home/srit/src/itk/lin64-dg")
set(ENV{CUDA_BIN_PATH} "/home/srit/Download/cuda42/bin")
set(ENV{CUDA_LIB_PATH} "/usr/lib64/nvidia-304xx")
set(CTEST_BUILD_FLAGS -j8)

include("${CTEST_SCRIPT_DIRECTORY}/rtk_common.cmake")

