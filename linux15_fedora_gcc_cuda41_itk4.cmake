# client maintainer: simon.rit@creatis.insa-lyon.fr
set(CTEST_SITE "linux15.dg.creatis.insa-lyon.fr")
set(CTEST_BUILD_NAME "Linux-64bit-gcc-cuda41_itk4")
set(CTEST_BUILD_CONFIGURATION Release)
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(CTEST_DASHBOARD_ROOT "/tmp/RTK_dashboard")
set(dashboard_binary_name "RTK_lin64_gcc_cuda41_itk4")
set(ENV{ITK_DIR} "/home/srit/src/itk4/lin64-dg")
set(ENV{CUDA_BIN_PATH} "/home/srit/Download/cuda41/bin")
set(ENV{PATH} "$ENV{PATH}:/home/srit/src/gcc/gcc436-install/bin")
set(CTEST_BUILD_FLAGS -j8)

include("${CTEST_SCRIPT_DIRECTORY}/rtk_common.cmake")

