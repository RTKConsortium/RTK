# client maintainer: simon.rit@creatis.insa-lyon.fr
set(CTEST_SITE "russula.clb")
set(CTEST_BUILD_NAME "Linux-64bit-gcc472-cuda41-itk4")
set(CTEST_BUILD_CONFIGURATION Release)
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(dashboard_binary_name "RTK_lin64_gcc472_cuda41_itk4")
set(ENV{ITK_DIR} "/home/srit/src/itk/lin64_gcc_472")
set(ENV{CUDA_BIN_PATH} "/home/srit/Download/cuda41/bin")
set(CTEST_BUILD_FLAGS -j12)

set(ENV{CC} "/home/srit/src/gcc/gcc472-install/bin/gcc")
set(ENV{CXX} "/home/srit/src/gcc/gcc472-install/bin/c++")
set(ENV{PATH} "/home/srit/src/gcc/gcc441-install/bin:$ENV{PATH}")
set(ENV{CXXFLAGS} "")

include("${CTEST_SCRIPT_DIRECTORY}/rtk_common.cmake")
