# client maintainer: simon.rit@creatis.insa-lyon.fr
set(CTEST_SITE "linux15.dg.creatis.insa-lyon.fr")
set(CTEST_BUILD_NAME "Linux-64bit-gcc-cuda-system-itk4")
set(CTEST_BUILD_CONFIGURATION Release)
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(CTEST_DASHBOARD_ROOT "/tmp/RTK_dashboard")
set(dashboard_binary_name "RTK_lin64_gcc_cuda_system_itk4")
set(ENV{ITK_DIR} "/home/srit/src/itk4/lin64-dg")
set(ENV{CUDA_BIN_PATH} "/usr/lib64/cuda/bin")
set(CTEST_BUILD_FLAGS -j8)

# OpenCL
set(ENV{LD_LIBRARY_PATH} "/usr/lib64/nvidia:$ENV{LD_LIBRARY_PATH}")
set(CONFIGURE_OPTIONS "-DOPENCL_ROOT_DIR=/usr/lib64/cuda")

include("${CTEST_SCRIPT_DIRECTORY}/rtk_common.cmake")

