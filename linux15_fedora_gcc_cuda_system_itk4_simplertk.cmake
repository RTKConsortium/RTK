# client maintainer: simon.rit@creatis.insa-lyon.fr
set(CTEST_SITE "linux15.dg.creatis.insa-lyon.fr")
set(CTEST_BUILD_NAME "Linux-64bit-gcc-cuda-system-itk4-simplertk")
set(CTEST_BUILD_CONFIGURATION Release)
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(CTEST_DASHBOARD_ROOT "/tmp/RTK_dashboard")
set(dashboard_binary_name "RTK_lin64_gcc_cuda_system_itk4_simplertk")
set(ENV{ITK_DIR} "/home/srit/src/itk/lin64-dg")
set(ENV{CUDA_BIN_PATH} "/usr/bin")
set(ENV{CUDA_LIB_PATH} "/usr/lib64")
set(CTEST_BUILD_FLAGS -j12)

set(ENV{LD_LIBRARY_PATH} "/usr/lib64:$ENV{LD_LIBRARY_PATH}")
set(CONFIGURE_OPTIONS
   -DRTK_USE_CUDA:BOOL=ON
   -DCUDA_CUDA_LIBRARY:PATH=/usr/lib64/libcuda.so
   -DBUILD_SIMPLERTK:BOOL=ON
   -DCUDA_NVCC_FLAGS:STRING=-std=c++11
)

include("${CTEST_SCRIPT_DIRECTORY}/rtk_common.cmake")

