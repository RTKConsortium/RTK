# client maintainer: simon.rit@creatis.insa-lyon.fr
set(CTEST_SITE "linux15.dg.creatis.insa-lyon.fr")
set(CTEST_BUILD_NAME "Linux-64bit-gcc-cuda-system-itk4-nofftw")
set(CTEST_BUILD_CONFIGURATION Release)
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(CTEST_DASHBOARD_ROOT "/tmp/RTK_dashboard")
set(dashboard_binary_name "RTK_lin64_gcc_cuda_system_itk4_nofftw")
set(ENV{ITK_DIR} "/home/srit/src/itk/lin64-dg-nofftw")
set(ENV{CUDA_BIN_PATH} "/usr/bin")
set(ENV{CUDA_LIB_PATH} "/usr/lib64")
set(CTEST_BUILD_FLAGS -j12)

set(ENV{LD_LIBRARY_PATH} "/usr/lib64:$ENV{LD_LIBRARY_PATH}")
set(CONFIGURE_OPTIONS
   -DRTK_USE_CUDA:BOOL=ON
   -DCUDA_CUDA_LIBRARY:PATH=/usr/lib64/libcuda.so
   -DCUDA_NVCC_FLAGS:STRING=-std=c++11
  )

include("${CTEST_SCRIPT_DIRECTORY}/rtk_common.cmake")

