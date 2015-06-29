# client maintainer: simon.rit@creatis.insa-lyon.fr
set(CTEST_SITE "linux15.dg.creatis.insa-lyon.fr")
set(CTEST_BUILD_NAME "Linux-64bit-gcc-cuda70-itk4-sharedlibs")
set(CTEST_BUILD_CONFIGURATION Release)
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(CTEST_DASHBOARD_ROOT "/tmp/RTK_dashboard")
set(dashboard_binary_name "RTK_lin64_gcc_cuda70_itk4_sharedlibs")
set(ENV{ITK_DIR} "/home/srit/src/itk/lin64-dg-sharedlibs")
set(ENV{CUDA_BIN_PATH} "/home/srit/Download/cuda70/bin")
set(ENV{CUDA_LIB_PATH} "/usr/lib64/nvidia")
set(CTEST_BUILD_FLAGS -j8)

set(CONFIGURE_OPTIONS
   -DCUDA_CUDA_LIBRARY:PATH=/usr/lib64/nvidia/libcuda.so.1
   -DBUILD_SHARED_LIBS:BOOL=ON
  )

include("${CTEST_SCRIPT_DIRECTORY}/rtk_common.cmake")

