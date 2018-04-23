# client maintainer: simon.rit@creatis.insa-lyon.fr
set(CTEST_SITE "linux15.dg.creatis.insa-lyon.fr")
set(CTEST_BUILD_NAME "Linux-64bit-gcc-cuda-system-itk4")
set(CTEST_BUILD_CONFIGURATION Release)
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(CTEST_DASHBOARD_ROOT "/tmp/RTK_dashboard")
set(dashboard_binary_name "RTK_lin64_gcc_cuda_system_itk4")
set(ENV{ITK_DIR} "/home/srit/src/itk/lin64-dg")
set(CTEST_BUILD_FLAGS -j12)
set(ENV{CXXFLAGS} "-fPIC -std=c++11")

include("${CTEST_SCRIPT_DIRECTORY}/rtk_common.cmake")

