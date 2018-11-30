# client maintainer: simon.rit@creatis.insa-lyon.fr
set(CTEST_SITE "russula.clb")
set(CTEST_BUILD_NAME "Linux-64bit-gcc-shared-itk-v4.13.1-nocuda")
set(CTEST_BUILD_CONFIGURATION Release)
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(dashboard_binary_name "RTK_lin64_gcc_shared_lin64_itk-v4.13.1-nocuda")
set(ENV{ITK_DIR} "/home/srit/src/itk/lin64-itk-v4.13.1-SharedLibs-Release")
set(CTEST_BUILD_FLAGS -j16)
set(CONFIGURE_OPTIONS -DRTK_USE_CUDA=FALSE) 
include("${CTEST_SCRIPT_DIRECTORY}/rtk_common.cmake")

