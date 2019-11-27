# client maintainer: simon.rit@creatis.insa-lyon.fr
set(CTEST_SITE "russula.clb")
set(CTEST_BUILD_NAME "Linux-64bit-gcc-itk-v4.12.0")
set(CTEST_BUILD_CONFIGURATION Release)
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(dashboard_binary_name "RTK_lin64_gcc_lin64_itk-v4.12.0")
set(ENV{ITK_DIR} "/home/srit/src/itk/lin64-itk-v4.12.0")
set(CTEST_BUILD_FLAGS -j16)
set(CTEST_CUSTOM_WARNING_EXCEPTION ${CTEST_CUSTOM_WARNING_EXCEPTION}
    ".* warning: dynamic exception specifications are deprecated in C\+\+11.*")
include("${CTEST_SCRIPT_DIRECTORY}/rtk_common.cmake")

